"""Opt-in prototype: full-field outer heatmap refinement from predicted geometry.

No ground-truth geometry, image crop, resampling, or historical model mutation.
Coordinates use heatmap pixel centres (integer indices), not endpoint linspace.
"""
from __future__ import annotations

import torch
from torch import nn


def predicted_reference_fields(logits: torch.Tensor, min_length: float = 4.0):
    """Return parallel/perpendicular coordinate fields and valid-reference mask.

    Detached discrete peaks are deliberately identical in training and inference.
    Fields are normalized by image diagonal, not predicted line length: tiny
    predicted segments cannot amplify coordinates. Degenerate references use
    finite zero fields and must fall back to the unmodified coarse prediction.
    """
    if logits.ndim != 4 or logits.shape[1] != 3 or min_length <= 0:
        raise ValueError('Expected Bx3xHxW and positive minimum segment length')
    h, w = logits.shape[-2:]
    if h < 2 or w < 2:
        raise ValueError('Heatmap is too small')
    indices = logits[:, :2].detach().flatten(2).argmax(-1)
    points = torch.stack((indices.remainder(w), indices.div(w, rounding_mode='floor')), -1).to(logits.dtype)
    delta = points[:, 1] - points[:, 0]
    length = delta.square().sum(-1).sqrt()
    valid = (length >= min_length) & torch.isfinite(logits[:, :2]).flatten(1).all(1)
    unit = delta / length.clamp_min(min_length)[:, None]
    yy, xx = torch.meshgrid(torch.arange(h, device=logits.device, dtype=logits.dtype),
                            torch.arange(w, device=logits.device, dtype=logits.dtype), indexing='ij')
    dx = xx[None] - points[:, 0, 0, None, None]
    dy = yy[None] - points[:, 0, 1, None, None]
    ux, uy = unit[:, 0, None, None], unit[:, 1, None, None]
    scale = float((h*h + w*w)**.5)
    fields = torch.stack(((dx*ux + dy*uy)/scale, (-dx*uy + dy*ux)/scale), 1)
    fields = torch.where(valid[:, None, None, None], fields, torch.zeros_like(fields))
    return fields, valid


class ReferenceConditionedOuterHead(nn.Module):
    """Residual outer-point logits; endpoints are unchanged, initial residual zero."""
    def __init__(self, feature_channels: int, hidden_channels: int = 32, min_length: float = 4.0):
        super().__init__()
        if feature_channels < 1 or hidden_channels < 1 or min_length <= 0:
            raise ValueError('Invalid head dimensions')
        self.min_length = min_length
        self.refine = nn.Sequential(
            nn.Conv2d(feature_channels + 5, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, 1, 1),
        )
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)

    def forward(self, features: torch.Tensor, coarse: torch.Tensor):
        if features.shape[0] != coarse.shape[0] or features.shape[-2:] != coarse.shape[-2:]:
            raise ValueError('Features and heatmaps must be spatially aligned')
        fields, valid = predicted_reference_fields(coarse, self.min_length)
        # Bounded logit context; no labels and no gradient through the coarse
        # localization used for conditioning. Direct coarse output path remains.
        context = coarse.detach() - coarse.detach().mean((-2, -1), keepdim=True)
        residual = self.refine(torch.cat((features, context.tanh(), fields), dim=1))
        residual = torch.where(valid[:, None, None, None], residual, torch.zeros_like(residual))
        return torch.cat((coarse[:, :2], coarse[:, 2:3] + residual), dim=1)


class ReferenceConditionedMLSPrototype(nn.Module):
    """Explicit wrapper, not yet wired into release/trainer checkpoint loaders."""
    def __init__(self, base: nn.Module):
        super().__init__()
        if base.selector_head is None or base.num_keypoints != 3:
            raise ValueError('Requires three-point multitask base')
        self.base = base
        self.outer_refinement = ReferenceConditionedOuterHead(base.backbone.feature_info.channels()[0])

    def forward_multitask(self, images: torch.Tensor):
        features = self.base.backbone(images)[0]
        coarse = self.base.head(features)
        selector = self.base.selector_head(features)
        if self.base.selector_head_mode == 'single': selector = selector.squeeze(1)
        return self.outer_refinement(features, coarse), selector

    def forward(self, images: torch.Tensor):
        return self.forward_multitask(images)[0]
