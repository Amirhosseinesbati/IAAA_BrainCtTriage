"""
model.py — HRNet heatmap model for MLS keypoint regression.

Uses an HRNet backbone from timm with a lightweight heatmap prediction head.
Supports configurable backbone size (HRNet-W32 / HRNet-W18) and input channels.

Architecture:
    Input:  (B, C, H, W)   — C=1 or 3 (windowed CT)
    Backbone: HRNet (timm) — outputs multi-scale feature maps
    Head: Conv2d(3, 3)     — predicts 3 heatmap channels at 1/4 resolution
    Output: (B, 3, H/4, W/4) — Gaussian heatmaps for 3 keypoints
"""

from __future__ import annotations

import logging
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Registry of supported backbones and their feature dimensions
HRNET_CONFIG: Dict[str, Dict] = {
    "hrnet_w32": {
        "feature_dim": 128,  # stage 1 output channels (1/4 resolution)
        "description": "HRNet-W32 (higher accuracy, ~28.5M params)",
    },
    "hrnet_w18": {
        "feature_dim": 128,  # stage 1 output channels (1/4 resolution)
        "description": "HRNet-W18 (faster/lighter, ~21.3M params)",
    },
}


class HeatmapHead(nn.Module):
    """
    Lightweight heatmap prediction head.

    Takes feature maps from the HRNet backbone and produces K heatmap
    channels at the same spatial resolution.

    Architecture:
        Conv2d(C, 64, 3) → BatchNorm → ReLU → [Dropout2d] → Conv2d(64, K, 1)

    The optional Dropout2d provides regularization, which is important on
    small training sets (it is identity in eval mode, so it has no effect
    during inference).
    """

    def __init__(self, in_channels: int, num_keypoints: int = 3, dropout: float = 0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)  # identity when dropout == 0
        self.conv2 = nn.Conv2d(64, num_keypoints, kernel_size=1)

        # Initialize the final conv layer with small weights
        nn.init.normal_(self.conv2.weight, mean=0.0, std=0.001)
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        return x


class HRNetHeatmapModel(nn.Module):
    """
    HRNet backbone + heatmap head for keypoint regression.

    Args:
        backbone_name: Name of the HRNet variant ('hrnet_w32' or 'hrnet_w18').
        in_channels: Number of input image channels (1 or 3).
        num_keypoints: Number of keypoint heatmap channels (default 3).
        pretrained: Whether to load pretrained ImageNet weights.
        head_dropout: Dropout2d probability in the heatmap head
            (0 disables it; identity in eval mode).
        use_ordinal_aux_head: Attach a training-only pooled-feature head for
            the three official MLS boundaries. Historical inference is unchanged.
    """

    def __init__(
        self,
        backbone_name: str = "hrnet_w32",
        in_channels: int = 3,
        num_keypoints: int = 3,
        pretrained: bool = True,
        head_dropout: float = 0.0,
        use_selector: bool = False,
        selector_head_mode: Literal["single", "dual"] = "single",
        use_ordinal_aux_head: bool = False,
        use_reference_refinement: bool = False,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.in_channels = in_channels
        self.num_keypoints = num_keypoints
        self.head_dropout = head_dropout
        self.use_selector = use_selector
        self.selector_head_mode = selector_head_mode
        self.use_ordinal_aux_head = use_ordinal_aux_head
        self.use_reference_refinement = use_reference_refinement
        if use_reference_refinement and (num_keypoints != 3 or not use_selector):
            raise ValueError('Reference refinement requires three-point multitask model')
        if selector_head_mode not in {"single", "dual"}:
            raise ValueError(f"Unsupported selector_head_mode: {selector_head_mode}")

        if backbone_name not in HRNET_CONFIG:
            raise ValueError(
                f"Unknown backbone '{backbone_name}'. "
                f"Available: {list(HRNET_CONFIG.keys())}"
            )

        # Build HRNet backbone using timm
        import timm

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,  # return list of feature maps
            out_indices=(1,),    # stage 1 → 1/4 resolution feature map (128×128 for 512 input)
        )

        feat_dim = self.backbone.feature_info.channels()[0]
        logger.info(
            f"HRNetHeatmapModel: backbone={backbone_name}, "
            f"in_channels={in_channels}, feat_dim={feat_dim}, "
            f"pretrained={pretrained}, head_dropout={head_dropout}"
        )

        # Handle mismatched input channels
        if in_channels != 3:
            self._adapt_input_channels()

        # Heatmap prediction head
        self.head = HeatmapHead(feat_dim, num_keypoints, dropout=head_dropout)
        self.selector_head = None
        if use_selector:
            self.selector_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=max(0.1, head_dropout)),
                nn.Linear(64, 2 if selector_head_mode == "dual" else 1),
            )
        self.ordinal_aux_head = None
        if use_ordinal_aux_head:
            self.ordinal_aux_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(feat_dim, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(p=max(0.1, head_dropout)),
                nn.Linear(64, 3),
            )

        self.outer_refinement = None
        if use_reference_refinement:
            from .reference_refinement import ReferenceConditionedOuterHead
            self.outer_refinement = ReferenceConditionedOuterHead(feat_dim)

    def _heatmaps_from_features(self, features: torch.Tensor) -> torch.Tensor:
        coarse = self.head(features)
        if self.outer_refinement is None:
            return coarse
        return self.outer_refinement(features, coarse)

    def _adapt_input_channels(self) -> None:
        """
        Replace the first convolution layer to accept `in_channels`.

        When in_channels != 3 (e.g., single-channel CT), we replace the
        first Conv2d and initialize by averaging the pretrained RGB weights.
        """
        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            self.in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )

        if self.in_channels == 1:
            # Average RGB weights → single channel
            new_conv.weight.data = old_conv.weight.data.mean(dim=1, keepdim=True)
        elif self.in_channels < 3:
            # Take first `in_channels` from RGB
            new_conv.weight.data = old_conv.weight.data[:, :self.in_channels]
        else:
            # For >3 channels, repeat the RGB weights
            repeats = self.in_channels // 3 + 1
            repeated = old_conv.weight.data.repeat(1, repeats, 1, 1)
            new_conv.weight.data = repeated[:, :self.in_channels]

        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data

        self.backbone.conv1 = new_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor (B, C, H, W) — C = in_channels.

        Returns:
            Heatmap tensor (B, K, H/4, W/4) where K = num_keypoints.
        """
        features = self.backbone(x)  # list of feature maps
        feat_1_4 = features[0]       # highest-res feature map (1/4 scale)
        heatmaps = self._heatmaps_from_features(feat_1_4)
        return heatmaps

    def forward_multitask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return heatmap logits and an explicit target-slice logit.

        ``forward`` keeps the historical heatmap-only contract so old
        checkpoints and the current submission loader remain compatible.
        """
        if self.selector_head is None:
            raise RuntimeError("Model was created without use_selector=True")
        features = self.backbone(x)
        feat_1_4 = features[0]
        selector_logits = self.selector_head(feat_1_4)
        if self.selector_head_mode == "single":
            selector_logits = selector_logits.squeeze(1)
        return self._heatmaps_from_features(feat_1_4), selector_logits

    def forward_selector_only_detached_backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Return selector logits while isolating a selector-only auxiliary update.

        The regular :meth:`forward_multitask` path remains the sole route for
        geometry, presence, and ordinary peak-selector gradients.  This method
        is intentionally restricted to a supplementary rank update: it obtains
        frozen backbone features under ``no_grad`` while temporarily preventing
        BatchNorm buffer updates.  The backbone preserves its caller's
        train/eval semantics, avoiding an auxiliary-only feature-distribution
        change; the selector head stays in its caller's mode, so that head
        alone can receive the ranking gradient.
        """
        if self.selector_head is None:
            raise RuntimeError("Model was created without use_selector=True")

        batch_norm_modules = tuple(
            module for module in self.backbone.modules()
            if isinstance(module, nn.modules.batchnorm._BatchNorm)
        )
        tracking_states = tuple(module.track_running_stats for module in batch_norm_modules)
        try:
            # In training mode BatchNorm normally both uses batch statistics and
            # updates long-lived running buffers.  Keep the first behaviour so
            # the selector sees the ordinary training distribution, but disable
            # the second so this rank-only call cannot alter the backbone state.
            for module in batch_norm_modules:
                module.track_running_stats = False
            with torch.no_grad():
                feat_1_4 = self.backbone(x)[0]
        finally:
            for module, was_tracking in zip(batch_norm_modules, tracking_states):
                module.track_running_stats = was_tracking

        selector_logits = self.selector_head(feat_1_4.detach())
        if self.selector_head_mode == "single":
            selector_logits = selector_logits.squeeze(1)
        return selector_logits

    def forward_multitask_extended(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return historical outputs plus optional training-only ordinal logits.

        Keeping :meth:`forward_multitask` unchanged preserves the exact inference
        contract for every historical checkpoint and submission runtime.
        """
        if self.selector_head is None:
            raise RuntimeError("Model was created without use_selector=True")
        features = self.backbone(x)
        feat_1_4 = features[0]
        selector_logits = self.selector_head(feat_1_4)
        if self.selector_head_mode == "single":
            selector_logits = selector_logits.squeeze(1)
        ordinal_logits: torch.Tensor | None = None
        if self.ordinal_aux_head is not None:
            ordinal_logits = self.ordinal_aux_head(feat_1_4)
        return self._heatmaps_from_features(feat_1_4), selector_logits, ordinal_logits

    @torch.no_grad()
    def predict_heatmaps(
        self, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Inference-ready forward pass (no grad tracking).

        Args:
            x: Input tensor (B, C, H, W).

        Returns:
            Heatmap tensor (B, K, H/4, W/4).
        """
        self.eval()
        return self.forward(x)
