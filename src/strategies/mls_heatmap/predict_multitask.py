"""CUDA-only study-level inference for multitask MLS checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.input_contract import create_study_windowed_input
from src.strategies.mls_heatmap.utils import (
    compute_mls_from_keypoints,
    decode_heatmap_dark_batch,
)


@dataclass(frozen=True)
class SliceMLSPrediction:
    index: int
    selector_probability: float
    mls_mm: float
    heatmap_peak: float
    peak_probability: float | None = None


def _rank_probability(item: SliceMLSPrediction) -> float:
    """Use the dedicated peak head when present, preserving old checkpoints/tests."""
    if item.peak_probability is None:
        return item.selector_probability
    return item.peak_probability


def split_selector_logits(
    selector_logits: torch.Tensor,
    selector_head_mode: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target-presence and peak-severity logits for either checkpoint schema."""
    if selector_head_mode == "dual":
        if selector_logits.ndim != 2 or selector_logits.shape[1] != 2:
            raise ValueError(
                f"Dual selector expected [batch, 2] logits, got {tuple(selector_logits.shape)}"
            )
        return selector_logits[:, 0], selector_logits[:, 1]
    if selector_logits.ndim != 1:
        raise ValueError(
            f"Single selector expected [batch] logits, got {tuple(selector_logits.shape)}"
        )
    return selector_logits, selector_logits


def fuse_horizontal_flip_tta_probabilities(
    heatmap_logits: torch.Tensor,
    selector_logits: torch.Tensor,
    flipped_heatmap_logits: torch.Tensor,
    flipped_selector_logits: torch.Tensor,
    selector_head_mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fuse original and left-right-reflected views in the original geometry.

    Heatmaps are converted to per-keypoint spatial probabilities *before*
    averaging.  The reflected probabilities are unflipped on their width axis,
    so DARK decoding always sees coordinates in the original input frame.
    Selector heads have no spatial coordinate, hence their sigmoid
    probabilities (not logits) are averaged.  This function intentionally
    makes no threshold or aggregation decision; it is safe to use only as a
    separately audited inference-time factor.
    """
    if heatmap_logits.shape != flipped_heatmap_logits.shape:
        raise ValueError("flip-TTA heatmap tensors must have identical shapes")
    if heatmap_logits.ndim != 4:
        raise ValueError("flip-TTA heatmaps must have shape [batch, keypoint, height, width]")
    if selector_logits.shape != flipped_selector_logits.shape:
        raise ValueError("flip-TTA selector tensors must have identical shapes")

    original_probability = torch.softmax(heatmap_logits.flatten(2), dim=-1).reshape_as(
        heatmap_logits
    )
    reflected_probability = torch.softmax(
        flipped_heatmap_logits.flatten(2), dim=-1
    ).reshape_as(flipped_heatmap_logits)
    fused_heatmap_probability = 0.5 * (
        original_probability + torch.flip(reflected_probability, dims=(-1,))
    )

    original_target, original_peak = split_selector_logits(
        selector_logits, selector_head_mode,
    )
    reflected_target, reflected_peak = split_selector_logits(
        flipped_selector_logits, selector_head_mode,
    )
    fused_target_probability = 0.5 * (
        torch.sigmoid(original_target) + torch.sigmoid(reflected_target)
    )
    fused_peak_probability = 0.5 * (
        torch.sigmoid(original_peak) + torch.sigmoid(reflected_peak)
    )
    return fused_heatmap_probability, fused_target_probability, fused_peak_probability


def load_multitask_model(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[HRNetHeatmapModel, MLSHeatmapConfig]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Multitask MLS inference is CUDA-only; CPU fallback is forbidden")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = MLSHeatmapConfig.model_validate(checkpoint["config"])
    if not config.use_selector:
        raise ValueError("Checkpoint does not contain an explicit MLS selector")
    model = HRNetHeatmapModel(
        backbone_name=config.backbone,
        in_channels=config.input_channels,
        num_keypoints=3,
        pretrained=False,
        head_dropout=config.head_dropout,
        use_selector=True,
        selector_head_mode=config.selector_head_mode,
        use_ordinal_aux_head=config.use_ordinal_aux_head,
        use_reference_refinement=config.use_reference_refinement,
    )
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device).eval()
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("CUDA guard failed while loading MLS model")
    return model, config


@torch.inference_mode()
def predict_study_slices(
    study_dir: str | Path,
    model: HRNetHeatmapModel,
    config: MLSHeatmapConfig,
    device: torch.device,
    *,
    batch_size: int = 6,
) -> list[SliceMLSPrediction]:
    if device.type != "cuda":
        raise RuntimeError("MLS model inference must run on CUDA")
    reader = BrainDicomReader(str(study_dir)).load_and_sort()
    return predict_reader_slices(
        reader, model, config, device, batch_size=batch_size,
    )


@torch.inference_mode()
def predict_reader_slices(
    reader: BrainDicomReader,
    model: HRNetHeatmapModel,
    config: MLSHeatmapConfig,
    device: torch.device,
    *,
    batch_size: int = 6,
    horizontal_flip_tta: bool = False,
) -> list[SliceMLSPrediction]:
    """Decode one already-loaded study without duplicating DICOM I/O."""
    if device.type != "cuda":
        raise RuntimeError("MLS model inference must run on CUDA")
    volume = reader.get_3d_volume_hu()
    original_height = int(volume.shape[0])
    effective_spacing = float(reader.metadata["spacing_x"]) * (
        original_height / config.image_size
    )
    predictions: list[SliceMLSPrediction] = []
    channels = int(model.in_channels)
    for start in range(0, volume.shape[2], batch_size):
        end = min(start + batch_size, volume.shape[2])
        tensors: list[torch.Tensor] = []
        for index in range(start, end):
            windowed = create_study_windowed_input(volume, index, channels)
            tensor = torch.from_numpy(windowed).float().unsqueeze(0)
            if tensor.shape[-2:] != (config.image_size, config.image_size):
                tensor = F.interpolate(
                    tensor, size=(config.image_size, config.image_size),
                    mode="bilinear", align_corners=False,
                )
            tensors.append(tensor)
        inputs = torch.cat(tensors).to(device, non_blocking=True)
        heatmap_logits, selector_logits = model.forward_multitask(inputs)
        if not torch.isfinite(heatmap_logits).all() or not torch.isfinite(selector_logits).all():
            raise FloatingPointError("Non-finite CUDA output during MLS study inference")
        if horizontal_flip_tta:
            flipped_heatmap_logits, flipped_selector_logits = model.forward_multitask(
                torch.flip(inputs, dims=(-1,))
            )
            if not torch.isfinite(flipped_heatmap_logits).all() or not torch.isfinite(flipped_selector_logits).all():
                raise FloatingPointError("Non-finite reflected CUDA output during MLS flip-TTA")
            spatial_probabilities, selector_probability_tensor, peak_probability_tensor = (
                fuse_horizontal_flip_tta_probabilities(
                    heatmap_logits,
                    selector_logits,
                    flipped_heatmap_logits,
                    flipped_selector_logits,
                    config.selector_head_mode,
                )
            )
        else:
            spatial_probabilities = torch.softmax(heatmap_logits.flatten(2), dim=-1).reshape_as(
                heatmap_logits
            )
            target_logits, peak_logits = split_selector_logits(
                selector_logits, config.selector_head_mode,
            )
            selector_probability_tensor = torch.sigmoid(target_logits)
            peak_probability_tensor = torch.sigmoid(peak_logits)
        coordinates, peaks = decode_heatmap_dark_batch(
            spatial_probabilities.cpu(), spatial_probabilities.shape[-1], config.image_size,
        )
        selector_probabilities = selector_probability_tensor.cpu().numpy()
        peak_probabilities = peak_probability_tensor.cpu().numpy()
        for offset, keypoints in enumerate(coordinates):
            mls = 0.0
            if (keypoints[:, 0] >= 0).all():
                mls = float(compute_mls_from_keypoints(keypoints, effective_spacing))
            predictions.append(SliceMLSPrediction(
                index=start + offset,
                selector_probability=float(selector_probabilities[offset]),
                peak_probability=float(peak_probabilities[offset]),
                mls_mm=mls,
                heatmap_peak=float(np.min(peaks[offset])),
            ))
    return predictions


def aggregate_study_mls(
    predictions: list[SliceMLSPrediction],
    *,
    selector_threshold: float = 0.5,
    top_k: int = 3,
    aggregation: str = "p90",
    relative_ratio: float = 0.3,
    aggregation_quantile: float = 0.75,
    probability_weighted: bool = False,
    anchor_window_radius: int = 2,
    min_active_slices: int = 1,
    heatmap_guard_ratio: float = 0.0,
    negative_value: float = 0.1,
) -> float:
    if not predictions:
        return negative_value
    gate_ranked = sorted(
        predictions, key=lambda item: item.selector_probability, reverse=True,
    )
    if gate_ranked[0].selector_probability < selector_threshold:
        return negative_value
    if sum(item.selector_probability >= selector_threshold for item in predictions) < min_active_slices:
        return negative_value
    if aggregation in {
        "relative_component", "anchor_window", "joint_component", "severity_window",
    }:
        ordered = sorted(predictions, key=lambda item: item.index)
        rank_probabilities = np.asarray(
            [_rank_probability(item) for item in ordered], dtype=float,
        )
        heatmap_peaks = np.asarray([item.heatmap_peak for item in ordered], dtype=float)
        if aggregation == "joint_component":
            peak_scale = heatmap_peaks / max(float(heatmap_peaks.max()), 1e-8)
            component_scores = rank_probabilities * np.sqrt(np.maximum(peak_scale, 0.0))
        elif aggregation == "severity_window":
            peak_scale = heatmap_peaks / max(float(heatmap_peaks.max()), 1e-8)
            values = np.asarray([item.mls_mm for item in ordered], dtype=float)
            clipped_values = np.clip(values, 0.0, 30.0)
            value_scale = clipped_values / max(float(clipped_values.max()), 1e-8)
            component_scores = rank_probabilities * np.sqrt(
                np.maximum(peak_scale, 0.0)
            ) * np.sqrt(np.maximum(value_scale, 0.0))
        else:
            component_scores = rank_probabilities
        anchor = int(np.argmax(component_scores))
        if aggregation in {"relative_component", "joint_component"}:
            active = component_scores >= component_scores[anchor] * relative_ratio
            left = anchor
            right = anchor
            while left > 0 and active[left - 1]:
                left -= 1
            while right + 1 < len(ordered) and active[right + 1]:
                right += 1
        else:
            left = max(0, anchor - anchor_window_radius)
            right = min(len(ordered) - 1, anchor + anchor_window_radius)
        selected = ordered[left:right + 1]
        if heatmap_guard_ratio > 0:
            selected_peaks = np.asarray([item.heatmap_peak for item in selected], dtype=float)
            peak_gate = float(selected_peaks.max()) * heatmap_guard_ratio
            guarded = [item for item in selected if item.heatmap_peak >= peak_gate]
            if guarded:
                selected = guarded
        values = np.asarray([item.mls_mm for item in selected], dtype=float)
        if probability_weighted:
            weights = np.asarray([_rank_probability(item) for item in selected], dtype=float)
            order = np.argsort(values)
            ordered_values = values[order]
            ordered_weights = np.maximum(weights[order], 1e-8)
            cutoff = aggregation_quantile * float(ordered_weights.sum())
            index = min(
                int(np.searchsorted(np.cumsum(ordered_weights), cutoff)),
                len(ordered_values) - 1,
            )
            return float(ordered_values[index])
        return float(np.quantile(values, aggregation_quantile))
    ranked = sorted(predictions, key=_rank_probability, reverse=True)
    selected = ranked[:top_k]
    values = np.asarray([item.mls_mm for item in selected], dtype=float)
    if aggregation == "median":
        return float(np.median(values))
    if aggregation == "p90":
        return float(np.percentile(values, 90))
    if aggregation == "max":
        return float(values.max())
    if aggregation == "quantile":
        if probability_weighted:
            weights = np.asarray([_rank_probability(item) for item in selected], dtype=float)
            order = np.argsort(values)
            ordered_values = values[order]
            ordered_weights = np.maximum(weights[order], 1e-8)
            cutoff = aggregation_quantile * float(ordered_weights.sum())
            index = min(
                int(np.searchsorted(np.cumsum(ordered_weights), cutoff)),
                len(ordered_values) - 1,
            )
            return float(ordered_values[index])
        return float(np.quantile(values, aggregation_quantile))
    raise ValueError(f"Unsupported MLS aggregation: {aggregation}")
