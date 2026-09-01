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
from src.strategies.mls_heatmap.predict import _create_windowed_input
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
            windowed = _create_windowed_input(volume[:, :, index], channels)
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
        spatial_probabilities = torch.softmax(heatmap_logits.flatten(2), dim=-1).reshape_as(
            heatmap_logits
        )
        coordinates, peaks = decode_heatmap_dark_batch(
            spatial_probabilities.cpu(), spatial_probabilities.shape[-1], config.image_size,
        )
        selector_probabilities = torch.sigmoid(selector_logits).cpu().numpy()
        for offset, keypoints in enumerate(coordinates):
            mls = 0.0
            if (keypoints[:, 0] >= 0).all():
                mls = float(compute_mls_from_keypoints(keypoints, effective_spacing))
            predictions.append(SliceMLSPrediction(
                index=start + offset,
                selector_probability=float(selector_probabilities[offset]),
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
    ranked = sorted(predictions, key=lambda item: item.selector_probability, reverse=True)
    if ranked[0].selector_probability < selector_threshold:
        return negative_value
    if sum(item.selector_probability >= selector_threshold for item in predictions) < min_active_slices:
        return negative_value
    if aggregation in {
        "relative_component", "anchor_window", "joint_component", "severity_window",
    }:
        ordered = sorted(predictions, key=lambda item: item.index)
        probabilities = np.asarray([item.selector_probability for item in ordered], dtype=float)
        heatmap_peaks = np.asarray([item.heatmap_peak for item in ordered], dtype=float)
        if aggregation == "joint_component":
            peak_scale = heatmap_peaks / max(float(heatmap_peaks.max()), 1e-8)
            component_scores = probabilities * np.sqrt(np.maximum(peak_scale, 0.0))
        elif aggregation == "severity_window":
            peak_scale = heatmap_peaks / max(float(heatmap_peaks.max()), 1e-8)
            values = np.asarray([item.mls_mm for item in ordered], dtype=float)
            clipped_values = np.clip(values, 0.0, 30.0)
            value_scale = clipped_values / max(float(clipped_values.max()), 1e-8)
            component_scores = probabilities * np.sqrt(
                np.maximum(peak_scale, 0.0)
            ) * np.sqrt(np.maximum(value_scale, 0.0))
        else:
            component_scores = probabilities
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
            weights = np.asarray([item.selector_probability for item in selected], dtype=float)
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
            weights = np.asarray([item.selector_probability for item in selected], dtype=float)
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
