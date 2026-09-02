"""Five-checkpoint MLS runtime for the conservative three-member ensemble.

Fold0 and fold1 retain every baseline selector/ranking/heatmap value and blend
only the slice-level MLS measurement from their dual-selector replication at
alpha 0.10. Fold2 remains the strict Exp15r baseline. All neural-network
forward passes are CUDA-only; CPU work is limited to DICOM decoding and small
post-processing arrays.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

try:
    from .dicom_io import DicomStudy
except ImportError:  # pragma: no cover - flat leaderboard import
    from dicom_io import DicomStudy


IMAGE_SIZE = 512
WINDOWS = ((40.0, 80.0), (80.0, 200.0), (400.0, 1000.0))
REGRESSION_ALPHA = 0.10
MODEL_FILES = {
    "fold0": "fold0.pth",
    "fold0_regression": "fold0_regression.pth",
    "fold1": "fold1.pth",
    "fold1_regression": "fold1_regression.pth",
    "fold2": "fold2.pth",
}
EXPECTED_SELECTOR_MODES = {
    "fold0": "single",
    "fold0_regression": "dual",
    "fold1": "single",
    "fold1_regression": "dual",
    "fold2": "single",
}


class HeatmapHead(nn.Module):
    def __init__(self, in_channels: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout)
        self.conv2 = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.dropout(self.relu(self.bn1(self.conv1(value)))))


class HRNetMultitask(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        head_dropout: float,
        selector_head_mode: str = "single",
    ) -> None:
        super().__init__()
        if selector_head_mode not in {"single", "dual"}:
            raise ValueError(f"Unsupported selector_head_mode: {selector_head_mode}")
        import timm

        self.in_channels = 3
        self.selector_head_mode = selector_head_mode
        self.backbone = timm.create_model(
            backbone_name, pretrained=False, features_only=True, out_indices=(1,)
        )
        feature_dim = self.backbone.feature_info.channels()[0]
        self.head = HeatmapHead(feature_dim, dropout=head_dropout)
        self.selector_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(feature_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=max(0.1, head_dropout)),
            nn.Linear(64, 2 if selector_head_mode == "dual" else 1),
        )

    def forward_multitask(self, value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(value)[0]
        selector = self.selector_head(features)
        if self.selector_head_mode == "single":
            selector = selector.squeeze(1)
        return self.head(features), selector


@dataclass(frozen=True)
class SlicePrediction:
    index: int
    selector_probability: float
    peak_probability: float
    mls_mm: float
    heatmap_peak: float


def _window(image: np.ndarray, level: float, width: float) -> np.ndarray:
    lower = level - width / 2.0
    return np.clip((image - lower) / width, 0.0, 1.0).astype(np.float32)


def _input(image: np.ndarray) -> np.ndarray:
    return np.stack([_window(image, level, width) for level, width in WINDOWS], axis=0)


def _decode_one(heatmap: torch.Tensor, heatmap_size: int) -> tuple[float, float]:
    height, width = heatmap.shape
    scale = IMAGE_SIZE / heatmap_size
    maximum = heatmap.max()
    if maximum < 1e-8:
        return -1.0, -1.0
    flat_index = int(heatmap.argmax().item())
    y0, x0 = divmod(flat_index, width)
    if y0 in (0, height - 1) or x0 in (0, width - 1):
        return float(x0) * scale, float(y0) * scale
    gx = (heatmap[y0, x0 + 1] - heatmap[y0, x0 - 1]) / 2.0
    gy = (heatmap[y0 + 1, x0] - heatmap[y0 - 1, x0]) / 2.0
    hxx = heatmap[y0, x0 + 1] - 2.0 * heatmap[y0, x0] + heatmap[y0, x0 - 1]
    hyy = heatmap[y0 + 1, x0] - 2.0 * heatmap[y0, x0] + heatmap[y0 - 1, x0]
    hxy = (
        heatmap[y0 + 1, x0 + 1]
        - heatmap[y0 + 1, x0 - 1]
        - heatmap[y0 - 1, x0 + 1]
        + heatmap[y0 - 1, x0 - 1]
    ) / 4.0
    determinant = hxx * hyy - hxy * hxy
    if abs(float(determinant)) < 1e-12:
        return float(x0) * scale, float(y0) * scale
    dx = float((-(hyy * gx - hxy * gy) / determinant).item())
    dy = float((-(hxx * gy - hxy * gx) / determinant).item())
    x = (float(x0) + float(np.clip(dx, -0.5, 0.5))) * scale
    y = (float(y0) + float(np.clip(dy, -0.5, 0.5))) * scale
    return float(np.clip(x, 0, IMAGE_SIZE - 1)), float(np.clip(y, 0, IMAGE_SIZE - 1))


def _decode_batch(heatmaps: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    batch, keypoints = heatmaps.shape[:2]
    coordinates = np.zeros((batch, keypoints, 2), dtype=np.float32)
    peaks = np.zeros((batch, keypoints), dtype=np.float32)
    size = heatmaps.shape[-1]
    for row in range(batch):
        for keypoint in range(keypoints):
            value = heatmaps[row, keypoint]
            peaks[row, keypoint] = float(value.max().item())
            coordinates[row, keypoint] = _decode_one(value, size)
    return coordinates, peaks


def _measurement(keypoints: np.ndarray, spacing_x: float) -> float:
    first, second, outer = keypoints
    dx, dy = second - first
    denominator = float(np.sqrt(dx * dx + dy * dy))
    if denominator < 1e-8:
        return 0.0
    numerator = abs(dx * (first[1] - outer[1]) - (first[0] - outer[0]) * dy)
    return float(numerator / denominator * spacing_x)


def _blend_regression(
    baseline: list[SlicePrediction],
    challenger: list[SlicePrediction],
    alpha: float = REGRESSION_ALPHA,
) -> list[SlicePrediction]:
    if len(baseline) != len(challenger):
        raise RuntimeError("Baseline/challenger slice-count mismatch")
    output: list[SlicePrediction] = []
    for base, extra in zip(baseline, challenger, strict=True):
        if base.index != extra.index:
            raise RuntimeError("Baseline/challenger slice-index mismatch")
        output.append(
            SlicePrediction(
                index=base.index,
                selector_probability=base.selector_probability,
                peak_probability=base.peak_probability,
                mls_mm=float((1.0 - alpha) * base.mls_mm + alpha * extra.mls_mm),
                heatmap_peak=base.heatmap_peak,
            )
        )
    return output


def _aggregate(predictions: list[SlicePrediction]) -> float:
    if not predictions:
        return 0.1
    probabilities = np.asarray([row.selector_probability for row in predictions])
    if float(probabilities.max()) < 0.5 or int(np.count_nonzero(probabilities >= 0.5)) < 3:
        return 0.1
    peaks = np.asarray([row.heatmap_peak for row in predictions], dtype=np.float64)
    values = np.asarray([row.mls_mm for row in predictions], dtype=np.float64)
    ranks = np.asarray([row.peak_probability for row in predictions], dtype=np.float64)
    peak_scale = peaks / max(float(peaks.max()), 1e-8)
    clipped = np.clip(values, 0.0, 30.0)
    value_scale = clipped / max(float(clipped.max()), 1e-8)
    severity = ranks * np.sqrt(np.maximum(peak_scale, 0.0)) * np.sqrt(
        np.maximum(value_scale, 0.0)
    )
    anchor = int(np.argmax(severity))
    selected = predictions[max(0, anchor - 3) : min(len(predictions), anchor + 4)]
    selected_values = np.asarray([row.mls_mm for row in selected], dtype=np.float64)
    weights = np.maximum(
        np.asarray([row.peak_probability for row in selected], dtype=np.float64), 1e-8
    )
    order = np.argsort(selected_values)
    cutoff = 0.75 * float(weights.sum())
    index = min(int(np.searchsorted(np.cumsum(weights[order]), cutoff)), len(order) - 1)
    return float(selected_values[order][index])


def _split_selector(
    logits: torch.Tensor, mode: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if mode == "dual":
        if logits.ndim != 2 or logits.shape[1] != 2:
            raise RuntimeError(f"Invalid dual selector shape: {tuple(logits.shape)}")
        return logits[:, 0], logits[:, 1]
    if logits.ndim != 1:
        raise RuntimeError(f"Invalid single selector shape: {tuple(logits.shape)}")
    return logits, logits


class MLSEnsemblePredictor:
    def __init__(self, models_dir: str | Path, device: torch.device) -> None:
        if device.type != "cuda" or not torch.cuda.is_available():
            raise RuntimeError("MLS package requires CUDA; CPU fallback is forbidden")
        self.device = device
        self.models: dict[str, HRNetMultitask] = {}
        self.selector_modes: dict[str, str] = {}
        root = Path(models_dir)
        for label, filename in MODEL_FILES.items():
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = torch.load(path, map_location="cpu", weights_only=True)
            config = dict(payload["config"])
            if not bool(config.get("use_selector")):
                raise ValueError(f"MLS checkpoint lacks selector: {path}")
            mode = str(config.get("selector_head_mode", "single"))
            if mode != EXPECTED_SELECTOR_MODES[label]:
                raise ValueError(f"Unexpected selector mode for {label}: {mode}")
            model = HRNetMultitask(
                str(config["backbone"]),
                float(config.get("head_dropout", 0.0)),
                mode,
            )
            model.load_state_dict(payload["model_state_dict"], strict=True)
            model = model.to(device).eval()
            if next(model.parameters()).device.type != "cuda":
                raise RuntimeError(f"CUDA guard failed for {label}")
            self.models[label] = model
            self.selector_modes[label] = mode
            del payload
        if set(self.models) != set(MODEL_FILES):
            raise RuntimeError("MLS package must contain exactly five named checkpoints")

    @torch.inference_mode()
    def predict_detailed(self, study: DicomStudy, batch_size: int = 6) -> dict[str, Any]:
        volume = study.volume_hwd
        effective_spacing = study.spacing_x * (study.rows / float(IMAGE_SIZE))
        raw: dict[str, list[SlicePrediction]] = {label: [] for label in self.models}
        for start in range(0, volume.shape[2], batch_size):
            tensors = [
                torch.from_numpy(_input(volume[:, :, index])).unsqueeze(0)
                for index in range(start, min(volume.shape[2], start + batch_size))
            ]
            inputs = torch.cat(tensors).to(self.device, non_blocking=True)
            if inputs.shape[-2:] != (IMAGE_SIZE, IMAGE_SIZE):
                inputs = F.interpolate(
                    inputs,
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode="bilinear",
                    align_corners=False,
                )
            if inputs.device.type != "cuda":
                raise RuntimeError("MLS input tensor left CUDA")
            for label, model in self.models.items():
                heatmap_logits, selector_logits = model.forward_multitask(inputs)
                if heatmap_logits.device.type != "cuda" or selector_logits.device.type != "cuda":
                    raise RuntimeError(f"MLS output left CUDA: {label}")
                if not torch.isfinite(heatmap_logits).all() or not torch.isfinite(
                    selector_logits
                ).all():
                    raise FloatingPointError(f"Non-finite MLS output: {label}")
                probabilities = torch.softmax(
                    heatmap_logits.flatten(2), dim=-1
                ).reshape_as(heatmap_logits)
                coordinates, heatmap_peaks = _decode_batch(probabilities.cpu())
                presence_logits, peak_logits = _split_selector(
                    selector_logits, self.selector_modes[label]
                )
                presence = torch.sigmoid(presence_logits).cpu().numpy()
                peak = torch.sigmoid(peak_logits).cpu().numpy()
                for offset, keypoints in enumerate(coordinates):
                    measurement = (
                        _measurement(keypoints, effective_spacing)
                        if (keypoints[:, 0] >= 0).all()
                        else 0.0
                    )
                    raw[label].append(
                        SlicePrediction(
                            index=start + offset,
                            selector_probability=float(presence[offset]),
                            peak_probability=float(peak[offset]),
                            mls_mm=float(measurement),
                            heatmap_peak=float(np.min(heatmap_peaks[offset])),
                        )
                    )
        member_predictions = {
            "fold0": _blend_regression(raw["fold0"], raw["fold0_regression"]),
            "fold1": _blend_regression(raw["fold1"], raw["fold1_regression"]),
            "fold2": raw["fold2"],
        }
        member_values = {
            label: float(np.clip(_aggregate(rows), 0.0, 30.0))
            for label, rows in member_predictions.items()
        }
        ensemble = float(np.clip(np.median(list(member_values.values())), 0.0, 30.0))
        if not np.isfinite(ensemble):
            raise FloatingPointError("Non-finite MLS ensemble output")
        return {
            "ensemble": ensemble,
            "member_values": member_values,
            "raw_predictions": raw,
            "member_predictions": member_predictions,
        }

    def predict(self, study: DicomStudy, batch_size: int = 6) -> float:
        return float(self.predict_detailed(study, batch_size=batch_size)["ensemble"])
