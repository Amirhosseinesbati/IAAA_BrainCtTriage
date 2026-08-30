"""Self-contained five-fold detector + smooth-attention MIL fracture inference."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from torch import nn


class SmoothAttentionMIL(nn.Module):
    """Architecture matching the trained tiny sequence heads."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        attention_dim: int = 32,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attention_value = nn.Linear(hidden_dim, attention_dim)
        self.attention_gate = nn.Linear(hidden_dim, attention_dim)
        self.attention_score = nn.Linear(attention_dim, 1, bias=False)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Invalid MIL feature shape")
        hidden = self.encoder(features)
        gated = torch.tanh(self.attention_value(hidden)) * torch.sigmoid(
            self.attention_gate(hidden)
        )
        logits = self.attention_score(gated).squeeze(-1)
        weights = torch.softmax(logits, dim=0)
        pooled = torch.sum(weights[:, None] * hidden, dim=0)
        return self.classifier(pooled).squeeze(-1)


@dataclass(frozen=True)
class _Standardizer:
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray
    score_mean: float
    score_scale: float


def _score_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(clipped) - np.log1p(-clipped)


def _adjacent_pair(values: Iterable[float]) -> float:
    scores = np.asarray(list(values), dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise ValueError("Slice scores must be a non-empty finite vector")
    if scores.size < 2:
        return 0.0
    return float(np.sqrt(scores[:-1] * scores[1:]).max())


def _empirical_cdf(training: Iterable[float], value: float) -> float:
    training_array = np.sort(np.asarray(list(training), dtype=np.float64))
    if training_array.size == 0 or not np.isfinite(training_array).all():
        raise ValueError("Calibration scores must be non-empty and finite")
    if not math.isfinite(value):
        raise ValueError("Inference score must be finite")
    left = np.searchsorted(training_array, value, side="left")
    right = np.searchsorted(training_array, value, side="right")
    return float((0.5 * (left + right) + 0.5) / (training_array.size + 1.0))


def _threshold_to_probability(score: float, threshold: float) -> float:
    if not 0.0 < threshold < 1.0:
        raise ValueError("Decision threshold must be strictly inside (0, 1)")
    score = float(np.clip(score, 0.0, 1.0))
    if score < threshold:
        return 0.5 * score / threshold
    return 0.5 + 0.5 * (score - threshold) / (1.0 - threshold)


def bone_images_from_volume(
    volume_hu: np.ndarray,
    *,
    width: float = 1000.0,
    level: float = 400.0,
    jpeg_quality: int | None = 95,
) -> list[np.ndarray]:
    """Create ordered grayscale RGB images matching detector training."""
    import cv2

    if volume_hu.ndim != 3 or volume_hu.shape[2] < 1:
        raise ValueError("volume_hu must have shape [height, width, slices]")
    lower = level - width / 2.0
    images: list[np.ndarray] = []
    for index in range(volume_hu.shape[2]):
        gray = np.clip((volume_hu[:, :, index] - lower) / width, 0.0, 1.0)
        image = cv2.cvtColor((gray * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)
        if jpeg_quality is not None:
            success, encoded = cv2.imencode(
                ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
            )
            if not success:
                raise RuntimeError("Failed to reproduce fracture JPEG preprocessing")
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        images.append(image)
    return images


def _slice_max_score(result: object) -> float:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return 0.0
    confidence = boxes.conf.detach().float().cpu().numpy()
    return float(np.max(confidence)) if confidence.size else 0.0


class _FoldPredictor:
    def __init__(
        self,
        fold: dict[str, object],
        package_root: Path,
        device: torch.device,
    ) -> None:
        from ultralytics import YOLO

        detector_path = package_root / str(fold["detector"])
        self.detector = YOLO(str(detector_path))
        self.embedder = YOLO(str(detector_path))
        self.device = device
        self.reference_training_scores = list(fold["reference_training_scores"])
        self.candidate_training_scores = list(fold["candidate_training_scores"])
        self.candidate_weight = float(fold["candidate_weight"])
        self.models: list[SmoothAttentionMIL] = []
        self.standardizer: _Standardizer | None = None
        for relative in fold["mil_heads"]:
            payload = torch.load(
                package_root / str(relative), map_location="cpu", weights_only=True
            )
            config = payload["config"]
            model = SmoothAttentionMIL(**config).to(device).eval()
            model.load_state_dict(payload["state_dict"])
            self.models.append(model)
            raw = payload["standardizer"]
            standardizer = _Standardizer(
                embedding_mean=raw["embedding_mean"].numpy(),
                embedding_scale=raw["embedding_scale"].numpy(),
                score_mean=float(raw["score_mean"]),
                score_scale=float(raw["score_scale"]),
            )
            if self.standardizer is None:
                self.standardizer = standardizer
            else:
                np.testing.assert_array_equal(
                    self.standardizer.embedding_mean, standardizer.embedding_mean
                )
                np.testing.assert_array_equal(
                    self.standardizer.embedding_scale, standardizer.embedding_scale
                )
                if (
                    self.standardizer.score_mean != standardizer.score_mean
                    or self.standardizer.score_scale != standardizer.score_scale
                ):
                    raise RuntimeError("MIL seed standardizers differ")
        if len(self.models) != 3 or self.standardizer is None:
            raise RuntimeError("Each fold must contain exactly three MIL heads")

    def predict(
        self,
        images: list[np.ndarray],
        *,
        image_size: int,
        batch_size: int,
        confidence: float,
    ) -> dict[str, float]:
        common = {
            "source": images,
            "imgsz": image_size,
            "batch": min(batch_size, len(images)),
            "device": str(self.device),
            "verbose": False,
        }
        results = self.detector.predict(conf=confidence, **common)
        slice_scores = np.asarray(
            [_slice_max_score(result) for result in results], dtype=np.float32
        )
        embedded = self.embedder.embed(**common)
        embeddings = np.stack(
            [item.detach().float().cpu().numpy() for item in embedded], axis=0
        ).astype(np.float32)
        if slice_scores.shape != (len(images),) or embeddings.shape[0] != len(images):
            raise RuntimeError("Ultralytics returned the wrong number of slices")
        standardizer = self.standardizer
        normalized_embedding = (
            embeddings - standardizer.embedding_mean
        ) / standardizer.embedding_scale
        normalized_score = (
            _score_logit(slice_scores) - standardizer.score_mean
        ) / standardizer.score_scale
        features = torch.from_numpy(
            np.concatenate([normalized_embedding, normalized_score[:, None]], axis=1)
        ).to(self.device)
        with torch.no_grad():
            mil_score = float(
                np.mean(
                    [
                        torch.sigmoid(model(features)).detach().float().cpu().item()
                        for model in self.models
                    ]
                )
            )
        adjacent = _adjacent_pair(slice_scores)
        reference_cdf = _empirical_cdf(self.reference_training_scores, adjacent)
        candidate_cdf = _empirical_cdf(self.candidate_training_scores, mil_score)
        blend = (
            (1.0 - self.candidate_weight) * reference_cdf
            + self.candidate_weight * candidate_cdf
        )
        return {
            "adjacent_pair": adjacent,
            "mil_score": mil_score,
            "reference_cdf": reference_cdf,
            "candidate_cdf": candidate_cdf,
            "blend_score": blend,
        }


class FractureMILPredictor:
    """Five-fold ensemble with train-only CDF calibration and a fixed decision map."""

    def __init__(self, package_dir: str | Path, device: str = "cuda:0") -> None:
        package_root = Path(package_dir)
        manifest = json.loads((package_root / "manifest.json").read_text("utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("Unsupported fracture MIL package schema")
        self.image_size = int(manifest["inference"]["image_size"])
        self.batch_size = int(manifest["inference"]["batch_size"])
        self.confidence = float(manifest["inference"]["confidence"])
        self.decision_threshold = float(manifest["decision_calibration"]["threshold"])
        self.device = torch.device(device)
        self.folds = [
            _FoldPredictor(fold, package_root, self.device)
            for fold in manifest["folds"]
        ]
        if len(self.folds) != 5:
            raise RuntimeError("Fracture package must contain five folds")

    def predict_images(self, images: list[np.ndarray]) -> dict[str, object]:
        if not images:
            raise ValueError("At least one ordered slice image is required")
        fold_results = [
            fold.predict(
                images,
                image_size=self.image_size,
                batch_size=self.batch_size,
                confidence=self.confidence,
            )
            for fold in self.folds
        ]
        ensemble_score = float(np.mean([row["blend_score"] for row in fold_results]))
        return {
            "fracture_prob": _threshold_to_probability(
                ensemble_score, self.decision_threshold
            ),
            "ensemble_score": ensemble_score,
            "folds": fold_results,
        }

    def predict_volume(self, volume_hu: np.ndarray) -> dict[str, object]:
        return self.predict_images(bone_images_from_volume(volume_hu))
