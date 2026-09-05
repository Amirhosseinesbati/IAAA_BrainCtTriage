"""Single-model MLS inference using heatmap confidence for slice selection."""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from src.config import MLS_CHECKPOINTS_DIR
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.strategies.config_models import MLSHeatmapConfig
from src.strategies.mls_heatmap.model import HRNetHeatmapModel
from src.strategies.mls_heatmap.input_contract import (
    create_study_windowed_input,
    create_windowed_input,
)
from src.strategies.mls_heatmap.utils import decode_heatmap_dark_batch, compute_mls_from_keypoints


def _resolve_checkpoint_path(path: Optional[str] = None) -> str:
    resolved = path or os.environ.get("MLS_HEATMAP_MODEL_PATH")
    resolved = resolved or str(MLS_CHECKPOINTS_DIR / "mls_heatmap" / "mls_heatmap_best.pth")
    if not os.path.isfile(resolved):
        raise FileNotFoundError(
            f"Missing MLS heatmap checkpoint: {resolved}. "
            "Train mls_heatmap or set MLS_HEATMAP_MODEL_PATH."
        )
    return resolved


def _load_heatmap_model(path: str, config: MLSHeatmapConfig, device: torch.device) -> HRNetHeatmapModel:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    saved = checkpoint.get("config", {}) if isinstance(checkpoint, dict) else {}
    if saved.get("use_reference_refinement", False) or any(
        key.startswith("outer_refinement.") for key in state
    ):
        raise ValueError("Reference refinement checkpoints require load_multitask_model")
    model = HRNetHeatmapModel(
        backbone_name=saved.get("backbone", config.backbone),
        in_channels=int(saved.get("input_channels", config.input_channels)),
        num_keypoints=3, pretrained=False,
        head_dropout=float(saved.get("head_dropout", config.head_dropout)),
    )
    model.load_state_dict(state, strict=False)
    model.in_channels = int(saved.get("input_channels", config.input_channels))
    return model.to(device).eval()


def _create_windowed_input(hu_image: np.ndarray, input_channels: int = 3) -> np.ndarray:
    """Compatibility alias for central-slice input construction."""
    return create_windowed_input(hu_image, input_channels)


def _create_3channel_window(hu_image: np.ndarray) -> np.ndarray:
    """Compatibility alias for callers that only need window construction."""
    return _create_windowed_input(hu_image, 3)


def _run_pipeline(
    heatmap_model: HRNetHeatmapModel,
    image_hu: np.ndarray,
    spacing_x: float,
    config: MLSHeatmapConfig,
    device: torch.device,
) -> float:
    """Decode all slices, select confident candidates and aggregate MLS."""
    channels = int(getattr(heatmap_model, "in_channels", config.input_channels))
    effective_spacing = float(spacing_x) * int(image_hu.shape[0]) / config.image_size
    candidates: list[tuple[float, float]] = []

    for start in range(0, image_hu.shape[2], 16):
        images = []
        for z in range(start, min(start + 16, image_hu.shape[2])):
            tensor = torch.from_numpy(
                create_study_windowed_input(image_hu, z, channels)
            ).float().unsqueeze(0)
            if tensor.shape[-2:] != (config.image_size, config.image_size):
                tensor = F.interpolate(tensor, size=(config.image_size, config.image_size), mode="bilinear", align_corners=False)
            images.append(tensor)
        batch = torch.cat(images).to(device)
        with torch.no_grad():
            heatmaps = heatmap_model(batch)
        coords, scores = decode_heatmap_dark_batch(heatmaps.cpu(), heatmaps.shape[-1], config.image_size)
        scores_np = scores.cpu().numpy() if torch.is_tensor(scores) else np.asarray(scores)
        for index, keypoints in enumerate(coords):
            if (keypoints[:, 0] >= 0).all():
                candidates.append((
                    float(compute_mls_from_keypoints(keypoints, effective_spacing)),
                    float(np.min(scores_np[index])),
                ))

    if not candidates:
        return 0.0
    selected = sorted(candidates, key=lambda item: item[1], reverse=True)[:config.top_k_slices]
    values = np.asarray([item[0] for item in selected], dtype=float)
    return float(np.percentile(values, 90) if config.aggregation == "p90" else values.max())


def predict_mls(study_dir: str, heatmap_model_path=None, config=None, device=None) -> float:
    predictor = MLSHeatmapPredictor(
        heatmap_model_path=heatmap_model_path, config=config, device=device,
    )
    reader = BrainDicomReader(study_dir).load_and_sort()
    return predictor.predict(reader)


def batch_predict_mls(study_dirs: list[str], **kwargs) -> list[float]:
    return [predict_mls(study_dir, **kwargs) for study_dir in study_dirs]


class MLSHeatmapPredictor:
    """Cached legacy-or-multitask MLS adapter used by the local UI."""

    def __init__(self, heatmap_model_path=None, config=None, device=None):
        self.device = device if isinstance(device, torch.device) else torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        checkpoint_path = _resolve_checkpoint_path(heatmap_model_path)
        saved = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        saved_config = saved.get("config", {}) if isinstance(saved, dict) else {}
        is_multitask = bool(saved_config.get("use_selector", False))
        self.is_multitask = is_multitask
        if is_multitask:
            from src.strategies.mls_heatmap.predict_multitask import load_multitask_model

            self.heatmap_model, checkpoint_config = load_multitask_model(
                checkpoint_path, self.device,
            )
            self.config = config or checkpoint_config
        else:
            self.config = config or MLSHeatmapConfig.model_validate(saved_config or {})
            self.heatmap_model = _load_heatmap_model(
                checkpoint_path, self.config, self.device,
            )

    def predict(self, reader) -> float:
        if self.is_multitask:
            from src.strategies.mls_heatmap.predict_multitask import (
                aggregate_study_mls,
                predict_reader_slices,
            )

            slices = predict_reader_slices(
                reader, self.heatmap_model, self.config, self.device,
                batch_size=self.config.batch_size,
            )
            return aggregate_study_mls(
                slices,
                selector_threshold=self.config.selector_threshold,
                top_k=self.config.top_k_slices,
                aggregation=self.config.aggregation,
                relative_ratio=self.config.selector_relative_ratio,
                aggregation_quantile=self.config.aggregation_quantile,
                probability_weighted=self.config.aggregation_probability_weighted,
                anchor_window_radius=self.config.anchor_window_radius,
                min_active_slices=self.config.min_active_slices,
                heatmap_guard_ratio=self.config.heatmap_guard_ratio,
                negative_value=self.config.negative_value_mm,
            )
        return _run_pipeline(
            self.heatmap_model, reader.get_3d_volume_hu(),
            reader.metadata["spacing_x"], self.config, self.device,
        )
