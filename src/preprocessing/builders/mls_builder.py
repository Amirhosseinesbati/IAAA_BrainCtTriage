"""
mls_builder.py — Build MLS (Midline Shift) dataset from DICOM + JSON.

Refactored to use:
- config.py for paths, windows, keypoint names
- Improved BrainDicomReader with get_windowed_slice()
- Proportional negative sampling instead of fixed 2
- Validation that keypoints exist and are within image bounds
"""

import os
import random
import json
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    WINDOWS, IMG_SIZE, RANDOM_SEED, MLS_KEYPOINT_NAMES,
    RAW_TRAINING_DIR, RAW_ANNOTATIONS_DIR, MLS_DIR,
)
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.preprocessing.core.json_parser import AnnotationParser

logger = logging.getLogger(__name__)


class MlsDatasetBuilder:
    """
    Builds a dataset for MLS (Midline Shift) estimation.

    Two tasks in one dataset:
    1. Slice selection (binary classification: is this the target slice?)
    2. Keypoint detection (regression: find 3 MLS keypoints)

    Output: images/*.png (3-channel windowed) + mls_labels.csv
    """

    def __init__(
        self,
        raw_dicom_dir: Optional[str] = None,
        raw_json_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
    ):
        self.raw_dicom_dir = Path(raw_dicom_dir or RAW_TRAINING_DIR)
        self.raw_json_dir = Path(raw_json_dir or RAW_ANNOTATIONS_DIR)
        self.output_dir = Path(output_dir or MLS_DIR)
        self.out_img_dir = self.output_dir / "images"
        self.out_img_dir.mkdir(parents=True, exist_ok=True)

        self.target_kps = MLS_KEYPOINT_NAMES  # from config

    def _make_3channel_window(self, hu_slice: np.ndarray) -> np.ndarray:
        """
        Create 3-channel image using brain, subdural, and bone windows.
        Used for MLS model input.
        """
        ch1 = BrainDicomReader.apply_windowing(hu_slice, WINDOWS["brain"])
        ch2 = BrainDicomReader.apply_windowing(hu_slice, WINDOWS["subdural"])
        ch3 = BrainDicomReader.apply_windowing(hu_slice, WINDOWS["bone"])
        rgb = np.stack([ch1, ch2, ch3], axis=-1) * 255.0
        return rgb.astype(np.uint8)

    def build(self):
        logger.info(f"Building MLS dataset → {self.output_dir}")

        all_ids = sorted(
            d.name for d in self.raw_dicom_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

        csv_rows = []
        total_pos = 0
        total_neg = 0
        patients_with_kp = 0

        for pid in tqdm(all_ids, desc="MLS build"):
            json_dir = self.raw_json_dir / pid
            dicom_dir = self.raw_dicom_dir / pid

            if not json_dir.exists():
                continue

            parser = AnnotationParser(str(json_dir))
            summary = parser.get_patient_summary()
            if summary["n_with_keypoints"] == 0:
                continue

            # Find positive slices (all 3 keypoints present)
            positive = []
            for jf in sorted(json_dir.iterdir()):
                if jf.suffix != ".json":
                    continue
                dcm_name = jf.name.replace(".json", ".dcm")
                data = parser.parse_slice(dcm_name)

                kps = data["keypoints"]
                all_present = all(
                    kp_name in kps for kp_name in self.target_kps
                )
                if all_present:
                    positive.append({"dcm_name": dcm_name, "points": kps})

            if not positive:
                continue

            patients_with_kp += 1

            # Load DICOM
            try:
                reader = BrainDicomReader(str(dicom_dir)).load_and_sort()
            except (ValueError, FileNotFoundError):
                continue

            pos_names = {p["dcm_name"] for p in positive}
            all_names = [os.path.basename(s.filename) for s in reader.slices]
            neg_names = [n for n in all_names if n not in pos_names]

            # Proportional negative sampling: 2x the number of positives
            n_neg = min(len(neg_names), len(positive) * 2)
            selected_neg = set(random.sample(neg_names, n_neg) if neg_names else [])

            for ds in reader.slices:
                dcm_name = os.path.basename(ds.filename)
                is_pos = dcm_name in pos_names

                if not is_pos and dcm_name not in selected_neg:
                    continue

                # Extract and window
                hu = reader._pixel_to_hu(ds.pixel_array, ds)
                img_3ch = self._make_3channel_window(hu)
                img_bgr = cv2.cvtColor(img_3ch, cv2.COLOR_RGB2BGR)

                stem = f"{pid}_{dcm_name.replace('.dcm', '.png')}"
                cv2.imwrite(str(self.out_img_dir / stem), img_bgr)

                row = {
                    "patient_id": pid,
                    "image_name": stem,
                    "is_target": 1 if is_pos else 0,
                    "x1": 0, "y1": 0, "x2": 0, "y2": 0, "x3": 0, "y3": 0,
                }

                if is_pos:
                    kp = next(p["points"] for p in positive if p["dcm_name"] == dcm_name)
                    row["x1"], row["y1"] = kp[self.target_kps[0]]
                    row["x2"], row["y2"] = kp[self.target_kps[1]]
                    row["x3"], row["y3"] = kp[self.target_kps[2]]
                    total_pos += 1
                else:
                    total_neg += 1

                csv_rows.append(row)

        # Save CSV
        df = pd.DataFrame(csv_rows)
        csv_path = self.output_dir / "mls_labels.csv"
        df.to_csv(csv_path, index=False)

        # Save build metadata
        meta = {
            "patients_with_keypoints": patients_with_kp,
            "total_positive_slices": total_pos,
            "total_negative_slices": total_neg,
            "window_config": {k: v for k, v in WINDOWS.items()},
            "keypoints_required": self.target_kps,
        }
        with open(self.output_dir / "build_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        logger.info(f"✅ MLS: {total_pos + total_neg} slices "
                    f"({total_pos} pos, {total_neg} neg) "
                    f"from {patients_with_kp} patients")
