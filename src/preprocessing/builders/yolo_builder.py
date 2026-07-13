"""
yolo_builder.py — Build YOLO fracture detection dataset from DICOM + JSON.

Refactored to use:
- config.py for paths, bone window definition, seed
- Improved BrainDicomReader for slice access
- config.WINDOWS["bone"] instead of hardcoded 1000/400
- Reproducible train/val split with saved patient lists
"""

import os
import json
import random
import logging
from pathlib import Path
from typing import Optional
from typing import Optional

import cv2
import numpy as np
from tqdm import tqdm

from src.config import (
    WINDOWS, IMG_SIZE, RANDOM_SEED,
    YOLO_TRAIN_RATIO, RAW_TRAINING_DIR, RAW_ANNOTATIONS_DIR, YOLO_DIR,
)
from src.preprocessing.core.dicom_reader import BrainDicomReader
from src.preprocessing.core.json_parser import AnnotationParser

logger = logging.getLogger(__name__)


class YoloDatasetBuilder:
    """
    Builds YOLO-format fracture detection dataset.

    Uses bone window (WW=1000, WL=400) for fracture visualization.
    Output: images/{train,val}/*.jpg and labels/{train,val}/*.txt + dataset.yaml
    """

    def __init__(
        self,
        raw_dicom_dir: Optional[str] = None,
        raw_json_dir: Optional[str] = None,
        output_dir: Optional[str] = None,
        split_ratio: float = YOLO_TRAIN_RATIO,
    ):
        self.raw_dicom_dir = Path(raw_dicom_dir or RAW_TRAINING_DIR)
        self.raw_json_dir = Path(raw_json_dir or RAW_ANNOTATIONS_DIR)
        self.output_dir = Path(output_dir or YOLO_DIR)
        self.split_ratio = split_ratio

        for sub in ["images/train", "images/val", "labels/train", "labels/val"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    def _scan_fracture_patients(self):
        """Scan annotations to find patients with/without fracture boxes."""
        fractured = {}
        healthy = []

        all_ids = sorted(
            d.name for d in self.raw_dicom_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )

        for pid in tqdm(all_ids, desc="Scanning fractures"):
            json_dir = self.raw_json_dir / pid
            if not json_dir.exists():
                healthy.append(pid)
                continue

            parser = AnnotationParser(str(json_dir))
            summary = parser.get_patient_summary()
            if summary["n_with_boxes"] > 0:
                # Collect per-slice box data
                box_slices = {}
                for jf in sorted(json_dir.iterdir()):
                    if jf.suffix != ".json":
                        continue
                    dcm_name = jf.name.replace(".json", ".dcm")
                    data = parser.parse_slice(dcm_name)
                    if data["bboxes"]:
                        box_slices[dcm_name] = data["bboxes"]
                if box_slices:
                    fractured[pid] = box_slices
                    continue

            healthy.append(pid)

        return fractured, healthy

    def build(self):
        logger.info(f"Building YOLO fracture dataset → {self.output_dir}")

        fractured, healthy = self._scan_fracture_patients()
        frac_pids = list(fractured.keys())

        random.seed(RANDOM_SEED)
        random.shuffle(frac_pids)
        random.shuffle(healthy)

        split = int(len(frac_pids) * self.split_ratio)
        n_healthy_train = int(len(healthy) * self.split_ratio)
        train_set = set(frac_pids[:split] + healthy[:n_healthy_train])

        logger.info(f"  Fractured: {len(frac_pids)} | Healthy: {len(healthy)}")
        logger.info(f"  Train: {len(train_set)} | Val: {len(all_ids := frac_pids + healthy) - len(train_set)}")

        # Save split info
        split_info = {
            "random_seed": RANDOM_SEED,
            "split_ratio": self.split_ratio,
            "train_patients": sorted(train_set),
            "val_patients": sorted(set(all_ids) - train_set),
        }
        with open(self.output_dir / "split_info.json", "w") as f:
            json.dump(split_info, f, indent=2)

        # Process each patient
        img_count = 0
        bone_win = WINDOWS["bone"]
        for pid in tqdm(all_ids, desc="Extracting YOLO slices"):
            subset = "train" if pid in train_set else "val"
            img_out = self.output_dir / "images" / subset
            lbl_out = self.output_dir / "labels" / subset

            dicom_dir = self.raw_dicom_dir / pid
            try:
                reader = BrainDicomReader(str(dicom_dir)).load_and_sort()
            except (ValueError, FileNotFoundError):
                continue

            is_frac = pid in fractured
            pos_names = set(fractured[pid].keys()) if is_frac else set()
            all_names = [os.path.basename(s.filename) for s in reader.slices]
            neg_names = [n for n in all_names if n not in pos_names]

            # Sample strategy
            slices_to_do = []
            if is_frac:
                n_neg = min(1, len(neg_names))
                slices_to_do = list(pos_names) + random.sample(neg_names, n_neg)
            else:
                if random.random() <= 0.2:
                    slices_to_do = random.sample(all_names, min(1, len(all_names)))

            for ds in reader.slices:
                dcm_name = os.path.basename(ds.filename)
                if dcm_name not in slices_to_do:
                    continue

                hu = reader._pixel_to_hu(ds.pixel_array, ds)
                bone = BrainDicomReader.apply_windowing(hu, bone_win)
                img_8 = (bone * 255).astype("uint8")
                rgb = cv2.cvtColor(img_8, cv2.COLOR_GRAY2RGB)

                stem = f"{pid}_{dcm_name.replace('.dcm', '')}"
                cv2.imwrite(str(img_out / f"{stem}.jpg"), rgb)

                # Labels
                with open(str(lbl_out / f"{stem}.txt"), "w") as f:
                    if is_frac and dcm_name in pos_names:
                        for box in fractured[pid][dcm_name]:
                            x, y, w, h = box
                            cx, cy = (x + w / 2) / IMG_SIZE, (y + h / 2) / IMG_SIZE
                            wn, hn = w / IMG_SIZE, h / IMG_SIZE
                            f.write(f"0 {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}\n")
                img_count += 1

        self._create_yaml()
        logger.info(f"✅ YOLO: {img_count} images, {len(frac_pids)} fractured patients")

    def _create_yaml(self):
        yaml = (
            f"path: {os.path.abspath(self.output_dir)}\n"
            f"train: images/train\nval: images/val\n"
            f"names:\n  0: fracture\n"
        )
        with open(self.output_dir / "dataset.yaml", "w") as f:
            f.write(yaml)
        logger.info(f"  dataset.yaml created")
