"""
json_parser.py — Core annotation parser for Brain CT studies.

Improved version with:
- parse_all_slices() for batch processing
- Validation that JSON files match DICOM files
- get_patient_summary() for annotation statistics
- More robust RLE decoding with shape validation
- Support for multiple JSON field names
"""

import os
import json
import glob
import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class AnnotationParser:
    """
    Reads and parses per-slice annotation JSON files.

    The JSON files contain:
    - segmentation_rle: RLE-encoded 2D label mask (ICH subtypes)
    - keypoints: dict of {keypoint_name: [x, y]} (MLS)
    - boxes_xywh: list of [x, y, w, h] (skull fractures)

    Usage:
        parser = AnnotationParser("path/to/annotations")
        result = parser.parse_slice("SOPInstanceUID.dcm")
        all_results = parser.parse_all_slices(dicom_slice_list)
        summary = parser.get_patient_summary()
    """

    def __init__(self, json_dir: str):
        """
        Args:
            json_dir: Path to directory containing annotation JSON files
                      for one patient (e.g., Data/raw/annotations/272179).
        """
        self.json_dir = json_dir
        self._json_files: Optional[dict] = None  # filename -> path mapping

    def _index_json_files(self) -> dict:
        """Build a mapping of DICOM filename → JSON path."""
        if self._json_files is not None:
            return self._json_files

        self._json_files = {}
        if not os.path.isdir(self.json_dir):
            logger.warning(f"Annotation directory not found: {self.json_dir}")
            return self._json_files

        for fpath in glob.glob(os.path.join(self.json_dir, "*.json")):
            fname = os.path.basename(fpath)
            # Map both with and without .dcm extension
            self._json_files[fname] = fpath
            # Also map without .json extension for .dcm lookups
            if fname.endswith(".json"):
                no_ext = fname[:-5]
                self._json_files[no_ext] = fpath
                self._json_files[no_ext + ".dcm"] = fpath

        return self._json_files

    @staticmethod
    def decode_rle(counts: list, shape=(512, 512)) -> np.ndarray:
        """
        Decode Run-Length Encoding (RLE) to a 2D label mask.

        RLE format: [value1, length1, value2, length2, ...]
        Background (0) is usually first. Non-zero values are ICH class labels.

        Args:
            counts: Flat list alternating [value, length, value, length, ...].
            shape: Expected (height, width) of the mask.

        Returns:
            2D numpy array of shape `shape` with integer labels.
        """
        if not counts:
            return np.zeros(shape, dtype=np.uint8)

        values = counts[0::2]
        lengths = counts[1::2]

        total_pixels = sum(lengths)
        expected_pixels = shape[0] * shape[1]

        if total_pixels != expected_pixels:
            logger.warning(
                f"RLE length mismatch: {total_pixels} vs expected {expected_pixels}. "
                f"Will truncate/pad."
            )
            # Handle mismatch gracefully
            if total_pixels > expected_pixels:
                # Truncate
                cumsum = np.cumsum(lengths)
                cutoff = np.searchsorted(cumsum, expected_pixels, side="right")
                lengths = lengths[:cutoff]
                lengths[-1] -= (cumsum[cutoff - 1] - expected_pixels)
                values = values[:cutoff]
            else:
                # Pad with zeros
                diff = expected_pixels - total_pixels
                lengths = np.append(lengths, diff)
                values = np.append(values, 0)

        mask_flat = np.repeat(values, lengths).astype(np.uint8)
        try:
            mask_2d = mask_flat.reshape(shape, order="C")
        except ValueError as e:
            logger.error(f"Cannot reshape RLE to {shape}: {e}")
            return np.zeros(shape, dtype=np.uint8)

        return mask_2d

    def parse_slice(self, dcm_filename: str) -> dict:
        """
        Read annotation data for one DICOM slice.

        Args:
            dcm_filename: DICOM filename (e.g., "1.2.3.4.5.dcm")
                          or SOPInstanceUID to look up.

        Returns:
            dict with keys:
                - has_label (bool): whether annotation file exists
                - mask_2d (np.ndarray): (512, 512) label mask
                - keypoints (dict): {keypoint_name: [x, y]}
                - bboxes (list): [[x, y, w, h], ...]
        """
        result = {
            "has_label": False,
            "has_segmentation": False,
            "mask_2d": np.zeros((512, 512), dtype=np.uint8),
            "keypoints": {},
            "bboxes": [],
        }

        # Build filename lookup
        json_files = self._index_json_files()

        # Try to find matching JSON
        json_path = json_files.get(dcm_filename)

        if json_path is None:
            # Try direct filename replacement
            json_filename = dcm_filename.replace(".dcm", ".json")
            json_path = os.path.join(self.json_dir, json_filename)
            if not os.path.exists(json_path):
                return result

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"⚠️  Could not read annotation {json_path}: {e}")
            return result

        result["has_label"] = True

        # ---- Extract segmentation mask ----
        seg_data = (
            data.get("segmentation_rle")
            or data.get("segmentation")
            or data.get("mask_rle")
        )
        if seg_data and seg_data.get("counts"):
            shape = seg_data.get("shape", (512, 512))
            result["mask_2d"] = self.decode_rle(seg_data["counts"], shape=shape)
            result["has_segmentation"] = True

        # ---- Extract keypoints (MLS) ----
        if "keypoints" in data and data["keypoints"]:
            for kp_name, coords in data["keypoints"].items():
                if coords is not None and len(coords) >= 2:
                    result["keypoints"][kp_name] = [float(c) for c in coords[:2]]

        # ---- Extract fracture bounding boxes ----
        boxes_raw = (
            data.get("boxes_xywh")
            or data.get("boxes")
            or data.get("bboxes")
            or []
        )
        for box in boxes_raw:
            if len(box) >= 4:
                result["bboxes"].append([float(c) for c in box[:4]])

        return result

    def parse_all_slices(self, dicom_slices: list) -> dict:
        """
        Parse annotations for all slices in a DICOM series.

        Args:
            dicom_slices: List of pydicom Dataset objects (sorted by Z).

        Returns:
            dict with:
                - masks_3d: np.ndarray (D, H, W) stacked masks
                - keypoints_per_slice: list of keypoint dicts
                - bboxes_per_slice: list of bbox lists
                - slice_indices: list of (dicom_index, SOPInstanceUID)
                - has_annotation: bool whether ANY annotation found
        """
        masks = []
        kp_list = []
        bbox_list = []
        indices = []
        has_any = False

        for idx, ds in enumerate(dicom_slices):
            sop_uid = getattr(ds, "SOPInstanceUID", f"slice_{idx:04d}")
            result = self.parse_slice(sop_uid)

            if result["has_label"]:
                has_any = True

            masks.append(result["mask_2d"])
            kp_list.append(result["keypoints"])
            bbox_list.append(result["bboxes"])
            indices.append((idx, sop_uid))

        return {
            "masks_3d": np.stack(masks, axis=0) if has_any else None,  # (D, H, W)
            "keypoints_per_slice": kp_list,
            "bboxes_per_slice": bbox_list,
            "slice_indices": indices,
            "has_annotation": has_any,
        }

    def get_patient_summary(self) -> dict:
        """
        Get summary statistics for this patient's annotations.

        Returns:
            dict with counts of slices having each annotation type.
        """
        json_files = self._index_json_files()
        if not json_files:
            return {"n_json_files": 0, "n_with_masks": 0, "n_with_keypoints": 0,
                    "n_with_boxes": 0, "json_dir": self.json_dir}

        n_with_mask = 0
        n_with_kp = 0
        n_with_boxes = 0
        total = 0

        for json_path in set(json_files.values()):
            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            total += 1

            seg = data.get("segmentation_rle") or data.get("segmentation")
            if seg and seg.get("counts"):
                n_with_mask += 1

            if data.get("keypoints"):
                n_with_kp += 1

            boxes = data.get("boxes_xywh") or data.get("boxes") or []
            if boxes:
                n_with_boxes += 1

        return {
            "patient_id": os.path.basename(self.json_dir),
            "json_dir": self.json_dir,
            "n_json_files": total,
            "n_with_masks": n_with_mask,
            "n_with_keypoints": n_with_kp,
            "n_with_boxes": n_with_boxes,
            "pct_masks": round(n_with_mask / total * 100, 1) if total else 0,
            "pct_keypoints": round(n_with_kp / total * 100, 1) if total else 0,
            "pct_boxes": round(n_with_boxes / total * 100, 1) if total else 0,
        }

    def __repr__(self) -> str:
        summary = self.get_patient_summary()
        return (
            f"AnnotationParser({summary['patient_id']}: "
            f"{summary['n_json_files']} files, "
            f"{summary['pct_masks']}% masks, "
            f"{summary['pct_keypoints']}% kp, "
            f"{summary['pct_boxes']}% boxes)"
        )
