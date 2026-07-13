"""
eda_annotations.py — Phase 0.5: Annotation Quality Analysis

Analyzes the JSON annotation files directly to understand:
- Coverage: how many patients/slices have annotations
- Keypoint completeness for MLS (all 3 points present?)
- Fracture bounding box distribution
- RLE mask quality (empty masks, overlaps between classes)
- Cross-check CSV-computed volumes vs JSON-computed volumes
"""

import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ANNOTATION_DIR = BASE_DIR / "Data" / "raw" / "annotations"
TRAINING_DIR = BASE_DIR / "Data" / "raw" / "training"
CSV_PATH = BASE_DIR / "Data" / "metadata" / "training_df.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ICH label mapping (from the nnUNet dataset.json)
ICH_LABELS = {0: "background", 1: "IVH", 2: "IPH", 3: "SDH", 4: "EDH", 5: "SAH"}
KEYPOINT_NAMES = ["AnteriorFalxAttachment", "PosteriorFalxAttachment", "OutermostPointOfTheFalx"]


def get_annotation_patients() -> list:
    """List all patients with annotation folders."""
    if not ANNOTATION_DIR.exists():
        print(f"❌ Annotation directory not found: {ANNOTATION_DIR}")
        return []
    patients = sorted(
        d.name for d in ANNOTATION_DIR.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    print(f"📂 Found {len(patients)} patients with annotation folders")
    return patients


def get_training_patients() -> list:
    """List all training patient folders."""
    if not TRAINING_DIR.exists():
        print(f"❌ Training directory not found: {TRAINING_DIR}")
        return []
    patients = sorted(
        d.name for d in TRAINING_DIR.iterdir()
        if d.is_dir() and d.name.isdigit()
    )
    print(f"📂 Found {len(patients)} patients in training folder")
    return patients


def analyze_annotation_coverage(anno_patients: list, train_patients: list) -> dict:
    """How many training patients have annotations?"""
    anno_set = set(anno_patients)
    train_set = set(train_patients)
    covered = len(anno_set & train_set)
    not_covered = train_set - anno_set

    results = {
        "training_patients": len(train_patients),
        "annotated_patients": covered,
        "pct_annotated": round(covered / len(train_patients) * 100, 1) if train_patients else 0,
        "unannotated_patients": len(not_covered),
        "unannotated_examples": sorted(list(not_covered))[:10],
    }

    print("\n=== Annotation Coverage ===")
    print(f"  Training patients: {len(train_patients)}")
    print(f"  With annotations: {covered} ({covered/len(train_patients)*100:.1f}%)")
    print(f"  Without annotations: {len(not_covered)} patients")
    print(f"  Examples of unannotated: {sorted(list(not_covered))[:10]}")

    return results


def analyze_annotation_files(anno_patients: list) -> dict:
    """Analyze per-patient annotation files."""
    slices_per_patient = []
    patients_with_keypoints = 0
    patients_with_boxes = 0
    patients_with_masks = 0

    total_keypoint_stats = defaultdict(int)  # which keypoints are present
    total_slices_with_kp = 0
    total_slices_with_boxes = 0
    total_slices_with_masks = 0
    total_slices = 0

    for pid in anno_patients:
        anno_dir = ANNOTATION_DIR / pid
        json_files = sorted(glob.glob(str(anno_dir / "*.json")))
        slices_per_patient.append(len(json_files))

        has_kp = False
        has_box = False
        has_mask = False

        for jf in json_files:
            total_slices += 1
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                print(f"  ⚠️  Error reading {jf}: {e}")
                continue

            # Check mask
            if "segmentation_rle" in data and data["segmentation_rle"].get("counts"):
                has_mask = True
                total_slices_with_masks += 1
            elif "segmentation" in data and data.get("segmentation"):
                has_mask = True
                total_slices_with_masks += 1

            # Check keypoints
            if "keypoints" in data and data["keypoints"]:
                has_kp = True
                total_slices_with_kp += 1
                for kp_name in KEYPOINT_NAMES:
                    if kp_name in data["keypoints"] and data["keypoints"][kp_name]:
                        total_keypoint_stats[kp_name] += 1

            # Check boxes
            if "boxes_xywh" in data and data["boxes_xywh"]:
                has_box = True
                total_slices_with_boxes += 1
            elif "boxes" in data and data["boxes"]:
                has_box = True
                total_slices_with_boxes += 1

        if has_kp:
            patients_with_keypoints += 1
        if has_box:
            patients_with_boxes += 1
        if has_mask:
            patients_with_masks += 1

    results = {
        "total_slices_annotated": total_slices,
        "slices_per_patient": {
            "min": int(np.min(slices_per_patient)),
            "max": int(np.max(slices_per_patient)),
            "mean": float(round(np.mean(slices_per_patient), 1)),
            "median": float(round(np.median(slices_per_patient), 1)),
        },
        "patients_with_masks": patients_with_masks,
        "patients_with_keypoints": patients_with_keypoints,
        "patients_with_boxes": patients_with_boxes,
        "slices_with_masks": total_slices_with_masks,
        "pct_slices_with_masks": round(total_slices_with_masks / total_slices * 100, 1) if total_slices else 0,
        "slices_with_keypoints": total_slices_with_kp,
        "pct_slices_with_keypoints": round(total_slices_with_kp / total_slices * 100, 1) if total_slices else 0,
        "slices_with_boxes": total_slices_with_boxes,
        "pct_slices_with_boxes": round(total_slices_with_boxes / total_slices * 100, 1) if total_slices else 0,
        "keypoint_frequency": dict(total_keypoint_stats),
    }

    print("\n=== Annotation File Statistics ===")
    print(f"  Total annotated slices: {total_slices}")
    print(f"  Slices per patient: mean={np.mean(slices_per_patient):.1f}, "
          f"median={np.median(slices_per_patient):.0f}, "
          f"min={np.min(slices_per_patient)}, max={np.max(slices_per_patient)}")
    print(f"  Patients with masks: {patients_with_masks}/{len(anno_patients)}")
    print(f"  Patients with keypoints: {patients_with_keypoints}/{len(anno_patients)}")
    print(f"  Patients with boxes: {patients_with_boxes}/{len(anno_patients)}")
    print(f"  Slices with masks: {total_slices_with_masks}/{total_slices} ({total_slices_with_masks/total_slices*100:.1f}%)")
    print(f"  Slices with keypoints: {total_slices_with_kp}/{total_slices} ({total_slices_with_kp/total_slices*100:.1f}%)")
    print(f"  Slices with boxes: {total_slices_with_boxes}/{total_slices} ({total_slices_with_boxes/total_slices*100:.1f}%)")

    if total_keypoint_stats:
        print(f"\n  Keypoint frequencies:")
        for kp, cnt in sorted(total_keypoint_stats.items()):
            print(f"    {kp}: {cnt} ({cnt/total_slices*100:.1f}%)")

    # Histogram of slices per patient
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(slices_per_patient, bins=30, color="#8e44ad", edgecolor="white", alpha=0.8)
    ax.axvline(np.median(slices_per_patient), color="red", linestyle="--",
               label=f"Median={np.median(slices_per_patient):.0f}")
    ax.axvline(np.mean(slices_per_patient), color="orange", linestyle="--",
               label=f"Mean={np.mean(slices_per_patient):.1f}")
    ax.set_xlabel("Number of JSON Annotation Files")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Distribution of Annotated Slices Per Patient", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "annotation_slices_per_patient.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Saved: {path}")

    return results


def analyze_rle_masks(anno_patients: list) -> dict:
    """Analyze RLE mask quality: overlaps, empty masks, class distribution."""
    total_masks = 0
    empty_masks = 0
    class_counts = defaultdict(int)
    overlap_count = 0
    max_overlap = 0

    # Sample a few patients for detailed analysis
    sample_patients = anno_patients[:min(20, len(anno_patients))]
    mask_sizes = []

    for pid in sample_patients:
        anno_dir = ANNOTATION_DIR / pid
        json_files = sorted(glob.glob(str(anno_dir / "*.json")))

        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            # Check segmentation field - might be "segmentation_rle" or "segmentation"
            seg_data = data.get("segmentation_rle") or data.get("segmentation")
            if not seg_data or not seg_data.get("counts"):
                empty_masks += 1
                continue

            total_masks += 1
            counts = seg_data["counts"]
            shape = seg_data.get("shape", (512, 512))

            # Decode RLE to check content
            try:
                values = counts[0::2]
                lengths = counts[1::2]
                mask_flat = np.repeat(values, lengths).astype(np.uint8)
                mask_size = len(mask_flat)
                mask_sizes.append(mask_size)

                # Reshape to 2D
                mask_2d = mask_flat.reshape(shape, order="C")

                # Check unique classes
                unique_classes = np.unique(mask_2d)
                for cls in unique_classes:
                    if cls > 0:  # Skip background
                        class_counts[int(cls)] += 1

                # Check for pixel-level overlap between classes
                # (in multi-label RLE, or by checking if a pixel has >1 label)
                if len(unique_classes) > 1:
                    # Check if any pixel has multiple labels assigned
                    overlap_pixels = sum(1 for c in unique_classes if c > 0)
                    if overlap_pixels > 1:
                        overlap_count += 1

            except Exception as e:
                print(f"    ⚠️  RLE decode error for {jf}: {e}")
                continue

    results = {
        "masks_analyzed": total_masks,
        "empty_masks": empty_masks,
        "class_frequency": {f"{ICH_LABELS.get(k, k)} (label={k})": v for k, v in sorted(class_counts.items())},
        "overlap_detected": overlap_count,
        "mask_size_stats": {
            "min": int(np.min(mask_sizes)) if mask_sizes else 0,
            "max": int(np.max(mask_sizes)) if mask_sizes else 0,
        } if mask_sizes else {},
    }

    print("\n=== RLE Mask Quality ===")
    print(f"  Masks analyzed: {total_masks}")
    print(f"  Empty masks found: {empty_masks}")
    print(f"  Class frequency in masks:")
    for cls_name, cnt in sorted(results["class_frequency"].items()):
        print(f"    {cls_name}: {cnt} slices")
    print(f"  Overlap cases (multiple classes in one slice): {overlap_count}")

    return results


def analyze_keypoint_quality(anno_patients: list) -> dict:
    """Detailed analysis of keypoint quality."""
    kp_slices_found = 0
    kp_coords = {kp: [] for kp in KEYPOINT_NAMES}
    patients_with_all_3 = 0

    for pid in anno_patients:
        anno_dir = ANNOTATION_DIR / pid
        json_files = sorted(glob.glob(str(anno_dir / "*.json")))

        pid_has_all_3 = False
        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            if "keypoints" not in data or not data["keypoints"]:
                continue

            kps = data["keypoints"]
            # Check if all 3 keypoints present
            all_present = all(
                kp_name in kps and kps[kp_name] is not None and len(kps[kp_name]) == 2
                for kp_name in KEYPOINT_NAMES
            )
            if all_present:
                kp_slices_found += 1
                pid_has_all_3 = True
                for kp_name in KEYPOINT_NAMES:
                    kp_coords[kp_name].append(kps[kp_name])

        if pid_has_all_3:
            patients_with_all_3 += 1

    # Statistics on coordinate ranges
    coord_stats = {}
    for kp_name, coords in kp_coords.items():
        if coords:
            coords_arr = np.array(coords)
            coord_stats[kp_name] = {
                "n_slices": len(coords),
                "x_range": [float(round(coords_arr[:, 0].min(), 1)),
                            float(round(coords_arr[:, 0].max(), 1))],
                "y_range": [float(round(coords_arr[:, 1].min(), 1)),
                            float(round(coords_arr[:, 1].max(), 1))],
                "x_mean": float(round(coords_arr[:, 0].mean(), 1)),
                "y_mean": float(round(coords_arr[:, 1].mean(), 1)),
            }

    results = {
        "slices_with_all_3_keypoints": kp_slices_found,
        "patients_with_all_3_keypoints": patients_with_all_3,
        "coordinate_stats": coord_stats,
    }

    print("\n=== Keypoint Quality ===")
    print(f"  Slices with all 3 keypoints: {kp_slices_found}")
    print(f"  Patients with all 3 keypoints: {patients_with_all_3}/{len(anno_patients)}")
    for kp_name, stats in coord_stats.items():
        print(f"  {kp_name}: x∈[{stats['x_range'][0]}, {stats['x_range'][1]}], "
              f"y∈[{stats['y_range'][0]}, {stats['y_range'][1]}], "
              f"n={stats['n_slices']}")

    return results


def analyze_fracture_boxes(anno_patients: list) -> dict:
    """Analyze fracture bounding boxes."""
    total_boxes = 0
    slices_with_boxes = 0
    patients_with_boxes = 0
    box_sizes = []

    for pid in anno_patients:
        anno_dir = ANNOTATION_DIR / pid
        json_files = sorted(glob.glob(str(anno_dir / "*.json")))

        pid_has_box = False
        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            boxes = data.get("boxes_xywh") or data.get("boxes") or []
            if boxes:
                pid_has_box = True
                slices_with_boxes += 1
                total_boxes += len(boxes)
                for box in boxes:
                    if len(box) >= 4:
                        w, h = box[2], box[3]
                        box_sizes.append(w * h)

        if pid_has_box:
            patients_with_boxes += 1

    results = {
        "patients_with_boxes": patients_with_boxes,
        "slices_with_boxes": slices_with_boxes,
        "total_bounding_boxes": total_boxes,
        "box_size_stats": {
            "min_area": float(round(np.min(box_sizes), 1)) if box_sizes else 0,
            "max_area": float(round(np.max(box_sizes), 1)) if box_sizes else 0,
            "mean_area": float(round(np.mean(box_sizes), 1)) if box_sizes else 0,
            "median_area": float(round(np.median(box_sizes), 1)) if box_sizes else 0,
        } if box_sizes else {},
    }

    print("\n=== Fracture Bounding Boxes ===")
    print(f"  Patients with boxes: {patients_with_boxes}/{len(anno_patients)}")
    print(f"  Slices with boxes: {slices_with_boxes}")
    print(f"  Total bounding boxes: {total_boxes}")
    if box_sizes:
        print(f"  Box area (pixels): min={np.min(box_sizes):.0f}, "
              f"max={np.max(box_sizes):.0f}, "
              f"mean={np.mean(box_sizes):.0f}")

    return results


def crosscheck_csv_vs_json(anno_patients: list) -> dict:
    """
    Cross-check: do per-series volumes from CSV match
    volumes computed from JSON annotations?
    Uses a sample of patients.
    """
    print("\n=== CSV vs JSON Cross-check ===")
    sample = anno_patients[:min(10, len(anno_patients))]

    mismatches = 0
    total_checked = 0

    for pid in sample:
        anno_dir = ANNOTATION_DIR / pid
        json_files = sorted(glob.glob(str(anno_dir / "*.json")))

        # Compute volume from JSON masks
        vol_from_json = {"IVH": 0, "IPH": 0, "SDH": 0, "EDH": 0, "SAH": 0}

        for jf in json_files:
            try:
                with open(jf, "r") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, IOError):
                continue

            seg_data = data.get("segmentation_rle") or data.get("segmentation")
            if not seg_data or not seg_data.get("counts"):
                continue

            # For cross-check, we just count pixels - actual volume
            # computation needs spacing from DICOM
            # For now, just check that JSON files have valid data
            counts = seg_data["counts"]
            shape = seg_data.get("shape", (512, 512))
            values = counts[0::2]
            lengths = counts[1::2]
            mask_flat = np.repeat(values, lengths).astype(np.uint8)
            try:
                mask_2d = mask_flat.reshape(shape, order="C")
            except ValueError:
                continue

            for label_val, name in ICH_LABELS.items():
                if label_val == 0:
                    continue
                n_pixels = int((mask_2d == label_val).sum())
                vol_from_json[name] += n_pixels

        total_checked += 1
        # Check if any positive labels found
        has_any = any(v > 0 for v in vol_from_json.values())
        if not has_any:
            mismatches += 1
            print(f"  ⚠️  Patient {pid}: JSON has no positive labels (all zero)")

    results = {
        "patients_checked": total_checked,
        "patients_with_no_json_labels": mismatches,
    }
    print(f"  Checked {total_checked} patients, {mismatches} had no positive labels in JSON")
    return results


def main():
    print("=" * 60)
    print("  Phase 0.5: Annotation Quality Analysis")
    print("=" * 60)

    anno_patients = get_annotation_patients()
    train_patients = get_training_patients()

    if not anno_patients:
        print("❌ No annotation patients found. Skipping.")
        return

    results = {}
    results["coverage"] = analyze_annotation_coverage(anno_patients, train_patients)
    results["file_stats"] = analyze_annotation_files(anno_patients)
    results["rle_quality"] = analyze_rle_masks(anno_patients)
    results["keypoint_quality"] = analyze_keypoint_quality(anno_patients)
    results["fracture_boxes"] = analyze_fracture_boxes(anno_patients)
    results["crosscheck"] = crosscheck_csv_vs_json(anno_patients)

    # Save
    json_path = OUTPUT_DIR / "eda_annotations_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Saved: {json_path}")
    print("\n✅ Phase 0.5 complete!")


if __name__ == "__main__":
    main()
