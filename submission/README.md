# IAAA 2026 Brain CT Triage — Submission Package

Self-contained package for the **IAAA 2026 Brain CT Triage Challenge**
leaderboard. Zip the contents of this folder and upload it — no installation
is needed on the server (the evaluation environment already provides every
dependency).

## 📁 Structure

```
submission/
├── submission.py        # CLI entry point (leaderboard interface)
├── model.py             # Model API: predict(study_dir) → 7 intermediates
├── triage.py            # Official triage function (0 / 1 / 2)
├── requirements.txt     # Imports used (all pre-installed on the server)
├── README.md            # This file
└── models/              # Trained weights
    ├── monai/
    │   └── SegResNet_best.pth        # ICH segmentation (MONAI SegResNet)
    ├── yolo/
    │   └── best.pt                   # fracture detection (YOLO)
    └── mls_heatmap/
        └── mls_heatmap_best.pth      # MLS keypoints (HRNet heatmap)
```

> **Single strategy package:** this submission ships the **new** `mls_heatmap`
> MLS strategy only. The legacy slice-selector / keypoint-regression models
> were removed, so the MLS path runs the heatmap model on every slice, keeps
> only confident slices (min keypoint peak ≥ `--mls-min-peak`, default 0.9)
> and aggregates the per-slice MLS with `max`.

## 🚀 Execution (leaderboard interface)

```bash
python submission.py \
    --data-dir /path/to/data/dir \
    --predictions-file-path /path/to/submission.csv
```

`data-dir` must contain one sub-directory per study, each with `*.dcm` files
(the competition layout: `dicom/{dicom_series.id}/*.dcm`).

Output CSV:

```
id,prediction
study_001,2
study_002,0
...
```

`prediction` is the triage class: **0** = non-urgent, **1** = urgent,
**2** = critical.

### Optional flags

| Flag | Default | Description |
|------|---------|-------------|
| `--models-dir` | `models` | Directory with model weights |
| `--device` | `auto` | `auto` (cuda when available), `cuda`, `cpu` |
| `--mls-min-peak` | `0.9` | Min heatmap peak (all 3 keypoints) to trust a slice |
| `--mls-top-k` | `-` | Keep the top-K most confident slices instead of the peak threshold |
| `--mls-aggregation` | `max` | `max` or `p90` across selected slices |
| `--mls-batch-size` | `16` | Slices per heatmap forward pass |

## 📦 Creating the submission ZIP

### Option A — helper script (recommended)

From the project root:

```bash
python scripts/make_submission_zip.py     # → submission.zip at project root
```

The script zips the **contents** of `submission/` (so `submission.py`,
`model.py`, `models/` live at the zip root) and excludes `__pycache__`,
`.gitkeep` and `*.pyc`.

### Option B — manual

```bash
cd submission
zip -r ../submission.zip . \
    -x "**/__pycache__/**" \
    -x "**/.gitkeep" \
    -x "*.pyc"
```

Then upload `submission.zip` to the competition leaderboard.

## 🧠 Model API

`model.predict(study_dir, models)` returns exactly 7 intermediate keys:

| Key | Description | Unit |
|---|---|---|
| `V_EDH` | Epidural hemorrhage volume | mL |
| `V_SDH` | Subdural hemorrhage volume | mL |
| `V_IPH` | Intraparenchymal hemorrhage volume | mL |
| `V_SAH` | Subarachnoid hemorrhage volume | mL |
| `V_IVH` | Intraventricular hemorrhage volume | mL |
| `fracture_prob` | Skull fracture probability | [0, 1] |
| `MLS_mm` | Midline shift at the foramen of Monro | mm |

`submission.py` applies the official `triage.py` rules to convert these into
the final triage class.
