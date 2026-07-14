# IAAA 2026 Brain CT Triage — Submission Package

This folder contains everything needed to submit your model to the
**IAAA 2026 Brain CT Triage Challenge** leaderboard.

## 📁 Structure

```
submission/
├── submission.py           # CLI entry point
├── model.py                # Competition Model API  (predict → intermediates)
├── triage.py               # Official triage function
├── download_models.py      # Download registered models from MLflow
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── models/                 # ⬅️ Place your trained model weights here
    ├── nnunet/
    │   └── checkpoint_best.pth    # ICH segmentation (nnU-Net)
    ├── yolo/
    │   └── best.pt                # Fracture detection (YOLO)
    └── mls/
        ├── slice_selector_best.ckpt   # MLS slice selector
        └── keypoint_best.ckpt         # MLS keypoint detector
```

## 🚀 How to Use

### Option 1: Download from MLflow Model Registry (recommended)

After training your models and **manually registering** the best runs in
the MLflow Model Registry (via DagsHub UI), run:

```bash
# Set your tracking URI (DagsHub example)
export MLFLOW_TRACKING_URI="https://dagshub.com/youruser/yourrepo.mlflow"

# Download all models at once
python download_models.py --tracking-uri $MLFLOW_TRACKING_URI --all

# Or download individually
python download_models.py --tracking-uri $MLFLOW_TRACKING_URI --model-name ich_nnunet --version latest
python download_models.py --tracking-uri $MLFLOW_TRACKING_URI --model-name fracture_yolo --version latest
python download_models.py --tracking-uri $MLFLOW_TRACKING_URI --model-name mls_slice_selector --version latest
python download_models.py --tracking-uri $MLFLOW_TRACKING_URI --model-name mls_keypoint --version latest
```

### Option 2: Place model files manually

Copy your trained weights into the corresponding sub-directories under
`models/` following the structure above.

## 🧪 Testing Locally

```bash
# Run on a test data directory
python submission.py \
    --data-dir /path/to/test/studies \
    --predictions-file-path ./output.csv

# The CSV will contain:
#   id,prediction
#   study_001,1
#   study_002,0
#   ...
```

## 📦 Creating the Submission ZIP

```bash
# From the project root
cd submission
zip -r ../submission.zip . \
    -x "models/**/.gitkeep" \
    -x "**/__pycache__/**" \
    -x "*.pyc"
```

Then upload `submission.zip` to the competition leaderboard.

## 🧠 Model API (`model.py`)

The `predict(study_dir)` function receives a path to a directory containing
DICOM files for one study and returns a dictionary with exactly 7 keys:

| Key | Description | Unit |
|---|---|---|
| `V_EDH` | Epidural hemorrhage volume | mL |
| `V_SDH` | Subdural hemorrhage volume | mL |
| `V_IPH` | Intraparenchymal hemorrhage volume | mL |
| `V_SAH` | Subarachnoid hemorrhage volume | mL |
| `V_IVH` | Intraventricular hemorrhage volume | mL |
| `fracture_prob` | Skull fracture probability | [0, 1] |
| `MLS_mm` | Midline shift at foramen of Monro | mm |

## 📊 Triage Output

The official triage function (`triage.py`) maps intermediates to classes:

- **0** = Non-urgent
- **1** = Urgent
- **2** = Critical
