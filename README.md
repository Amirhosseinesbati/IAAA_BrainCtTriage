# IAAA Brain CT Triage — IAAA 2026 Challenge

> **Current development workflow:** competition-aligned patient-level OOF
> evaluation, manifest-driven Vast.ai experiments and complete MLflow tracking
> are implemented on `codex/competition-winning-pipeline`. See
> [reports/competition_pipeline_implementation.md](reports/competition_pipeline_implementation.md)
> and [config/README.md](config/README.md).

## Competition-aligned workflow

```bash
# Rebuild/verify the immutable patient-level folds
python scripts/build_folds.py

# Open the experiment control center
streamlit run src/deploy/deployApp.py

# Run a saved manifest locally or on a prepared worker
python -m src.pipelines.run_pipeline \
  --manifest config/experiments/example-mls-fold0.yaml

# Evaluate genuine seven-intermediate OOF predictions
python scripts/assemble_oof.py \
  --ich reports/oof_ich.csv \
  --fracture reports/oof_fracture.csv \
  --mls reports/oof_mls.csv
python scripts/evaluate_oof.py reports/oof_predictions.csv

# Nested-OOF calibration; the submission bundle is saved only if the
# configured paired-bootstrap acceptance gate passes.
python scripts/fit_calibration.py reports/oof_predictions.csv

# Build and enforce the 1 GiB submission limit
python scripts/make_submission_zip.py
```

Macro-F1 is the primary selection metric stated in the official guide. QWK is
retained as a secondary diagnostic. Experiments must use `config/folds.csv` so
multiple studies from the same patient can never cross train/validation folds.

**Project summary**
- **Overview:** This repository contains an in-progress research / engineering project built for the IAAA 2026 "Brain CT Triage" challenge. The system combines pretrained and custom models to detect intracranial hemorrhage (ICH), skull fracture, and midline shift (MLS) from head CT DICOM studies and produce a structured triage decision.
- **Primary goal:** Support rapid, automated triage decisions from CT studies to prioritize critical cases for clinical review.
- **Challenge reference:** The problem statement and rules are in [Competition-Guide/iaaa-competition-2026-brain-ct-triage-challenge.pdf](Competition-Guide/iaaa-competition-2026-brain-ct-triage-challenge.pdf).

**Status**
- **Project state:** Incomplete / prototype. Core components (data pipeline, training scripts, inference pipeline, Streamlit demo) exist, but not all model training runs or dataset conversions are finalized.
- **What remains:** finalize dataset prep for all tasks, complete end-to-end training + evaluation, add CI / reproducible experiment scripts, consolidate model artifact storage and inference tests.

**Repository structure (high level)**
- **`app.py`**: Streamlit demo and lightweight UI for local inference. See [app.py](app.py).
- **`src/`**: Main source code for preprocessing, inference, pipelines and training. Primary entry points:
	- [src/inference/main_predict.py](src/inference/main_predict.py) — model loading and CLI-style predict helper.
	- [src/pipelines/prepare_all_data.py](src/pipelines/prepare_all_data.py) — central data preparation pipeline.
	- [src/training/train_yolo.py](src/training/train_yolo.py) — example training script for YOLO fracture detector.
- **`Data/`**: raw and processed datasets. Contains DVC pointers and processed builds used for nnU-Net, YOLO and MLS datasets.
- **`models/`**: model artifacts and weights (checkpoints, YOLO weights, pretrained backbones).
- **`experiments/`**: training outputs, YOLO experiment outputs and result folders.
- **`logs/`** and **`lightning_logs/`**: MLflow / Lightning logs for experiments and hyperparameters.
- **`requirements.txt`**: pinned Python dependencies (heavy medical imaging + DL stack). See [requirements.txt](requirements.txt).

**Key components and design**
- **Preprocessing:** Custom DICOM reader under `src.preprocessing.core` converts ACR/NEMA DICOM series into consistent inputs used by downstream modules (nnU-Net, YOLO, and custom MLS builders).
- **Segmentation (ICH):** nnU-Net pipeline prepared under `src.preprocessing.builders.nnunet_builder` and processed data saved to `Data/processed/nnUNet`.
- **Detection (Fracture):** YOLO pipeline implemented in `src.preprocessing.builders.yolo_builder` with training script in `src/training/train_yolo.py` and dataset at `Data/processed/yolo_fracture`.
- **Midline shift (MLS):** heatmap-based keypoint regression strategy (`mls_heatmap` — HRNet backbone, DARK sub-pixel decoding, top-K slice aggregation) estimates MLS in mm. The legacy slice-selector + direct-regression keypoint models remain available as a fallback. See the [MLS Heatmap Strategy](#mls-heatmap-strategy-mls_heatmap) section.
- **Triage logic:** Business rules live in `src/inference/triage_rules.py` (applies thresholds on ICH volumes, MLS, and fracture presence to decide Level 1 / Level 2 / Normal).
- **Demo & inference:** `app.py` runs a Streamlit UI for manual local inference using `src/inference/main_predict.py`.

## MLS Heatmap Strategy (`mls_heatmap`)

A pluggable strategy (same pattern as the ICH strategies in `src/strategies/`) for midline-shift estimation built on heatmap-based keypoint regression:

- **Architecture:** HRNet-W32 (default) or HRNet-W18 backbone via `timm` predicts 3 Gaussian heatmap channels — `AnteriorFalxAttachment`, `PosteriorFalxAttachment`, `OutermostPointOfTheFalx` — at 1/4 input resolution. Input is a 3-channel (brain + subdural + bone) or single-channel windowed CT slice at 512×512.
- **Sub-pixel decoding:** DARK (Distribution-Aware coordinate Representation) extracts sub-pixel keypoints (round-trip error < 0.05 px) instead of plain argmax — critical around the 3 mm / 5 mm triage thresholds.
- **MLS geometry:** perpendicular distance from `OutermostPointOfTheFalx` to the ideal falx line through the two attachment points, converted to mm with the DICOM `PixelSpacing`.
- **Slice selection:** the existing ResNet18 SliceSelector picks the top-K candidate slices (K configurable, default 3) and the per-slice MLS values are aggregated (`max` or `p90`) for robustness against slice-selection error.
- **Training:** pure PyTorch (no Lightning) with MLflow logging, early stopping, `ReduceLROnPlateau`, AMP, and validation metrics reported in mm — `kp_mae_px`, `mls_mae_mm` (MAE of the final MLS), `mls_bin_acc` (correct `<1 / 1–3 / 3–5 / ≥5` mm bucket), and critical-regime MAE. Augmentation (rotation ±10°, translation, intensity jitter) transforms keypoints consistently with the image.
- **Missing keypoints:** a keypoint that is `None` yields an all-zero heatmap and its loss contribution is masked (no fake target).

### Files

| File | Purpose |
|------|---------|
| `src/strategies/mls_heatmap/model.py` | HRNet backbone + heatmap head (`HRNetHeatmapModel`) |
| `src/strategies/mls_heatmap/utils.py` | Gaussian heatmap generation, DARK decode, MLS math, triage binning |
| `src/strategies/mls_heatmap/dataset.py` | Dataset + augmentation + patient-level train/val split |
| `src/strategies/mls_heatmap/train.py` | Pure-PyTorch training loop (`train_mls_heatmap`) |
| `src/strategies/mls_heatmap/predict.py` | `predict_mls(study_dir)` + cached `MLSHeatmapPredictor` |
| `src/strategies/mls_heatmap/_strategy.py` | Strategy class (auto-registered via `MLSStrategyRegistry`) |
| `src/strategies/config_models.py` | `MLSHeatmapConfig` (Pydantic — drives the dynamic UI form) |
| `tests/test_heatmap_utils.py`, `tests/test_mls_integration.py` | Unit tests (27 total, round-trip < 0.5 px) |

### Train

```bash
# via the strategy-aware ZenML pipeline
uv run python -m src.pipelines.run_pipeline --run mls-strategy \
    --strategy mls_heatmap \
    --config '{"backbone":"hrnet_w32","epochs":100,"batch_size":8}'

# list available MLS strategies
uv run python -m src.pipelines.run_pipeline --run mls-strategy --list-mls-strategies
```

Checkpoints are written to `models/checkpoints/mls_heatmap/` (`mls_heatmap_best.pth` + `mls_heatmap_final.pth`) and logged to MLflow experiment `IAAA_BrainCT_MLS_Heatmap`.

### Infer

```python
from src.strategies.mls_heatmap.predict import predict_mls

mls_mm = predict_mls("dicom/12345")
# Checkpoints are resolved from: explicit args → MLS_SLICE_SELECTOR_PATH /
# MLS_HEATMAP_MODEL_PATH env vars → models/checkpoints/ defaults.
```

`MLSHeatmapPredictor` loads both models once (Streamlit `@st.cache_resource` friendly) and exposes a `predict(reader)` duck-typed interface. `main_predict.load_all_models()` auto-detects the heatmap checkpoint and falls back to the legacy keypoint model when absent.

### Run from the UI / Vast.ai

- **`src/deploy/deployApp.py`:** choose the MLS pipeline → the `mls_heatmap` strategy renders a dynamic config form from its Pydantic JSON schema; "Launch on Vast.ai" forwards `MLS_STRATEGY` + `MLS_CONFIG` through `src/deploy/deploy.py`.
- **`setup_vast.sh`:** for `TARGET_PIPELINE=mls`, decodes `MLS_CONFIG_B64` and runs `run_pipeline --run mls-strategy --strategy "${MLS_STRATEGY:-mls_heatmap}" --config "$MLS_CONFIG"` on the rented GPU.

**Installation (recommended local env)**
1. Create and activate a Python virtual environment (Python >= 3.10 recommended).

```bash
python -m venv venv
# Windows PowerShell
venv\\Scripts\\Activate.ps1
# or on Unix
source venv/bin/activate
```

2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Optional: install GPU-enabled PyTorch matching your CUDA version following the official instructions if you plan to train or run models on GPU.

**Quick start — local inference (Streamlit demo)**
1. Ensure you have model weights available under `models/` as referenced by `src/inference/main_predict.py` (YOLO weights in `models/yolo_weights/best.pt`, checkpoints in `models/checkpoints`).
2. Run the demo:

```bash
python app.py
# open the shown Streamlit URL in your browser
```

3. In the sidebar, enter a DICOM study folder (for example `Data/raw/training/<patient_id>`) and click the run button. The UI shows ICH volumes, fracture flag, MLS, and the final triage label.

**Training & data preparation**
- **Prepare data:** run `python src/pipelines/prepare_all_data.py` to build the nnU-Net, YOLO and MLS processed datasets (some builders are commented out as the project is in-progress). See [src/pipelines/prepare_all_data.py](src/pipelines/prepare_all_data.py).
- **Train fracture detector:** `python src/training/train_yolo.py` uses Ultralytics YOLO and MLflow logging. Results are stored under `experiments/yolo_results/` and MLflow logs under `logs/mlflow_runs/`.
- **Train segmentation / MLS:** nnU-Net and custom training scripts exist in `src/training/` and are partially configured; follow the builders and training helpers there to complete experiments.

**Experiments & artifacts**
- Trained weights, best checkpoints and experiment outputs are expected in `models/` and `experiments/`. Some weights are provided (e.g., `models/pretrained/yolov8s.pt`), but full training runs and final evaluation reports remain to be completed.

**Reproducibility notes**
- The repository uses DVC pointers (see `Data/*.dvc`) for large raw datasets. Ensure you have DVC installed and configured to pull large files if needed.
- MLflow is used for experiment tracking in the training scripts; logs are written to `logs/mlflow_runs/` by default.

**Project limitations & next steps (short roadmap)**
- Finalize dataset builders and verify all processed outputs for nnU-Net and YOLO.
- Run end-to-end training for ICH segmentation and MLS models and add evaluation scripts that produce ROC / confusion matrices and per-class metrics.
- Add unit/integration tests for the DICOM reader and inference pipeline.
- Add lightweight Dockerization and simple CI to validate installs and run a small inference smoke test on CI.

**Contacts**
- **Author / Maintainer:** see repository metadata and commit history for contact information. For a resume archive, mention your role (e.g., "Lead developer, data pipeline & detection models").

**License**
- This repository does not include an explicit license file. Add a `LICENSE` when ready to share publicly.

---
If you want, I can now:
- draft a concise "resume bullet" (1–2 lines) summarizing your role in this project, or
- run the repository's unit scripts / a smoke inference to verify the `app.py` demo locally (requires models and data). 
