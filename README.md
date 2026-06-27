# IAAA Brain CT Triage — IAAA 2026 Challenge

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
- **Midline shift (MLS):** custom slice selector and keypoint models (checkpoints stored in `models/checkpoints`) are used to estimate MLS in mm.
- **Triage logic:** Business rules live in `src/inference/triage_rules.py` (applies thresholds on ICH volumes, MLS, and fracture presence to decide Level 1 / Level 2 / Normal).
- **Demo & inference:** `app.py` runs a Streamlit UI for manual local inference using `src/inference/load_all_models`.

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
