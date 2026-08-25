# Configuration contract

`project.yaml` is the repository-wide source of truth for paths, imaging
windows, label maps, official triage thresholds, MLflow experiment names,
artifact layout, training defaults, post-processing and Vast.ai profiles.

`folds.csv` is the immutable five-fold patient-grouped manifest. All series
from one `patient_id` remain in one fold. Regenerate it only deliberately:

```bash
python scripts/build_folds.py
```

The experiment UI writes reusable manifests under `config/experiments/`.
The exact selected manifest is transferred to Vast.ai and logged as an MLflow
artifact. `config/runtime/` is ignored because it contains the decoded live
manifest on a remote worker.

To test a temporary project configuration without modifying the default:

```bash
IAAA_CONFIG_PATH=config/my_project.yaml python -m src.pipelines.run_pipeline --manifest experiment.yaml
```

Secrets never belong in YAML. They stay in `.env` / environment variables.
