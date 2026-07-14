"""
train_nnunet.py — Automated nnU-Net training with real-time MLflow logging,
periodic model checkpoint upload, and early stopping support.
"""

import os
import json
import re
import time
import subprocess
from pathlib import Path
from glob import glob
import mlflow

from src.config import MLFLOW_EXP_NNUNET, log_src_snapshot, NNUNET_DEFAULTS


def get_dataset_name(nnunet_raw_dir: Path, dataset_id: str) -> str | None:
    """Extract actual dataset name from dataset.json inside the raw dataset folder."""
    pattern = nnunet_raw_dir / f"Dataset{int(dataset_id):03d}_*"
    try:
        dataset_folder = next(Path(p) for p in glob(str(pattern)))
        with open(dataset_folder / "dataset.json") as f:
            return json.load(f)["name"]
    except (StopIteration, FileNotFoundError):
        return None


def find_fold_folder(nnunet_results_dir: Path, dataset_id: str,
                     dataset_name: str, configuration: str, fold: int) -> Path | None:
    """Locate the fold output folder after nnU-Net creates it (trainer folder name unknown a priori)."""
    dataset_dir = nnunet_results_dir / f"Dataset{int(dataset_id):03d}_{dataset_name}"
    if not dataset_dir.exists():
        return None
    # Trainer folder pattern: nnUNetTrainer*__*__{configuration}
    pattern = str(dataset_dir / f"nnUNetTrainer*__*__{configuration}")
    matches = glob(pattern)
    if not matches:
        return None
    return Path(matches[0]) / f"fold_{fold}"


def find_latest_log(fold_folder: Path) -> Path | None:
    """Return the most recently modified training_log file in fold_folder."""
    logs = list(fold_folder.glob("training_log_*.txt"))
    return max(logs, key=os.path.getctime) if logs else None


def parse_epoch_from_log(log_path: Path, last_position: int = 0):
    """
    Read new lines from the training log and extract epoch metrics.

    Returns (list_of_epoch_dicts, new_last_position).
    Each epoch dict contains: epoch, train_loss, val_loss, dice_per_class, epoch_time, lr.
    """
    if not log_path.exists():
        return [], last_position

    with open(log_path, "r", encoding="utf-8") as f:
        f.seek(last_position)
        lines = f.readlines()
        new_position = f.tell()

    epochs = []
    current = {}

    for line in lines:
        line = line.strip()

        m = re.match(r"^Epoch\s+(\d+)", line)
        if m:
            if current.get("epoch") is not None:
                epochs.append(current)
            current = {"epoch": int(m.group(1))}
            continue

        m = re.match(r"Current learning rate:\s*([\d.eE+-]+)", line)
        if m:
            current["lr"] = float(m.group(1))
            continue

        m = re.match(r"train_loss\s+([\d.eE+-]+)", line)
        if m:
            current["train_loss"] = float(m.group(1))
            continue

        m = re.match(r"val_loss\s+([\d.eE+-]+)", line)
        if m:
            current["val_loss"] = float(m.group(1))
            continue

        m = re.match(r"Pseudo dice\s+\[(.+)\]", line)
        if m:
            try:
                current["dice_per_class"] = [float(x.strip()) for x in m.group(1).split(",")]
            except ValueError:
                pass
            continue

        m = re.match(r"Epoch time:\s*([\d.]+)\s*s", line)
        if m:
            current["epoch_time"] = float(m.group(1))
            continue

        m = re.match(r".*New best EMA pseudo Dice:\s*([\d.]+)", line)
        if m:
            current["best_ema_dice"] = float(m.group(1))

    # Don't forget the last partial epoch
    if current.get("epoch") is not None and current.get("val_loss") is not None:
        epochs.append(current)

    return epochs, new_position


def train_nnunet_pipeline(dataset_id="501", fold=0, configuration=None):
    """
    Run the full nnU-Net pipeline: plan & preprocess, then train with
    real-time MLflow logging, early stopping, and periodic model uploads.
    """
    if configuration is None:
        configuration = NNUNET_DEFAULTS.get("configuration", "2d")

    patience = NNUNET_DEFAULTS.get("early_stopping_patience", 100)
    save_every = NNUNET_DEFAULTS.get("save_every", 20)

    print(f"=== nnU-Net Pipeline | dataset={dataset_id} | config={configuration} | fold={fold} ===")
    print(f"    early_stopping_patience={patience} | save_every={save_every}")

    BASE_DIR = Path(__file__).resolve().parent.parent.parent
    NNUNET_DIR = BASE_DIR / "Data" / "processed" / "nnUNet"
    os.makedirs(NNUNET_DIR / "nnUNet_raw", exist_ok=True)
    os.makedirs(NNUNET_DIR / "nnUNet_preprocessed", exist_ok=True)
    os.makedirs(NNUNET_DIR / "nnUNet_results", exist_ok=True)

    env = os.environ.copy()
    env["nnUNet_raw"] = str(NNUNET_DIR / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(NNUNET_DIR / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(NNUNET_DIR / "nnUNet_results")

    # ─── MLflow run ───────────────────────────────────────────
    mlflow.set_experiment(MLFLOW_EXP_NNUNET)
    run_name = f"Dataset{dataset_id}_{configuration}_Fold{fold}"

    with mlflow.start_run(run_name=run_name) as active_run:
        mlflow.log_params({
            "dataset_id": dataset_id,
            "fold": fold,
            "configuration": configuration,
            "early_stopping_patience": patience,
            "save_every": save_every,
        })

        # ─── Step 1: Plan & Preprocess ────────────────────────
        print("\n--- nnUNetv2_plan_and_preprocess ---")
        preprocess_cmd = [
            "nnUNetv2_plan_and_preprocess",
            "-d", str(dataset_id),
            "--verify_dataset_integrity",
        ]
        subprocess.run(preprocess_cmd, env=env, check=True)

        # ─── Step 2: Train ────────────────────────────────────
        train_cmd = [
            "nnUNetv2_train",
            str(dataset_id),
            configuration,
            str(fold),
            "--save_every", str(save_every),
        ]
        print(f"\n--- Running: {' '.join(train_cmd)} ---")
        process = subprocess.Popen(
            train_cmd, env=env,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, bufsize=1,
        )

        # ─── Monitoring loop ──────────────────────────────────
        fold_folder = None
        log_path = None
        last_position = 0
        last_logged_epoch = 0
        best_val_loss = float("inf")
        epochs_no_improve = 0
        last_best_mtime = 0
        last_latest_upload_epoch = 0

        # Stream stdout in background so buffer doesn't fill up
        stdout_lines = []
        def drain_stdout():
            while True:
                line = process.stdout.readline()
                if not line:
                    break
                stdout_lines.append(line)

        import threading
        stdout_thread = threading.Thread(target=drain_stdout, daemon=True)
        stdout_thread.start()

        try:
            while process.poll() is None:
                # Try to locate fold folder & log file (appears after first epoch)
                if fold_folder is None:
                    dataset_name = get_dataset_name(NNUNET_DIR / "nnUNet_raw", dataset_id)
                    if dataset_name:
                        fold_folder = find_fold_folder(
                            NNUNET_DIR / "nnUNet_results", dataset_id,
                            dataset_name, configuration, fold,
                        )
                        if fold_folder:
                            print(f"   Monitoring folder: {fold_folder}")

                if fold_folder and log_path is None:
                    log_path = find_latest_log(fold_folder)
                    if log_path:
                        print(f"   Training log found: {log_path.name}")

                # Parse new epoch data
                if log_path:
                    epochs_data, last_position = parse_epoch_from_log(log_path, last_position)

                    for ep in epochs_data:
                        epoch_num = ep["epoch"]
                        if epoch_num <= last_logged_epoch:
                            continue
                        last_logged_epoch = epoch_num

                        # ── Log metrics ──
                        metrics = {}
                        for key in ("train_loss", "val_loss", "epoch_time", "lr"):
                            if key in ep:
                                metrics[key] = ep[key]
                        if "dice_per_class" in ep:
                            for i, d in enumerate(ep["dice_per_class"]):
                                metrics[f"dice_class_{i}"] = d
                            metrics["mean_fg_dice"] = sum(ep["dice_per_class"]) / len(ep["dice_per_class"])

                        if metrics:
                            mlflow.log_metrics(metrics, step=epoch_num)
                            print(f"   📊 Epoch {epoch_num}: train_loss={ep.get('train_loss','?'):.4f}, "
                                  f"val_loss={ep.get('val_loss','?'):.4f}, "
                                  f"mean_dice={metrics.get('mean_fg_dice', 0):.4f}")

                        # ── Early stopping check ──
                        if "val_loss" in ep:
                            if ep["val_loss"] < best_val_loss:
                                best_val_loss = ep["val_loss"]
                                epochs_no_improve = 0
                            else:
                                epochs_no_improve += 1
                                if epochs_no_improve >= patience > 0:
                                    print(f"\n⏹️  Early stopping triggered after {epoch_num} epochs "
                                          f"(no improvement for {patience} epochs)")
                                    process.terminate()
                                    break

                        # ── Upload best model (if changed) ──
                        if fold_folder:
                            best_ckpt = fold_folder / "checkpoint_best.pth"
                            if best_ckpt.exists():
                                mtime = best_ckpt.stat().st_mtime
                                if mtime > last_best_mtime:
                                    last_best_mtime = mtime
                                    try:
                                        mlflow.log_artifact(str(best_ckpt), artifact_path="models")
                                        print(f"   ☁️  Best model uploaded (epoch {epoch_num})")
                                    except Exception as e:
                                        print(f"   ⚠️  Best model upload failed: {e}")

                        # ── Upload latest model periodically ──
                        if fold_folder and (epoch_num - last_latest_upload_epoch) >= save_every:
                            last_latest_upload_epoch = epoch_num
                            latest_ckpt = fold_folder / "checkpoint_latest.pth"
                            if latest_ckpt.exists():
                                try:
                                    mlflow.log_artifact(str(latest_ckpt), artifact_path="models")
                                    print(f"   ☁️  Latest model uploaded (epoch {epoch_num})")
                                except Exception as e:
                                    print(f"   ⚠️  Latest model upload failed: {e}")

                time.sleep(10)

        except KeyboardInterrupt:
            print("\n⚠️  Training interrupted by user. Terminating...")
            process.terminate()

        # Wait for process to finish
        process.wait()
        stdout_thread.join(timeout=5)
        print(f"\n--- nnUNetv2_train exited with code {process.returncode} ---")

        # ─── Step 3: Log final artifacts ──────────────────────
        print("\n--- Logging final artifacts ---")

        dataset_name = get_dataset_name(NNUNET_DIR / "nnUNet_raw", dataset_id)
        if not dataset_name:
            print("Could not log artifacts — dataset name not found.")
            log_src_snapshot()
            return

        if fold_folder is None:
            fold_folder = find_fold_folder(
                NNUNET_DIR / "nnUNet_results", dataset_id,
                dataset_name, configuration, fold,
            )

        if fold_folder and fold_folder.exists():
            # progress.png
            progress_png = fold_folder / "progress.png"
            if progress_png.exists():
                mlflow.log_artifact(str(progress_png), artifact_path="training_plots")
                print("Logged progress.png")

            # training logs
            log_files = list(fold_folder.glob("training_log_*.txt"))
            if log_files:
                latest = max(log_files, key=os.path.getctime)
                mlflow.log_artifact(str(latest), artifact_path="training_logs")
                print(f"Logged training log: {latest.name}")

            # Best model (final copy)
            best_ckpt = fold_folder / "checkpoint_best.pth"
            if best_ckpt.exists():
                mlflow.log_artifact(str(best_ckpt), artifact_path="models")
                print("Logged best model (checkpoint_best.pth)")

            # Latest model (final copy)
            latest_ckpt = fold_folder / "checkpoint_latest.pth"
            if latest_ckpt.exists():
                mlflow.log_artifact(str(latest_ckpt), artifact_path="models")
                print("Logged latest model (checkpoint_latest.pth)")
        else:
            print(f"Warning: fold folder not found: {fold_folder}")

        # Code snapshot
        log_src_snapshot()

    print(f"=== nnU-Net Pipeline Completed Successfully! ===")


if __name__ == "__main__":
    for i in range(0):  # nnU-Net default is 5-fold CV
        train_nnunet_pipeline(dataset_id="501", fold=i)
