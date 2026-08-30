"""Nested patient-disjoint training of a frozen-feature fracture SA-MIL head."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.fracture.pooling import aggregate_study_scores
from src.fracture.smooth_attention_mil import (
    SmoothAttentionMIL,
    SmoothAttentionMILConfig,
    smooth_attention_loss,
)
from src.mlops.tracking import ExperimentContext, experiment_run


@dataclass(frozen=True)
class StudyBag:
    study_id: str
    patient_id: str
    truth: int
    outer_fold: int
    indices: np.ndarray


@dataclass(frozen=True)
class Standardizer:
    embedding_mean: np.ndarray
    embedding_scale: np.ndarray
    score_mean: float
    score_scale: float


def _study_bags(slices: pd.DataFrame) -> list[StudyBag]:
    bags: list[StudyBag] = []
    for study_id, group in slices.groupby("study_id", sort=True):
        ordered = group.sort_values("slice_index", kind="stable")
        for column in ("patient_id", "study_fracture", "outer_fold"):
            if ordered[column].nunique() != 1:
                raise ValueError(f"Inconsistent {column} for study {study_id}")
        bags.append(
            StudyBag(
                study_id=str(study_id),
                patient_id=str(ordered["patient_id"].iloc[0]),
                truth=int(ordered["study_fracture"].iloc[0]),
                outer_fold=int(ordered["outer_fold"].iloc[0]),
                indices=ordered.index.to_numpy(dtype=np.int64),
            )
        )
    return bags


def _score_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(clipped) - np.log1p(-clipped)


def _fit_standardizer(
    embeddings: np.ndarray,
    slice_scores: np.ndarray,
    bags: list[StudyBag],
) -> Standardizer:
    indices = np.concatenate([bag.indices for bag in bags])
    selected_embeddings = np.asarray(embeddings[indices], dtype=np.float32)
    embedding_mean = selected_embeddings.mean(axis=0)
    embedding_scale = selected_embeddings.std(axis=0)
    embedding_scale = np.maximum(embedding_scale, 1e-5)
    transformed_score = _score_logit(slice_scores[indices])
    score_scale = max(float(transformed_score.std()), 1e-5)
    return Standardizer(
        embedding_mean=embedding_mean,
        embedding_scale=embedding_scale,
        score_mean=float(transformed_score.mean()),
        score_scale=score_scale,
    )


def _prepare_features(
    embeddings: np.ndarray,
    slice_scores: np.ndarray,
    bags: list[StudyBag],
    standardizer: Standardizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    prepared: dict[str, torch.Tensor] = {}
    for bag in bags:
        embedding = (
            np.asarray(embeddings[bag.indices], dtype=np.float32)
            - standardizer.embedding_mean
        ) / standardizer.embedding_scale
        score = (
            _score_logit(slice_scores[bag.indices]) - standardizer.score_mean
        ) / standardizer.score_scale
        features = np.concatenate([embedding, score[:, None]], axis=1)
        prepared[bag.study_id] = torch.from_numpy(features).to(device=device)
    return prepared


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _predict(
    model: SmoothAttentionMIL,
    bags: list[StudyBag],
    prepared: dict[str, torch.Tensor],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    model.eval()
    probabilities: list[float] = []
    attention: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for bag in bags:
            logit, _, weights = model(prepared[bag.study_id])
            probabilities.append(float(torch.sigmoid(logit).cpu()))
            attention[bag.study_id] = weights.detach().float().cpu().numpy()
    return np.asarray(probabilities, dtype=np.float64), attention


def _train_model(
    train_bags: list[StudyBag],
    train_features: dict[str, torch.Tensor],
    config: SmoothAttentionMILConfig,
    *,
    alpha: float,
    seed: int,
    learning_rate: float,
    weight_decay: float,
    max_epochs: int,
    validation_bags: list[StudyBag] | None = None,
    validation_features: dict[str, torch.Tensor] | None = None,
    patience: int = 30,
    minimum_epochs: int = 20,
) -> tuple[dict[str, torch.Tensor], int, float | None]:
    _set_seed(seed)
    device = next(iter(train_features.values())).device
    model = SmoothAttentionMIL(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    positives = sum(bag.truth for bag in train_bags)
    negatives = len(train_bags) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("Training bags must contain both classes")
    positive_weight = torch.tensor(negatives / positives, device=device)
    rng = np.random.default_rng(seed)
    best_state: dict[str, torch.Tensor] | None = None
    best_auc = -math.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        for index in rng.permutation(len(train_bags)):
            bag = train_bags[int(index)]
            optimizer.zero_grad(set_to_none=True)
            logit, attention_logits, _ = model(train_features[bag.study_id])
            target = torch.tensor(float(bag.truth), device=device)
            total, _, _ = smooth_attention_loss(
                logit,
                target,
                attention_logits,
                alpha=alpha,
                positive_weight=positive_weight,
                smoothness_order=1,
            )
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

        if validation_bags is None:
            continue
        if validation_features is None:
            raise ValueError("validation_features are required with validation_bags")
        truth = np.asarray([bag.truth for bag in validation_bags], dtype=np.int64)
        probability, _ = _predict(model, validation_bags, validation_features)
        auc = float(roc_auc_score(truth, probability))
        if auc > best_auc + 1e-12:
            best_auc = auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epoch >= minimum_epochs and epochs_without_improvement >= patience:
            break

    if validation_bags is None:
        return copy.deepcopy(model.state_dict()), max_epochs, None
    if best_state is None:
        raise RuntimeError("Inner validation did not produce a checkpoint")
    return best_state, best_epoch, best_auc


def _inner_splits(
    bags: list[StudyBag],
    n_splits: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    truth = np.asarray([bag.truth for bag in bags], dtype=np.int64)
    groups = np.asarray([bag.patient_id for bag in bags])
    splitter = StratifiedGroupKFold(
        n_splits=n_splits, shuffle=True, random_state=seed
    )
    splits = list(splitter.split(np.zeros(len(bags)), truth, groups))
    for train_index, validation_index in splits:
        train_patients = {bags[int(i)].patient_id for i in train_index}
        validation_patients = {bags[int(i)].patient_id for i in validation_index}
        if train_patients.intersection(validation_patients):
            raise RuntimeError("Patient leakage in inner split")
        if np.unique(truth[validation_index]).size != 2:
            raise RuntimeError("Inner validation split lacks one class")
    return splits


def _selected_final_epochs(selected: dict[str, object]) -> int:
    """Use the nested best epoch directly; never impose the stopping floor."""
    return max(1, int(selected["median_best_epoch"]))


def _save_checkpoint(
    path: Path,
    state: dict[str, torch.Tensor],
    config: SmoothAttentionMILConfig,
    standardizer: Standardizer,
    metadata: dict[str, object],
) -> None:
    payload = {
        "state_dict": {key: value.detach().cpu() for key, value in state.items()},
        "config": asdict(config),
        "standardizer": {
            "embedding_mean": torch.from_numpy(standardizer.embedding_mean),
            "embedding_scale": torch.from_numpy(standardizer.embedding_scale),
            "score_mean": standardizer.score_mean,
            "score_scale": standardizer.score_scale,
        },
        "metadata": metadata,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alphas", default="0,0.01,0.1,0.5")
    parser.add_argument("--inner-splits", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--attention-dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--final-seeds", default="42,43,44")
    parser.add_argument("--split-seed", type=int, default=20260831)
    parser.add_argument("--device", default="0")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--run-name")
    parser.add_argument("--disable-mlflow", action="store_true")
    args = parser.parse_args()

    if args.outer_fold not in range(5):
        raise ValueError("outer-fold must be in [0, 4]")
    alphas = [float(value) for value in args.alphas.split(",")]
    final_seeds = [int(value) for value in args.final_seeds.split(",")]
    if not alphas or any(not 0.0 <= alpha <= 1.0 for alpha in alphas):
        raise ValueError("alphas must be a non-empty list within [0, 1]")
    if not final_seeds:
        raise ValueError("final-seeds must not be empty")
    if args.torch_threads < 1:
        raise ValueError("torch-threads must be positive")
    torch.set_num_threads(args.torch_threads)
    args.output.mkdir(parents=True, exist_ok=True)

    slices = pd.read_csv(
        args.cache / "slices.csv", dtype={"study_id": str, "patient_id": str}
    )
    embeddings = np.load(
        args.cache / "embeddings.npy", allow_pickle=False, mmap_mode="r"
    )
    slice_scores = np.load(args.cache / "slice_scores.npy", allow_pickle=False)
    cache_manifest = json.loads(
        (args.cache / "manifest.json").read_text(encoding="utf-8")
    )
    if int(cache_manifest["outer_model_fold"]) != args.outer_fold:
        raise ValueError("Cache model fold does not match --outer-fold")
    bags = _study_bags(slices)
    outer_train = [bag for bag in bags if bag.outer_fold != args.outer_fold]
    outer_validation = [bag for bag in bags if bag.outer_fold == args.outer_fold]
    if {bag.patient_id for bag in outer_train}.intersection(
        {bag.patient_id for bag in outer_validation}
    ):
        raise RuntimeError("Patient leakage between outer train and validation")
    device = torch.device(
        f"cuda:{args.device}" if args.device != "cpu" and torch.cuda.is_available() else "cpu"
    )
    model_config = SmoothAttentionMILConfig(
        input_dim=int(embeddings.shape[1]) + 1,
        hidden_dim=args.hidden_dim,
        attention_dim=args.attention_dim,
        dropout=args.dropout,
    )
    run_config = {
        "outer_fold": args.outer_fold,
        "alphas": alphas,
        "inner_splits": args.inner_splits,
        "model": asdict(model_config),
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "max_epochs": args.max_epochs,
        "patience": args.patience,
        "minimum_epochs": args.minimum_epochs,
        "final_seeds": final_seeds,
        "split_seed": args.split_seed,
        "torch_threads": args.torch_threads,
        "cache_manifest": cache_manifest,
    }
    context = ExperimentContext(
        task_key="fracture",
        run_name=args.run_name or f"fracture-sa-mil-f{args.outer_fold}",
        run_config=run_config,
        strategy="frozen-yolov8s-smooth-attention-mil",
        tags={"outer_fold": str(args.outer_fold), "stage": "nested-oof"},
        notes="Nested patient-disjoint SA-MIL alpha selection on frozen YOLOv8s features.",
    )

    def execute() -> dict[str, object]:
        splits = _inner_splits(outer_train, args.inner_splits, args.split_seed)
        inner_results: list[dict[str, object]] = []
        for alpha in alphas:
            for split_number, (train_index, validation_index) in enumerate(splits):
                train_bags = [outer_train[int(index)] for index in train_index]
                validation_bags = [outer_train[int(index)] for index in validation_index]
                standardizer = _fit_standardizer(embeddings, slice_scores, train_bags)
                train_features = _prepare_features(
                    embeddings, slice_scores, train_bags, standardizer, device
                )
                validation_features = _prepare_features(
                    embeddings, slice_scores, validation_bags, standardizer, device
                )
                _, best_epoch, best_auc = _train_model(
                    train_bags,
                    train_features,
                    model_config,
                    alpha=alpha,
                    seed=args.split_seed + split_number,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    max_epochs=args.max_epochs,
                    validation_bags=validation_bags,
                    validation_features=validation_features,
                    patience=args.patience,
                    minimum_epochs=args.minimum_epochs,
                )
                inner_results.append(
                    {
                        "alpha": alpha,
                        "split": split_number,
                        "best_epoch": best_epoch,
                        "auc": best_auc,
                        "n_train": len(train_bags),
                        "n_validation": len(validation_bags),
                        "validation_positives": sum(bag.truth for bag in validation_bags),
                    }
                )
                print(inner_results[-1], flush=True)
        alpha_summary = []
        for alpha in alphas:
            rows = [row for row in inner_results if row["alpha"] == alpha]
            alpha_summary.append(
                {
                    "alpha": alpha,
                    "mean_auc": float(np.mean([float(row["auc"]) for row in rows])),
                    "worst_auc": float(np.min([float(row["auc"]) for row in rows])),
                    "median_best_epoch": int(
                        np.median([int(row["best_epoch"]) for row in rows])
                    ),
                }
            )
        selected = max(
            alpha_summary,
            key=lambda row: (row["mean_auc"], row["worst_auc"], -row["alpha"]),
        )
        selected_alpha = float(selected["alpha"])
        # Final training must reproduce the epoch selected inside the nested
        # validation loop.  minimum_epochs controls when early stopping may
        # terminate, but must not overwrite an earlier best checkpoint.
        final_epochs = _selected_final_epochs(selected)
        standardizer = _fit_standardizer(embeddings, slice_scores, outer_train)
        train_features = _prepare_features(
            embeddings, slice_scores, outer_train, standardizer, device
        )
        validation_features = _prepare_features(
            embeddings, slice_scores, outer_validation, standardizer, device
        )
        seed_probabilities: list[np.ndarray] = []
        seed_attention: list[dict[str, np.ndarray]] = []
        for seed in final_seeds:
            state, _, _ = _train_model(
                outer_train,
                train_features,
                model_config,
                alpha=selected_alpha,
                seed=seed,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                max_epochs=final_epochs,
            )
            model = SmoothAttentionMIL(model_config).to(device)
            model.load_state_dict(state)
            probability, attention = _predict(
                model, outer_validation, validation_features
            )
            seed_probabilities.append(probability)
            seed_attention.append(attention)
            _save_checkpoint(
                args.output / f"model_seed{seed}.pt",
                state,
                model_config,
                standardizer,
                {
                    "outer_fold": args.outer_fold,
                    "selected_alpha": selected_alpha,
                    "epochs": final_epochs,
                    "seed": seed,
                    "feature_checkpoint_sha256": cache_manifest[
                        "checkpoint_sha256"
                    ],
                },
            )
        ensemble_probability = np.mean(np.stack(seed_probabilities), axis=0)
        prediction_rows: list[dict[str, object]] = []
        attention_rows: list[dict[str, object]] = []
        for bag_index, bag in enumerate(outer_validation):
            pooled = aggregate_study_scores(slice_scores[bag.indices])
            prediction_rows.append(
                {
                    "study_id": bag.study_id,
                    "patient_id": bag.patient_id,
                    "truth": bag.truth,
                    "outer_fold": bag.outer_fold,
                    **{f"prob_{name}": value for name, value in pooled.items()},
                    **{
                        f"mil_seed{seed}": float(seed_probabilities[index][bag_index])
                        for index, seed in enumerate(final_seeds)
                    },
                    "mil_score": float(ensemble_probability[bag_index]),
                }
            )
            mean_attention = np.mean(
                np.stack([values[bag.study_id] for values in seed_attention]), axis=0
            )
            bag_slice_rows = slices.iloc[bag.indices]
            for row_index, (_, slice_row) in enumerate(bag_slice_rows.iterrows()):
                attention_rows.append(
                    {
                        "study_id": bag.study_id,
                        "truth": bag.truth,
                        "slice_index": int(slice_row["slice_index"]),
                        "slice_score": float(slice_scores[bag.indices[row_index]]),
                        "attention_weight": float(mean_attention[row_index]),
                    }
                )
        predictions = pd.DataFrame(prediction_rows)
        predictions.to_csv(args.output / "study_predictions.csv", index=False)
        pd.DataFrame(attention_rows).to_csv(
            args.output / "slice_attention.csv", index=False
        )
        truth = predictions["truth"].to_numpy(dtype=np.int64)
        metrics = {
            "outer_fold": args.outer_fold,
            "n_train": len(outer_train),
            "n_validation": len(outer_validation),
            "n_validation_positive": int(truth.sum()),
            "selected_alpha": selected_alpha,
            "final_epochs": final_epochs,
            "mil_auc": float(roc_auc_score(truth, predictions["mil_score"])),
            "pooling_auc": {
                name: float(roc_auc_score(truth, predictions[f"prob_{name}"]))
                for name in (
                    "max",
                    "top2_mean",
                    "top3_mean",
                    "top5_mean",
                    "adjacent_pair",
                    "window3_mean",
                    "noisy_or",
                )
            },
            "alpha_summary": alpha_summary,
            "inner_results": inner_results,
            "final_seeds": final_seeds,
        }
        (args.output / "metrics.json").write_text(
            json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
        )
        return metrics

    if args.disable_mlflow:
        metrics = execute()
    else:
        import mlflow

        with experiment_run(context):
            active = mlflow.active_run()
            if active is None:
                raise RuntimeError("MLflow run was not activated")
            (args.output / "mlflow_run.json").write_text(
                json.dumps({"run_id": active.info.run_id}, indent=2) + "\n",
                encoding="utf-8",
            )
            metrics = execute()
            mlflow.log_metrics(
                {
                    "outer_mil_auc": float(metrics["mil_auc"]),
                    "selected_alpha": float(metrics["selected_alpha"]),
                    "final_epochs": float(metrics["final_epochs"]),
                    **{
                        f"outer_{name}_auc": float(value)
                        for name, value in metrics["pooling_auc"].items()
                    },
                }
            )
            mlflow.set_tags(
                {
                    "validation_scope": "nested_patient_disjoint_outer_fold",
                    "feature_extractor": "fold_specific_yolov8s_epoch10",
                    "sequence_coverage": "union_of_full_validation_manifests",
                }
            )
            mlflow.log_artifacts(str(args.output), artifact_path="smooth_attention_mil")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
