"""
eda_metadata.py — Phase 0.1: Metadata Analysis & Class Distribution

Analyzes the training metadata CSV to understand:
- Triage class distribution (0: Non-urgent, 1: Urgent, 2: Critical)
- AnyICH and SkullFracture prevalence
- Number of slices per patient
- PixelSpacing and SliceThickness variability
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

# === Config ===
BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / "Data" / "metadata" / "training_df.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    """Load the training metadata CSV."""
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} rows from {CSV_PATH}")
    print(f"   Columns: {list(df.columns)}")
    return df


def plot_triage_distribution(df: pd.DataFrame) -> dict:
    """Analyze and plot triage class distribution."""
    counts = df["triage_class"].value_counts().sort_index()
    proportions = counts / counts.sum() * 100

    print("\n=== Triage Class Distribution ===")
    label_map = {0: "Non-urgent (0)", 1: "Urgent (1)", 2: "Critical (2)"}
    for cls in [0, 1, 2]:
        n = counts.get(cls, 0)
        p = proportions.get(cls, 0)
        print(f"  {label_map[cls]}: {n:>5} ({p:.1f}%)")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot
    colors = ["#2ecc71", "#f39c12", "#e74c3c"]
    bars = axes[0].bar(
        [label_map[c] for c in [0, 1, 2]],
        [counts.get(c, 0) for c in [0, 1, 2]],
        color=colors,
        edgecolor="white",
        linewidth=1.2,
    )
    axes[0].set_title("Triage Class Distribution", fontsize=14, fontweight="bold")
    axes[0].set_ylabel("Number of Samples")
    axes[0].grid(axis="y", alpha=0.3)
    for bar, count in zip(bars, [counts.get(c, 0) for c in [0, 1, 2]]):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 20,
            str(count),
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        [counts.get(c, 0) for c in [0, 1, 2]],
        labels=[label_map[c] for c in [0, 1, 2]],
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        explode=(0.02, 0.02, 0.02),
    )
    axes[1].set_title("Triage Class Proportions", fontsize=14, fontweight="bold")

    plt.tight_layout()
    path = OUTPUT_DIR / "triage_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")

    return {int(k): int(v) for k, v in counts.items()}


def plot_binary_flags(df: pd.DataFrame) -> dict:
    """Analyze AnyICH and SkullFracture distributions."""
    results = {}
    for flag_name in ["AnyICH", "SkullFracture"]:
        counts = df[flag_name].value_counts()
        total = len(df)
        print(f"\n=== {flag_name} Distribution ===")
        for val, cnt in counts.items():
            pct = cnt / total * 100
            print(f"  {val}: {cnt:>5} ({pct:.1f}%)")
        results[flag_name] = {str(k): int(v) for k, v in counts.items()}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for idx, flag_name in enumerate(["AnyICH", "SkullFracture"]):
        counts = df[flag_name].value_counts()
        colors_bin = ["#3498db", "#e74c3c"] if flag_name == "AnyICH" else ["#2ecc71", "#e74c3c"]
        bars = axes[idx].bar(
            [str(v) for v in counts.index],
            counts.values,
            color=colors_bin,
            edgecolor="white",
        )
        axes[idx].set_title(f"{flag_name} Distribution", fontsize=12, fontweight="bold")
        axes[idx].grid(axis="y", alpha=0.3)
        for bar, cnt in zip(bars, counts.values):
            pct = cnt / total * 100
            axes[idx].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 15,
                f"{cnt}\n({pct:.1f}%)",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    path = OUTPUT_DIR / "binary_flags_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Saved: {path}")
    return results


def plot_slices_per_patient(df: pd.DataFrame) -> dict:
    """Analyze number of DICOM files per patient/series."""
    n_files = df["dicom_series.NumDicomFiles"]
    stats = {
        "min": int(n_files.min()),
        "max": int(n_files.max()),
        "mean": float(round(n_files.mean(), 1)),
        "median": float(round(n_files.median(), 1)),
        "std": float(round(n_files.std(), 1)),
        "q25": float(round(n_files.quantile(0.25))),
        "q75": float(round(n_files.quantile(0.75))),
    }
    print("\n=== Slices Per Patient/Series ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(n_files, bins=50, color="#9b59b6", edgecolor="white", alpha=0.8)
    axes[0].axvline(n_files.median(), color="red", linestyle="--", label=f"Median={n_files.median():.0f}")
    axes[0].axvline(n_files.mean(), color="orange", linestyle="--", label=f"Mean={n_files.mean():.1f}")
    axes[0].set_xlabel("Number of DICOM Files")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Distribution of Slices Per Series", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].boxplot(n_files, vert=True, patch_artist=True, boxprops=dict(facecolor="#9b59b6", alpha=0.6))


    axes[1].set_ylabel("Number of DICOM Files")
    axes[1].set_title("Boxplot: Slices Per Series", fontsize=13, fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "slices_per_patient.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return stats


def plot_spacing_analysis(df: pd.DataFrame) -> dict:
    """Analyze PixelSpacing and SliceThickness."""
    spacing_x = df["dicom_series.PixelSpacing0"]
    spacing_y = df["dicom_series.PixelSpacing1"]
    thickness = df["dicom_series.SliceThickness"]

    stats = {
        "pixel_spacing_x": {
            "unique": int(spacing_x.nunique()),
            "values": sorted(spacing_x.unique().round(4).tolist()),
            "min": float(spacing_x.min()),
            "max": float(spacing_x.max()),
        },
        "pixel_spacing_y": {
            "unique": int(spacing_y.nunique()),
            "values": sorted(spacing_y.unique().round(4).tolist()),
        },
        "slice_thickness": {
            "unique": int(thickness.nunique()),
            "values": sorted(thickness.unique().round(4).tolist()),
            "min": float(thickness.min()),
            "max": float(thickness.max()),
        },
    }

    print("\n=== Pixel Spacing & Slice Thickness ===")
    print(f"  PixelSpacingX unique values: {spacing_x.nunique()}")
    print(f"  PixelSpacingY unique values: {spacing_y.nunique()}")
    print(f"  SliceThickness unique values: {thickness.nunique()}")
    print(f"  SliceThickness range: {thickness.min():.4f} - {thickness.max():.4f} mm")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(spacing_x, bins=30, color="#e67e22", edgecolor="white", alpha=0.8)
    axes[0].set_xlabel("Pixel Spacing X (mm)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Pixel Spacing X", fontsize=12, fontweight="bold")
    axes[0].grid(alpha=0.3)

    axes[1].hist(spacing_y, bins=30, color="#1abc9c", edgecolor="white", alpha=0.8)
    axes[1].set_xlabel("Pixel Spacing Y (mm)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Pixel Spacing Y", fontsize=12, fontweight="bold")
    axes[1].grid(alpha=0.3)

    axes[2].hist(thickness, bins=30, color="#3498db", edgecolor="white", alpha=0.8)
    axes[2].set_xlabel("Slice Thickness (mm)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Slice Thickness", fontsize=12, fontweight="bold")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "spacing_analysis.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return stats


def plot_triage_vs_features(df: pd.DataFrame) -> dict:
    """Cross-tabulate triage_class with AnyICH and SkullFracture."""
    results = {}
    for flag_name in ["AnyICH", "SkullFracture"]:
        cross = pd.crosstab(df["triage_class"], df[flag_name], margins=True)
        print(f"\n=== Triage Class vs {flag_name} ===")
        print(cross)
        results[flag_name] = cross.to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for idx, flag_name in enumerate(["AnyICH", "SkullFracture"]):
        cross_pct = pd.crosstab(df["triage_class"], df[flag_name], normalize="index") * 100
        cross_pct.plot(
            kind="bar",
            ax=axes[idx],
            color=["#3498db", "#e74c3c"],
            edgecolor="white",
            legend=True,
        )
        axes[idx].set_title(f"Triage Class vs {flag_name}", fontsize=12, fontweight="bold")
        axes[idx].set_xlabel("Triage Class")
        axes[idx].set_ylabel("Percentage (%)")
        axes[idx].grid(axis="y", alpha=0.3)
        axes[idx].legend(title=flag_name)

    plt.tight_layout()
    path = OUTPUT_DIR / "triage_vs_features.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return results


def main():
    print("=" * 60)
    print("  Phase 0.1: Metadata Analysis & Class Distribution")
    print("=" * 60)

    df = load_data()

    # Store all results
    all_results = {
        "total_samples": len(df),
        "unique_patients": int(df["dicom_series.PatientID"].nunique()),
        "unique_series": int(df["dicom_series.id"].nunique()),
    }

    all_results["triage_distribution"] = plot_triage_distribution(df)
    all_results["binary_flags"] = plot_binary_flags(df)
    all_results["slices_per_patient"] = plot_slices_per_patient(df)
    all_results["spacing"] = plot_spacing_analysis(df)
    all_results["triage_vs_features"] = plot_triage_vs_features(df)

    # Save numerical results as JSON
    json_path = OUTPUT_DIR / "eda_metadata_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Saved numerical results: {json_path}")
    print("\n✅ Phase 0.1 complete!")


if __name__ == "__main__":
    main()
