"""
eda_volumes.py — Phase 0.2: Hemorrhage Volume Distribution Analysis

Analyzes ICH volume distributions for each subtype:
- Convert pixel area to volume (mL) using PixelSpacing
- Histograms for all 5 hemorrhage types
- Boxplot by triage class
- Correlation matrix between hemorrhage types
- Cumulative distribution analysis relative to competition thresholds
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / "Data" / "metadata" / "training_df.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ICH types and their area columns in CSV
ICH_TYPES = {
    "IVH": "IntraventricularHemorrhage",
    "IPH": "IntraparenchymalHemorrhage",
    "SAH": "SubarachnoidHemorrhage",
    "EDH": "EpiduralHemorrhage",
    "SDH": "SubduralHemorrhage",
}

# Competition triage thresholds (for reference in analysis)
TRIAGE_THRESHOLDS = {
    "EDH_CRIT": 30.0,
    "SDH_CRIT": 70.0,
    "IPH_CRIT": 70.0,
    "TOTAL_VOL_CRIT": 60.0,
    "COMBO_VOL": 40.0,
    "COMBO_MLS": 3.0,
    "FRAC_VOL_CRIT": 15.0,
    "EPS_VOLUME": 0.1,
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} rows")
    return df


def compute_volumes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pixel-area columns to volume (mL).
    Volume = area_pixels * pixel_spacing_x * pixel_spacing_y * slice_thickness / 1000
    (1 mm^3 = 0.001 mL)
    """
    df_v = df.copy()
    spacing_x = df_v["dicom_series.PixelSpacing0"]
    spacing_y = df_v["dicom_series.PixelSpacing1"]
    thickness = df_v["dicom_series.SliceThickness"]

    for short_name, col_name in ICH_TYPES.items():
        area_col = f"{col_name}_Area"
        if area_col in df_v.columns:
            # Volume in mL: area (pixels) * spacing_x * spacing_y * thickness / 1000
            df_v[f"V_{short_name}"] = (
                df_v[area_col].fillna(0)
                * spacing_x * spacing_y * thickness / 1000.0
            )
        else:
            print(f"⚠️  Column '{area_col}' not found, setting V_{short_name}=0")
            df_v[f"V_{short_name}"] = 0.0

    df_v["V_total"] = sum(df_v[f"V_{t}"] for t in ICH_TYPES)
    return df_v


def plot_volume_histograms(df_v: pd.DataFrame) -> dict:
    """Plot histograms of volumes for each ICH type."""
    print("\n=== Volume Distribution ===")
    stats = {}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for idx, (short_name, _) in enumerate(ICH_TYPES.items()):
        vol = df_v[f"V_{short_name}"]
        pos_vol = vol[vol > 0]

        n_pos = len(pos_vol)
        n_total = len(vol)
        pct_pos = n_pos / n_total * 100 if n_total > 0 else 0

        stats[short_name] = {
            "n_positive": int(n_pos),
            "pct_positive": round(pct_pos, 1),
            "mean_vol": float(round(vol.mean(), 2)),
            "median_vol": float(round(vol.median(), 2)),
            "std_vol": float(round(vol.std(), 2)),
            "max_vol": float(round(vol.max(), 2)),
            "min_positive_vol": float(round(pos_vol.min(), 4)) if len(pos_vol) > 0 else 0,
        }

        print(f"  {short_name}: {n_pos}/{n_total} positive ({pct_pos:.1f}%)")
        print(f"      Mean={vol.mean():.2f} mL, Median={vol.median():.2f} mL, Max={vol.max():.2f} mL")

        # Histogram (log scale for better visualization)
        if n_pos > 0:
            axes_flat[idx].hist(pos_vol, bins=50, color="#e74c3c", alpha=0.7, edgecolor="white")
            axes_flat[idx].axvline(
                pos_vol.median(), color="blue", linestyle="--", label=f"Median={pos_vol.median():.1f}"
            )
            axes_flat[idx].set_yscale("log")
        axes_flat[idx].set_xlabel(f"{short_name} Volume (mL)")
        axes_flat[idx].set_ylabel("Frequency (log scale)" if n_pos > 0 else "Frequency")
        axes_flat[idx].set_title(f"{short_name} — Positive Cases: {n_pos} ({pct_pos:.1f}%)", fontsize=11)
        axes_flat[idx].legend() if n_pos > 0 else None
        axes_flat[idx].grid(alpha=0.3)

    # Total volume histogram
    total_vol = df_v["V_total"]
    pos_total = total_vol[total_vol > 0]
    n_pos_total = len(pos_total)
    pct_pos_total = n_pos_total / len(total_vol) * 100
    stats["total"] = {
        "n_positive": int(n_pos_total),
        "pct_positive": round(pct_pos_total, 1),
        "mean_vol": float(round(total_vol.mean(), 2)),
        "median_vol": float(round(total_vol.median(), 2)),
        "std_vol": float(round(total_vol.std(), 2)),
        "max_vol": float(round(total_vol.max(), 2)),
    }
    print(f"  TOTAL: {n_pos_total}/{len(total_vol)} positive ({pct_pos_total:.1f}%)")
    print(f"      Mean={total_vol.mean():.2f} mL, Median={total_vol.median():.2f} mL, Max={total_vol.max():.2f} mL")

    if n_pos_total > 0:
        axes_flat[5].hist(pos_total, bins=50, color="#8e44ad", alpha=0.7, edgecolor="white")
        axes_flat[5].axvline(
            pos_total.median(), color="blue", linestyle="--", label=f"Median={pos_total.median():.1f}"
        )
        axes_flat[5].set_yscale("log")
    axes_flat[5].set_xlabel("Total ICH Volume (mL)")
    axes_flat[5].set_ylabel("Frequency (log scale)" if n_pos_total > 0 else "Frequency")
    axes_flat[5].set_title(f"TOTAL ICH — Positive: {n_pos_total} ({pct_pos_total:.1f}%)", fontsize=11)
    axes_flat[5].grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "volume_histograms.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Saved: {path}")
    return stats


def plot_volume_by_triage(df_v: pd.DataFrame) -> dict:
    """Boxplot of volumes grouped by triage class."""
    print("\n=== Volume by Triage Class ===")
    results = {}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes_flat = axes.flatten()

    for idx, (short_name, _) in enumerate(ICH_TYPES.items()):
        vol_col = f"V_{short_name}"
        data = []
        labels = []
        for cls in sorted(df_v["triage_class"].unique()):
            subset = df_v[df_v["triage_class"] == cls][vol_col]
            # Add small epsilon for log scale
            data.append(subset.values + 1e-6)
            labels.append(f"Class {cls}")

        bp = axes_flat[idx].boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
        colors_class = ["#2ecc71", "#f39c12", "#e74c3c"]
        for patch, color in zip(bp["boxes"], colors_class):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        axes_flat[idx].set_yscale("log")
        axes_flat[idx].set_ylabel(f"{short_name} Volume (mL) — log scale")
        axes_flat[idx].set_title(f"{short_name} by Triage Class", fontsize=11)
        axes_flat[idx].grid(alpha=0.3)

        # Compute stats
        class_stats = {}
        for cls in sorted(df_v["triage_class"].unique()):
            subset = df_v[df_v["triage_class"] == cls][vol_col]
            class_stats[int(cls)] = {
                "mean": float(round(subset.mean(), 2)),
                "median": float(round(subset.median(), 2)),
                "max": float(round(subset.max(), 2)),
                "n_positive": int((subset > 0).sum()),
            }
        results[short_name] = class_stats

    # Total volume by triage
    data_total = []
    for cls in sorted(df_v["triage_class"].unique()):
        data_total.append(df_v[df_v["triage_class"] == cls]["V_total"].values + 1e-6)
    bp = axes_flat[5].boxplot(data_total, tick_labels=["Class 0", "Class 1", "Class 2"], patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#2ecc71", "#f39c12", "#e74c3c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes_flat[5].set_yscale("log")
    axes_flat[5].set_ylabel("Total ICH Volume (mL) — log scale")
    axes_flat[5].set_title("Total ICH Volume by Triage Class", fontsize=11)
    axes_flat[5].grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "volume_by_triage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return results


def plot_correlation_matrix(df_v: pd.DataFrame) -> None:
    """Correlation matrix between ICH types."""
    vol_cols = [f"V_{t}" for t in ICH_TYPES] + ["V_total"]
    corr = df_v[vol_cols].corr()

    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
        vmin=-1, vmax=1, center=0, square=True,
        linewidths=1, cbar_kws={"shrink": 0.8},
    )
    plt.title("Correlation Matrix of ICH Volumes", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = OUTPUT_DIR / "volume_correlation_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")

    print("\n=== Volume Correlation Matrix ===")
    print(corr.round(3))


def analyze_thresholds(df_v: pd.DataFrame) -> dict:
    """
    Analyze how competition thresholds apply to real data.
    For each threshold, what fraction of cases would be flagged?
    """
    print("\n=== Threshold Analysis (Competition Rules) ===")
    total = len(df_v)
    threshold_stats = {}

    checks = {
        "EDH >= 30mL (EDH_CRIT)": df_v["V_EDH"] >= 30,
        "SDH >= 70mL (SDH_CRIT)": df_v["V_SDH"] >= 70,
        "IPH >= 70mL (IPH_CRIT)": df_v["V_IPH"] >= 70,
        "Total >= 60mL (TOTAL_VOL_CRIT)": df_v["V_total"] >= 60,
        "Total >= 40mL + MLS >= 3mm (COMBO)": (df_v["V_total"] >= 40) & (df_v["MidlineShiftMM"] >= 3),
        "Total >= 15mL + Fracture (FRAC)": (df_v["V_total"] >= 15) & (df_v["SkullFracture"] == True),
        "Any ICH >= 0.1mL (EPS_VOLUME)": df_v["V_total"] >= 0.1,
    }

    for label, condition in checks.items():
        count = condition.sum()
        pct = count / total * 100
        threshold_stats[label] = {"count": int(count), "pct": round(pct, 2)}
        print(f"  {label}: {count:>5} cases ({pct:.2f}%)")

    # Multiple threshold crossings per case
    n_checks = int(sum(condition.sum() for condition in checks.values()))
    print(f"\n  Total cases: {total}")
    print(f"  Average thresholds triggered per case: {n_checks / total:.2f}")

    return threshold_stats


def plot_cumulative_distribution(df_v: pd.DataFrame) -> dict:
    """Cumulative distribution of volumes at key thresholds."""
    vol_cols = [f"V_{t}" for t in ICH_TYPES] + ["V_total"]
    thresholds = [0.1, 1, 5, 10, 15, 20, 30, 40, 50, 60, 70, 100]

    print("\n=== Cumulative Distribution ===")
    cum_stats = {}
    for col in vol_cols:
        cum_stats[col] = {}
        for t in thresholds:
            pct = (df_v[col] >= t).mean() * 100
            cum_stats[col][t] = round(pct, 2)

    fig, ax = plt.subplots(figsize=(12, 6))
    for col in vol_cols:
        pcts = [(df_v[col] >= t).mean() * 100 for t in thresholds]
        label = col.replace("V_", "")
        ax.plot(thresholds, pcts, marker="o", label=label, linewidth=2)

    ax.axhline(50, color="gray", linestyle=":", alpha=0.5, label="50% mark")
    ax.set_xlabel("Volume Threshold (mL)")
    ax.set_ylabel("Percentage of Cases Above Threshold (%)")
    ax.set_title("Cumulative Distribution: Cases Above Volume Thresholds", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "volume_cumulative_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return cum_stats


def main():
    print("=" * 60)
    print("  Phase 0.2: Hemorrhage Volume Distribution Analysis")
    print("=" * 60)

    df = load_data()
    df_v = compute_volumes(df)

    results = {}
    results["volume_distribution"] = plot_volume_histograms(df_v)
    results["volume_by_triage"] = plot_volume_by_triage(df_v)
    plot_correlation_matrix(df_v)
    results["threshold_analysis"] = analyze_thresholds(df_v)
    results["cumulative_distribution"] = plot_cumulative_distribution(df_v)

    # Save results
    json_path = OUTPUT_DIR / "eda_volumes_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Saved: {json_path}")
    print("\n✅ Phase 0.2 complete!")


if __name__ == "__main__":
    main()
