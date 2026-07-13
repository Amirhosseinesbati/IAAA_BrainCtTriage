"""
eda_mls.py — Phase 0.3: Midline Shift (MLS) Analysis

Analyzes MLS distribution:
- Basic statistics and histogram
- MLS by triage class
- MLS vs ICH presence
- Threshold analysis (1mm, 3mm, 5mm)
- Correlation between MLS and ICH volume
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

MLS_THRESHOLDS = {"EPS_MLS": 1.0, "MLS_URGENT_LOW": 3.0, "MLS_CRITICAL": 5.0}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} rows")
    return df


def plot_mls_distribution(df: pd.DataFrame) -> dict:
    """Histogram and basic stats of MLS."""
    mls = df["MidlineShiftMM"]
    stats = {
        "count": int(len(mls)),
        "n_zero": int((mls == 0).sum()),
        "pct_zero": round((mls == 0).mean() * 100, 2),
        "min": float(round(mls.min(), 2)),
        "max": float(round(mls.max(), 2)),
        "mean": float(round(mls.mean(), 3)),
        "median": float(round(mls.median(), 3)),
        "std": float(round(mls.std(), 3)),
        "q25": float(round(mls.quantile(0.25), 3)),
        "q75": float(round(mls.quantile(0.75), 3)),
        "p99": float(round(mls.quantile(0.99), 3)),
    }

    print("\n=== MLS Distribution ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Full histogram
    axes[0].hist(mls, bins=100, color="#3498db", edgecolor="white", alpha=0.8)
    axes[0].axvline(MLS_THRESHOLDS["MLS_CRITICAL"], color="red", linestyle="--", linewidth=2,
                    label=f"Critical ≥ {MLS_THRESHOLDS['MLS_CRITICAL']}mm")
    axes[0].axvline(MLS_THRESHOLDS["MLS_URGENT_LOW"], color="orange", linestyle="--", linewidth=2,
                    label=f"Urgent ≥ {MLS_THRESHOLDS['MLS_URGENT_LOW']}mm")
    axes[0].axvline(MLS_THRESHOLDS["EPS_MLS"], color="green", linestyle=":", linewidth=2,
                    label=f"Minimal ≥ {MLS_THRESHOLDS['EPS_MLS']}mm")
    axes[0].set_xlabel("Midline Shift (mm)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("MLS Distribution with Thresholds", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Zoom into 0-15mm
    mls_zoom = mls[mls <= 15]
    axes[1].hist(mls_zoom, bins=80, color="#2ecc71", edgecolor="white", alpha=0.8)
    for thresh_name, thresh_val in MLS_THRESHOLDS.items():
        color_map = {"EPS_MLS": "green", "MLS_URGENT_LOW": "orange", "MLS_CRITICAL": "red"}
        axes[1].axvline(thresh_val, color=color_map[thresh_name], linestyle="--", linewidth=1.5)
    axes[1].set_xlabel("Midline Shift (mm)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("MLS Distribution (Zoom: 0-15mm)", fontsize=13, fontweight="bold")
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "mls_distribution.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return stats


def plot_mls_by_triage(df: pd.DataFrame) -> dict:
    """MLS distribution by triage class."""
    print("\n=== MLS by Triage Class ===")
    results = {}

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot
    data = []
    labels = []
    for cls in sorted(df["triage_class"].unique()):
        subset = df[df["triage_class"] == cls]["MidlineShiftMM"]
        data.append(subset.values)
        labels.append(f"Class {cls}")

    bp = axes[0].boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    colors_class = ["#2ecc71", "#f39c12", "#e74c3c"]
    for patch, color in zip(bp["boxes"], colors_class):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel("Midline Shift (mm)")
    axes[0].set_title("MLS by Triage Class (Boxplot)", fontsize=13, fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Violin plot
    parts = axes[1].violinplot(
        [df[df["triage_class"] == c]["MidlineShiftMM"].values for c in sorted(df["triage_class"].unique())],
        positions=[0, 1, 2],
        showmedians=True,
        showextrema=True,
    )
    for pc, color in zip(parts["bodies"], colors_class):
        pc.set_facecolor(color)
        pc.set_alpha(0.5)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["Class 0", "Class 1", "Class 2"])
    axes[1].set_ylabel("Midline Shift (mm)")
    axes[1].set_title("MLS by Triage Class (Violin Plot)", fontsize=13, fontweight="bold")
    axes[1].grid(alpha=0.3)

    for cls in sorted(df["triage_class"].unique()):
        subset = df[df["triage_class"] == cls]["MidlineShiftMM"]
        s = {
            "mean": float(round(subset.mean(), 3)),
            "median": float(round(subset.median(), 3)),
            "max": float(round(subset.max(), 2)),
            "n_pos": int((subset > 0).sum()),
            "pct_pos": round((subset > 0).mean() * 100, 1),
        }
        results[int(cls)] = s
        print(f"  Class {cls}: Mean={s['mean']:.3f}, Median={s['median']:.3f}, Max={s['max']:.2f}, "
              f"Pos={s['n_pos']}/{len(subset)} ({s['pct_pos']}%)")

    plt.tight_layout()
    path = OUTPUT_DIR / "mls_by_triage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return results


def plot_mls_vs_ich(df: pd.DataFrame) -> dict:
    """MLS in cases with and without ICH."""
    has_ich = df["AnyICH"] == True
    no_ich = df["AnyICH"] == False

    stats = {
        "has_ich": {
            "mean_mls": float(round(df[has_ich]["MidlineShiftMM"].mean(), 3)),
            "median_mls": float(round(df[has_ich]["MidlineShiftMM"].median(), 3)),
            "max_mls": float(round(df[has_ich]["MidlineShiftMM"].max(), 2)),
            "n": int(has_ich.sum()),
        },
        "no_ich": {
            "mean_mls": float(round(df[no_ich]["MidlineShiftMM"].mean(), 3)),
            "median_mls": float(round(df[no_ich]["MidlineShiftMM"].median(), 3)),
            "max_mls": float(round(df[no_ich]["MidlineShiftMM"].max(), 2)),
            "n": int(no_ich.sum()),
        },
    }

    print("\n=== MLS with/without ICH ===")
    for group, s in stats.items():
        print(f"  {group}: n={s['n']}, Mean={s['mean_mls']:.3f}, Median={s['median_mls']:.3f}, Max={s['max_mls']:.2f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Boxplot comparison
    data = [df[no_ich]["MidlineShiftMM"].values, df[has_ich]["MidlineShiftMM"].values]
    bp = axes[0].boxplot(data, tick_labels=["No ICH", "Has ICH"], patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#3498db", "#e74c3c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    axes[0].set_ylabel("Midline Shift (mm)")
    axes[0].set_title("MLS: ICH vs No ICH", fontsize=13, fontweight="bold")
    axes[0].grid(alpha=0.3)

    # Scatter: MLS vs Total ICH Volume
    axes[1].scatter(
        df[no_ich]["MidlineShiftMM"], df[no_ich].get("MidlineShiftMM", 0),
        alpha=0.3, s=10, color="#3498db", label="No ICH"
    )
    # We need volume data - try to compute it
    spacing_x = df["dicom_series.PixelSpacing0"]
    spacing_y = df["dicom_series.PixelSpacing1"]
    thickness = df["dicom_series.SliceThickness"]
    area_cols = [f"{name}_Area" for name in
                 ["IntraventricularHemorrhage", "IntraparenchymalHemorrhage",
                  "SubarachnoidHemorrhage", "EpiduralHemorrhage", "SubduralHemorrhage"]]
    total_vol = sum(df[c].fillna(0) * spacing_x * spacing_y * thickness / 1000.0 for c in area_cols)

    axes[1].scatter(
        df[has_ich]["MidlineShiftMM"], total_vol[has_ich],
        alpha=0.5, s=15, color="#e74c3c", label="Has ICH"
    )
    axes[1].set_xlabel("MLS (mm)")
    axes[1].set_ylabel("Total ICH Volume (mL)")
    axes[1].set_title("MLS vs Total ICH Volume", fontsize=13, fontweight="bold")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "mls_vs_ich.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return stats


def plot_mls_threshold_analysis(df: pd.DataFrame) -> dict:
    """Analyze how many cases fall into each MLS threshold bucket."""
    mls = df["MidlineShiftMM"]

    buckets = {
        "No MLS (0mm)": (mls == 0),
        f"Minimal (0–{MLS_THRESHOLDS['EPS_MLS']}mm)": (mls > 0) & (mls < MLS_THRESHOLDS["EPS_MLS"]),
        f"Minor ({MLS_THRESHOLDS['EPS_MLS']}–{MLS_THRESHOLDS['MLS_URGENT_LOW']}mm)": (
            mls >= MLS_THRESHOLDS["EPS_MLS"]) & (mls < MLS_THRESHOLDS["MLS_URGENT_LOW"]),
        f"Moderate ({MLS_THRESHOLDS['MLS_URGENT_LOW']}–{MLS_THRESHOLDS['MLS_CRITICAL']}mm)": (
            mls >= MLS_THRESHOLDS["MLS_URGENT_LOW"]) & (mls < MLS_THRESHOLDS["MLS_CRITICAL"]),
        f"Severe (≥{MLS_THRESHOLDS['MLS_CRITICAL']}mm)": mls >= MLS_THRESHOLDS["MLS_CRITICAL"],
    }

    # Also check: high MLS without ICH (clinical edge case)
    high_mls_no_ich = (mls >= MLS_THRESHOLDS["MLS_CRITICAL"]) & (df["AnyICH"] == False)
    moderate_mls_no_ich = (mls >= MLS_THRESHOLDS["MLS_URGENT_LOW"]) & (mls < MLS_THRESHOLDS["MLS_CRITICAL"]) & (df["AnyICH"] == False)

    results = {}
    print("\n=== MLS Threshold Buckets ===")
    total = len(df)
    for label, condition in buckets.items():
        cnt = condition.sum()
        pct = cnt / total * 100
        results[label] = {"count": int(cnt), "pct": round(pct, 2)}
        print(f"  {label}: {cnt:>5} ({pct:.1f}%)")

    print(f"\n  ⚠️  High MLS (≥{MLS_THRESHOLDS['MLS_CRITICAL']}mm) WITHOUT ICH: {high_mls_no_ich.sum()} cases "
          f"({high_mls_no_ich.sum()/total*100:.1f}%)")
    print(f"  ⚠️  Moderate MLS ({MLS_THRESHOLDS['MLS_URGENT_LOW']}–{MLS_THRESHOLDS['MLS_CRITICAL']}mm) WITHOUT ICH: "
          f"{moderate_mls_no_ich.sum()} cases ({moderate_mls_no_ich.sum()/total*100:.1f}%)")

    results["high_mls_no_ich"] = int(high_mls_no_ich.sum())
    results["moderate_mls_no_ich"] = int(moderate_mls_no_ich.sum())

    # Chart
    labels = list(buckets.keys())
    values = [buckets[l].sum() for l in labels]
    colors = ["#bdc3c7", "#95a5a6", "#3498db", "#f39c12", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(labels, values, color=colors, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height() / 2,
                f"{val} ({val/total*100:.1f}%)", va="center", fontsize=10)
    ax.set_xlabel("Number of Cases")
    ax.set_title("MLS Severity Buckets", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "mls_threshold_buckets.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return results


def main():
    print("=" * 60)
    print("  Phase 0.3: Midline Shift (MLS) Analysis")
    print("=" * 60)

    df = load_data()

    results = {}
    results["mls_distribution"] = plot_mls_distribution(df)
    results["mls_by_triage"] = plot_mls_by_triage(df)
    results["mls_vs_ich"] = plot_mls_vs_ich(df)
    results["mls_thresholds"] = plot_mls_threshold_analysis(df)

    # Save
    json_path = OUTPUT_DIR / "eda_mls_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Saved: {json_path}")
    print("\n✅ Phase 0.3 complete!")


if __name__ == "__main__":
    main()
