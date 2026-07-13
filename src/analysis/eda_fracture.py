"""
eda_fracture.py — Phase 0.4: Skull Fracture Analysis

Analyzes skull fracture prevalence and relationships:
- Fracture prevalence overall
- Co-occurrence of fracture with ICH
- Fracture by triage class
- ICH volume in fracture+ vs fracture- cases
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / "Data" / "metadata" / "training_df.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # Compute total ICH volume
    spacing_x = df["dicom_series.PixelSpacing0"]
    spacing_y = df["dicom_series.PixelSpacing1"]
    thickness = df["dicom_series.SliceThickness"]
    area_cols = [
        "IntraventricularHemorrhage_Area", "IntraparenchymalHemorrhage_Area",
        "SubarachnoidHemorrhage_Area", "EpiduralHemorrhage_Area", "SubduralHemorrhage_Area",
    ]
    total_vol = sum(df[c].fillna(0) * spacing_x * spacing_y * thickness / 1000.0 for c in area_cols if c in df.columns)
    df["V_total"] = total_vol
    print(f"✅ Loaded {len(df)} rows")
    return df


def plot_fracture_prevalence(df: pd.DataFrame) -> dict:
    """Overall fracture statistics."""
    total = len(df)
    n_fracture = df["SkullFracture"].sum()
    n_no_fracture = total - n_fracture

    stats = {
        "total_cases": total,
        "n_fracture": int(n_fracture),
        "pct_fracture": round(n_fracture / total * 100, 2),
        "n_no_fracture": int(n_no_fracture),
        "pct_no_fracture": round(n_no_fracture / total * 100, 2),
    }

    print("\n=== Fracture Prevalence ===")
    print(f"  Fracture detected: {n_fracture} ({n_fracture/total*100:.1f}%)")
    print(f"  No fracture: {n_no_fracture} ({n_no_fracture/total*100:.1f}%)")

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(["No Fracture", "Fracture Detected"], [n_no_fracture, n_fracture],
                  color=["#2ecc71", "#e74c3c"], edgecolor="white", width=0.5)
    for bar, val in zip(bars, [n_no_fracture, n_fracture]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                f"{val}\n({val/total*100:.1f}%)", ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Number of Cases")
    ax.set_title("Skull Fracture Prevalence", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "fracture_prevalence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return stats


def plot_fracture_ich_cooccurrence(df: pd.DataFrame) -> dict:
    """Co-occurrence matrix: Fracture vs ICH."""
    fracture = df["SkullFracture"] == True
    ich = df["AnyICH"] == True

    matrix = {
        "Fracture + ICH": int((fracture & ich).sum()),
        "Fracture only": int((fracture & ~ich).sum()),
        "ICH only": int((~fracture & ich).sum()),
        "Neither": int((~fracture & ~ich).sum()),
    }

    total = len(df)
    print("\n=== Fracture vs ICH Co-occurrence ===")
    for k, v in matrix.items():
        print(f"  {k}: {v} ({v/total*100:.1f}%)")

    # Chi-square-like visualization
    contingency = pd.crosstab(df["SkullFracture"], df["AnyICH"])
    print("\nContingency Table:")
    print(contingency)
    print()

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = list(matrix.keys())
    values = list(matrix.values())
    colors_bar = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    bars = ax.bar(labels, values, color=colors_bar, edgecolor="white")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 15,
                f"{val} ({val/total*100:.1f}%)", ha="center", va="bottom")
    ax.set_ylabel("Number of Cases")
    ax.set_title("Fracture vs ICH Co-occurrence", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "fracture_ich_cooccurrence.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return matrix


def plot_fracture_by_triage(df: pd.DataFrame) -> dict:
    """Fracture distribution across triage classes."""
    print("\n=== Fracture by Triage Class ===")

    results = {}
    fracture_counts = {}
    for cls in sorted(df["triage_class"].unique()):
        subset = df[df["triage_class"] == cls]
        n_frac = subset["SkullFracture"].sum()
        n_total = len(subset)
        pct = n_frac / n_total * 100 if n_total > 0 else 0
        fracture_counts[int(cls)] = {"n_fracture": int(n_frac), "n_total": int(n_total), "pct": round(pct, 2)}
        results[int(cls)] = fracture_counts[int(cls)]
        print(f"  Class {cls}: {n_frac}/{n_total} fracture ({pct:.1f}%)")

    # Also compute: fracture prevalence within each triage class as percentage
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Absolute counts
    classes = sorted(df["triage_class"].unique())
    frac_yes = [fracture_counts[c]["n_fracture"] for c in classes]
    frac_no = [fracture_counts[c]["n_total"] - fracture_counts[c]["n_fracture"] for c in classes]
    x = np.arange(len(classes))
    width = 0.35
    axes[0].bar(x - width / 2, frac_no, width, label="No Fracture", color="#2ecc71", edgecolor="white")
    axes[0].bar(x + width / 2, frac_yes, width, label="Fracture", color="#e74c3c", edgecolor="white")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"Class {c}" for c in classes])
    axes[0].set_ylabel("Number of Cases")
    axes[0].set_title("Fracture by Triage Class (Absolute)", fontsize=13, fontweight="bold")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # Percentage within each class
    pct_frac = [fracture_counts[c]["pct"] for c in classes]
    axes[1].bar(classes, pct_frac, color=["#2ecc71", "#f39c12", "#e74c3c"], edgecolor="white", width=0.5)
    axes[1].set_xticks(classes)
    axes[1].set_xticklabels([f"Class {c}" for c in classes])
    axes[1].set_ylabel("Fracture Rate (%)")
    axes[1].set_title("Fracture Rate by Triage Class", fontsize=13, fontweight="bold")
    axes[1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "fracture_by_triage.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return results


def plot_volume_fracture_relationship(df: pd.DataFrame) -> dict:
    """Compare ICH volume in fracture+ vs fracture- cases."""
    frac = df["SkullFracture"] == True
    no_frac = df["SkullFracture"] == False

    stats = {
        "fracture_positive": {
            "mean_vol": float(round(df[frac]["V_total"].mean(), 2)),
            "median_vol": float(round(df[frac]["V_total"].median(), 2)),
            "max_vol": float(round(df[frac]["V_total"].max(), 2)),
            "n": int(frac.sum()),
        },
        "fracture_negative": {
            "mean_vol": float(round(df[no_frac]["V_total"].mean(), 2)),
            "median_vol": float(round(df[no_frac]["V_total"].median(), 2)),
            "max_vol": float(round(df[no_frac]["V_total"].max(), 2)),
            "n": int(no_frac.sum()),
        },
    }

    print("\n=== ICH Volume: Fracture+ vs Fracture- ===")
    for group, s in stats.items():
        print(f"  {group}: n={s['n']}, Mean={s['mean_vol']:.2f}, Median={s['median_vol']:.2f}, Max={s['max_vol']:.2f}")

    # Check: how many fracture+ cases have total_vol >= FRAC_VOL_CRIT (15mL)?
    vol_crit = 15.0
    frac_high_vol = (df["SkullFracture"] == True) & (df["V_total"] >= vol_crit)
    frac_low_vol = (df["SkullFracture"] == True) & (df["V_total"] < vol_crit)
    print(f"\n  Fracture + ICH >= {vol_crit}mL (→ Critical by combo rule): {frac_high_vol.sum()} cases")
    print(f"  Fracture + ICH < {vol_crit}mL (→ Urgent by rule): {frac_low_vol.sum()} cases")
    stats["combo_rule"] = {
        "frac_high_vol": int(frac_high_vol.sum()),
        "frac_low_vol": int(frac_low_vol.sum()),
        "threshold_ml": vol_crit,
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    data = [
        df[no_frac]["V_total"].values + 1e-6,
        df[frac]["V_total"].values + 1e-6,
    ]
    bp = ax.boxplot(data, tick_labels=["No Fracture", "Fracture"], patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#2ecc71", "#e74c3c"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_yscale("log")
    ax.set_ylabel("Total ICH Volume (mL) — log scale")
    ax.set_title("ICH Volume: Fracture+ vs Fracture-", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)
    plt.tight_layout()

    path = OUTPUT_DIR / "fracture_volume_comparison.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"📊 Saved: {path}")
    return stats


def main():
    print("=" * 60)
    print("  Phase 0.4: Skull Fracture Analysis")
    print("=" * 60)

    df = load_data()

    results = {}
    results["prevalence"] = plot_fracture_prevalence(df)
    results["cooccurrence"] = plot_fracture_ich_cooccurrence(df)
    results["by_triage"] = plot_fracture_by_triage(df)
    results["volume_relationship"] = plot_volume_fracture_relationship(df)

    json_path = OUTPUT_DIR / "eda_fracture_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Saved: {json_path}")
    print("\n✅ Phase 0.4 complete!")


if __name__ == "__main__":
    main()
