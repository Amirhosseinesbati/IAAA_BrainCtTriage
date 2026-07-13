"""
eda_triage_simulation.py — Phase 0.6: Official Triage Function Simulation

Implements the exact triage_from_intermediates() function from the competition PDF
and runs it on the training data to understand:
- How the official triage rules behave on real data
- Discrepancies between CSV triage_class and official computation
- Borderline cases near thresholds
- Sensitivity of each primitive (volume, MLS, fracture)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Any, Dict, Mapping
import json

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CSV_PATH = BASE_DIR / "Data" / "metadata" / "training_df.csv"
OUTPUT_DIR = BASE_DIR / "reports" / "eda"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# Official Competition Triage Function
# (extracted verbatim from competition PDF)
# ==========================================

TRIAGE_REQUIRED_KEYS = {
    "V_EDH", "V_SDH", "V_IPH", "V_SAH", "V_IVH",
    "fracture_prob", "MLS_mm",
}


def validate_intermediates(intermediates: Mapping[str, Any]) -> Dict[str, float]:
    """Validate and normalize intermediate values for a single series."""  # noqa: D401
    missing = TRIAGE_REQUIRED_KEYS - intermediates.keys()
    extra = intermediates.keys() - TRIAGE_REQUIRED_KEYS
    if missing:
        raise ValueError(f"Missing keys in intermediates: {sorted(missing)}.")
    if extra:
        raise ValueError(f"Unexpected keys in intermediates: {sorted(extra)}.")
    cleaned: Dict[str, float] = {}
    for key in TRIAGE_REQUIRED_KEYS:
        try:
            cleaned[key] = float(intermediates[key])
        except Exception as exc:
            raise TypeError(
                f"Value for key {key!r} must be convertible to float, "
                f"got type {type(intermediates[key]).__name__}."
            ) from exc
    return cleaned


def triage_from_intermediates(intermediates: Mapping[str, Any]) -> int:
    """
    Compute triage class from intermediate imaging primitives.

    Returns:
        0 = non-urgent
        1 = urgent
        2 = critical
    """
    vals = validate_intermediates(intermediates)

    # Extract and clamp primitives
    V_EDH = max(0.0, vals["V_EDH"])
    V_SDH = max(0.0, vals["V_SDH"])
    V_IPH = max(0.0, vals["V_IPH"])
    V_SAH = max(0.0, vals["V_SAH"])
    V_IVH = max(0.0, vals["V_IVH"])
    MLS_mm = max(0.0, vals["MLS_mm"])
    fracture_prob = float(vals["fracture_prob"])

    total_vol = V_EDH + V_SDH + V_IPH + V_SAH + V_IVH

    # Hard-coded thresholds
    EPS_VOLUME = 0.1
    EPS_MLS = 1.0
    MLS_CRITICAL = 5.0
    MLS_URGENT_LOW = 3.0
    EDH_CRIT = 30.0
    SDH_CRIT = 70.0
    IPH_CRIT = 70.0
    TOTAL_VOL_CRIT = 60.0
    COMBO_MLS = 3.0
    COMBO_VOL = 40.0
    FRAC_VOL_CRIT = 15.0
    FRACTURE_PRESENCE_THRESHOLD = 0.5

    # Derived flags
    has_ich = total_vol >= EPS_VOLUME
    mls_present = MLS_mm >= EPS_MLS
    fracture_present = fracture_prob >= FRACTURE_PRESENCE_THRESHOLD

    # Critical triage (2)
    if MLS_mm >= MLS_CRITICAL and (has_ich or fracture_present):
        return 2
    if V_EDH >= EDH_CRIT:
        return 2
    if V_SDH >= SDH_CRIT:
        return 2
    if V_IPH >= IPH_CRIT:
        return 2
    if total_vol >= TOTAL_VOL_CRIT:
        return 2
    if has_ich and MLS_mm >= COMBO_MLS and total_vol >= COMBO_VOL:
        return 2
    if fracture_present and total_vol >= FRAC_VOL_CRIT:
        return 2

    # Urgent triage (1)
    if MLS_mm >= MLS_CRITICAL and not (has_ich or fracture_present):
        return 1
    if has_ich:
        return 1
    if MLS_URGENT_LOW <= MLS_mm < MLS_CRITICAL:
        return 1
    if fracture_present and total_vol < FRAC_VOL_CRIT:
        return 1
    if total_vol >= EPS_VOLUME and mls_present:
        return 1

    # Non-urgent
    return 0


# ==========================================
# DATA LOADING & PREPARATION
# ==========================================

def load_and_prepare_data() -> pd.DataFrame:
    """Load CSV and compute per-series intermediate quantities."""
    df = pd.read_csv(CSV_PATH)
    print(f"✅ Loaded {len(df)} rows from CSV")

    # Compute volumes from pixel areas (per-slice)
    spacing_x = df["dicom_series.PixelSpacing0"]
    spacing_y = df["dicom_series.PixelSpacing1"]
    thickness = df["dicom_series.SliceThickness"]
    factor = spacing_x * spacing_y * thickness / 1000.0

    df["V_IVH"] = df["IntraventricularHemorrhage_Area"].fillna(0) * factor
    df["V_IPH"] = df["IntraparenchymalHemorrhage_Area"].fillna(0) * factor
    df["V_SAH"] = df["SubarachnoidHemorrhage_Area"].fillna(0) * factor
    df["V_EDH"] = df["EpiduralHemorrhage_Area"].fillna(0) * factor
    df["V_SDH"] = df["SubduralHemorrhage_Area"].fillna(0) * factor

    # Aggregate to series-level (sum across slices per series)
    series_cols = ["dicom_series.id", "dicom_series.PatientID",
                   "dicom_series.StudyInstanceUID", "dicom_series.SeriesInstanceUID"]
    vol_cols = ["V_IVH", "V_IPH", "V_SAH", "V_EDH", "V_SDH"]

    agg_dict = {c: "sum" for c in vol_cols}
    agg_dict["MidlineShiftMM"] = "max"  # MLS at foramen of Monro level
    agg_dict["SkullFracture"] = "max"   # Any fracture in series
    agg_dict["triage_class"] = "first"  # Should be same for all slices in series
    agg_dict["dicom_series.NumDicomFiles"] = "first"

    series_df = df.groupby("dicom_series.id").agg(agg_dict).reset_index()

    print(f"✅ Aggregated to {len(series_df)} unique series "
          f"(from {len(df)} slice-level rows)")

    # Create intermediate dict format
    series_df["fracture_prob"] = series_df["SkullFracture"].astype(float)
    series_df["MLS_mm"] = series_df["MidlineShiftMM"]

    return series_df, df  # Return both aggregated and slice-level


# ==========================================
# ANALYSIS FUNCTIONS
# ==========================================

def compute_official_triage(series_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the official triage function to each series."""
    results = []
    for _, row in series_df.iterrows():
        intermediates = {
            "V_EDH": row["V_EDH"],
            "V_SDH": row["V_SDH"],
            "V_IPH": row["V_IPH"],
            "V_SAH": row["V_SAH"],
            "V_IVH": row["V_IVH"],
            "fracture_prob": row["fracture_prob"],
            "MLS_mm": row["MLS_mm"],
        }
        official_class = triage_from_intermediates(intermediates)
        results.append(official_class)

    series_df["official_triage"] = results
    return series_df


def compare_triage_labels(series_df: pd.DataFrame) -> dict:
    """Compare CSV triage_class with official computation."""
    csv_label = series_df["triage_class"]
    official_label = series_df["official_triage"]

    agreement = (csv_label == official_label).sum()
    total = len(series_df)
    agreement_rate = agreement / total * 100

    print("\n=== Triage Label Comparison (CSV vs Official) ===")
    print(f"  Agreement rate: {agreement}/{total} ({agreement_rate:.1f}%)")

    # Confusion matrix
    cm = pd.crosstab(csv_label, official_label,
                     rownames=["CSV triage_class"],
                     colnames=["Official triage"])
    print("\nConfusion Matrix:")
    print(cm)

    # Disagreement analysis
    disagree = series_df[csv_label != official_label]
    print(f"\n  Disagreements: {len(disagree)} cases")
    if len(disagree) > 0:
        print(f"\n  Sample of disagreements (first 10):")
        cols_to_show = ["dicom_series.id", "V_IVH", "V_IPH", "V_SAH", "V_EDH", "V_SDH",
                        "MLS_mm", "SkullFracture", "triage_class", "official_triage"]
        print(disagree[cols_to_show].head(10).to_string())

    # Direction of disagreement
    upgrades = ((official_label > csv_label)).sum()
    downgrades = ((official_label < csv_label)).sum()
    print(f"\n  Official is HIGHER (upgrade): {upgrades} cases")
    print(f"  Official is LOWER (downgrade): {downgrades} cases")

    results = {
        "total_series": int(total),
        "agreement_count": int(agreement),
        "agreement_rate": round(agreement_rate, 2),
        "disagreement_count": int(len(disagree)),
        "upgrades": int(upgrades),
        "downgrades": int(downgrades),
        "confusion_matrix": cm.to_dict(),
    }
    return results


def analyze_official_distribution(series_df: pd.DataFrame) -> dict:
    """Distribution of official triage classes."""
    counts = series_df["official_triage"].value_counts().sort_index()
    total = len(series_df)
    label_map = {0: "Non-urgent (0)", 1: "Urgent (1)", 2: "Critical (2)"}

    print("\n=== Official Triage Distribution ===")
    dist = {}
    for cls in [0, 1, 2]:
        n = counts.get(cls, 0)
        pct = n / total * 100
        dist[cls] = {"count": int(n), "pct": round(pct, 2)}
        print(f"  {label_map[cls]}: {n} ({pct:.1f}%)")

    # Comparison with CSV distribution
    csv_counts = series_df["triage_class"].value_counts().sort_index()
    print("\n  vs CSV triage_class distribution:")
    for cls in [0, 1, 2]:
        n_csv = csv_counts.get(cls, 0)
        n_off = counts.get(cls, 0)
        diff = n_off - n_csv
        print(f"    {label_map[cls]}: CSV={n_csv}, Official={n_off}, Diff={diff:+d}")

    return dist


def analyze_threshold_sensitivity(series_df: pd.DataFrame) -> dict:
    """
    Analyze which rules trigger for each case.
    Shows the sensitivity of each primitive/threshold.
    """
    print("\n=== Threshold Sensitivity Analysis ===")
    total = len(series_df)

    # Track which rules would trigger Critical (2)
    rules = {
        "MLS>=5 + (ICH or Frac)": (
            (series_df["MLS_mm"] >= 5.0) &
            ((series_df[["V_IVH","V_IPH","V_SAH","V_EDH","V_SDH"]].sum(axis=1) >= 0.1) |
             (series_df["SkullFracture"] == True))
        ),
        "EDH>=30": series_df["V_EDH"] >= 30,
        "SDH>=70": series_df["V_SDH"] >= 70,
        "IPH>=70": series_df["V_IPH"] >= 70,
        "Total>=60": series_df[["V_IVH","V_IPH","V_SAH","V_EDH","V_SDH"]].sum(axis=1) >= 60,
        "ICH+MLS>=3+Vol>=40": (
            (series_df[["V_IVH","V_IPH","V_SAH","V_EDH","V_SDH"]].sum(axis=1) >= 0.1) &
            (series_df["MLS_mm"] >= 3.0) &
            (series_df[["V_IVH","V_IPH","V_SAH","V_EDH","V_SDH"]].sum(axis=1) >= 40)
        ),
        "Frac+Vol>=15": (
            (series_df["SkullFracture"] == True) &
            (series_df[["V_IVH","V_IPH","V_SAH","V_EDH","V_SDH"]].sum(axis=1) >= 15)
        ),
    }

    rule_stats = {}
    for rule_name, condition in rules.items():
        n_true = condition.sum()
        pct = n_true / total * 100
        rule_stats[rule_name] = {"count": int(n_true), "pct": round(pct, 2)}
        print(f"  Critical Rule - {rule_name}: {n_true} cases ({pct:.2f}%)")

    # Track which rules trigger Urgent (1)
    total_vol = series_df[["V_IVH","V_IPH","V_SAH","V_EDH","V_SDH"]].sum(axis=1)
    urgent_rules = {
        "MLS>=5 + (no ICH & no Frac)": (
            (series_df["MLS_mm"] >= 5.0) &
            ~((total_vol >= 0.1) | (series_df["SkullFracture"] == True))
        ),
        "Any ICH": total_vol >= 0.1,
        "MLS 3-5mm": (series_df["MLS_mm"] >= 3.0) & (series_df["MLS_mm"] < 5.0),
        "Frac+Vol<15": (
            (series_df["SkullFracture"] == True) & (total_vol < 15)
        ),
    }

    for rule_name, condition in urgent_rules.items():
        n_true = condition.sum()
        pct = n_true / total * 100
        rule_stats[rule_name] = {"count": int(n_true), "pct": round(pct, 2)}
        print(f"  Urgent Rule   - {rule_name}: {n_true} cases ({pct:.2f}%)")

    return rule_stats


def plot_decision_boundaries(series_df: pd.DataFrame) -> None:
    """Visualize decision boundaries."""
    total_vol = series_df[["V_IVH","V_IPH","V_SAH","V_EDH","V_SDH"]].sum(axis=1)
    mls = series_df["MLS_mm"]
    triage = series_df["official_triage"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Scatter: MLS vs Total Volume colored by triage
    colors = {0: "#2ecc71", 1: "#f39c12", 2: "#e74c3c"}
    for cls in [0, 1, 2]:
        mask = triage == cls
        axes[0].scatter(total_vol[mask], mls[mask],
                       c=colors[cls], label=f"Class {cls}",
                       alpha=0.4, s=15, edgecolors="none")

    # Decision boundary lines
    axes[0].axhline(5.0, color="red", linestyle="--", alpha=0.5, label="MLS=5 (Critical)")
    axes[0].axhline(3.0, color="orange", linestyle="--", alpha=0.5, label="MLS=3 (Urgent)")
    axes[0].axvline(60, color="red", linestyle=":", alpha=0.5, label="Vol=60 (Critical)")
    axes[0].axvline(40, color="orange", linestyle=":", alpha=0.5, label="Vol=40 (Combo)")
    axes[0].axvline(0.1, color="green", linestyle=":", alpha=0.3, label="Vol=0.1 (EPS)")

    axes[0].set_xlabel("Total ICH Volume (mL)")
    axes[0].set_ylabel("MLS (mm)")
    axes[0].set_title("Decision Space: MLS vs Total ICH Volume", fontsize=13, fontweight="bold")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].grid(alpha=0.3)

    # Bar chart: distribution of which rules triggered
    # For critical cases, what drove them to class 2?
    crit = series_df[triage == 2]
    if len(crit) > 0:
        reasons = []
        cv = crit[["V_IVH","V_IPH","V_SAH","V_EDH","V_SDH"]].sum(axis=1)
        reasons.append(("MLS>=5+(ICH/Frac)", ((crit["MLS_mm"]>=5) & ((cv>=0.1)|(crit["SkullFracture"]==True))).sum()))
        reasons.append(("EDH>=30", (crit["V_EDH"]>=30).sum()))
        reasons.append(("SDH>=70", (crit["V_SDH"]>=70).sum()))
        reasons.append(("IPH>=70", (crit["V_IPH"]>=70).sum()))
        reasons.append(("Total>=60", (cv>=60).sum()))
        reasons.append(("ICH+MLS+Vol>=40", ((cv>=0.1)&(crit["MLS_mm"]>=3)&(cv>=40)).sum()))
        reasons.append(("Frac+Vol>=15", ((crit["SkullFracture"]==True)&(cv>=15)).sum()))

        # Remove zero entries
        reasons = [(r, n) for r, n in reasons if n > 0]
        if reasons:
            labels, values = zip(*reasons)
            bars = axes[1].barh(labels, values, color="#e74c3c", edgecolor="white", alpha=0.7)
            for bar, val in zip(bars, values):
                axes[1].text(bar.get_width() + 1,
                            bar.get_y() + bar.get_height() / 2,
                            str(val), va="center", fontsize=9)
            axes[1].set_xlabel("Number of Critical Cases")
            axes[1].set_title("What Drives Critical (Class 2) Decisions?", fontsize=13, fontweight="bold")
            axes[1].grid(axis="x", alpha=0.3)

    plt.tight_layout()
    path = OUTPUT_DIR / "triage_decision_boundaries.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Saved: {path}")


def analyze_borderline_cases(series_df: pd.DataFrame) -> dict:
    """Identify borderline cases near thresholds."""
    total_vol = series_df[["V_IVH","V_IPH","V_SAH","V_EDH","V_SDH"]].sum(axis=1)
    mls = series_df["MLS_mm"]

    borderlines = {
        "MLS near critical (4-6mm)": ((mls >= 4.0) & (mls <= 6.0)).sum(),
        "Volume near critical (50-70mL)": ((total_vol >= 50) & (total_vol <= 70)).sum(),
        "MLS borderline (2.5-3.5mm)": ((mls >= 2.5) & (mls <= 3.5)).sum(),
        "Fracture + volume near 15mL (10-20mL)": (
            (series_df["SkullFracture"] == True) &
            (total_vol >= 10) & (total_vol <= 20)
        ).sum(),
    }

    print("\n=== Borderline Cases ===")
    results = {}
    for label, count in borderlines.items():
        results[label] = int(count)
        print(f"  {label}: {count}")

    return results


def compute_primary_driver(series_df: pd.DataFrame) -> dict:
    """For each official class, what's the primary driving factor?"""
    total_vol = series_df[["V_IVH","V_IPH","V_SAH","V_EDH","V_SDH"]].sum(axis=1)

    drivers = {"class_0": {}, "class_1": {}, "class_2": {}}

    for cls in [0, 1, 2]:
        subset = series_df[series_df["official_triage"] == cls]
        n = len(subset)
        if n == 0:
            continue

        key = f"class_{cls}"
        drivers[key]["count"] = n
        drivers[key]["avg_total_vol"] = float(round(total_vol[subset.index].mean(), 3))
        drivers[key]["avg_mls"] = float(round(subset["MLS_mm"].mean(), 3))
        drivers[key]["pct_with_fracture"] = round((subset["SkullFracture"] == True).mean() * 100, 1)
        drivers[key]["pct_with_any_ich"] = round((total_vol[subset.index] >= 0.1).mean() * 100, 1)

    print("\n=== Primary Drivers by Class ===")
    for cls_name, stats in drivers.items():
        if stats:
            print(f"  {cls_name}: n={stats['count']}, "
                  f"avg_vol={stats['avg_total_vol']:.3f}, "
                  f"avg_mls={stats['avg_mls']:.2f}mm, "
                  f"fracture={stats['pct_with_fracture']}%, "
                  f"any_ich={stats['pct_with_any_ich']}%")

    return drivers


# ==========================================
# MAIN
# ==========================================

def main():
    print("=" * 60)
    print("  Phase 0.6: Official Triage Function Simulation")
    print("=" * 60)

    series_df, slice_df = load_and_prepare_data()
    series_df = compute_official_triage(series_df)

    results = {}
    results["official_distribution"] = analyze_official_distribution(series_df)
    results["comparison"] = compare_triage_labels(series_df)
    results["threshold_sensitivity"] = analyze_threshold_sensitivity(series_df)
    plot_decision_boundaries(series_df)
    results["borderline_cases"] = analyze_borderline_cases(series_df)
    results["primary_drivers"] = compute_primary_driver(series_df)

    # Save
    json_path = OUTPUT_DIR / "eda_triage_simulation_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n📄 Saved: {json_path}")
    print("\n✅ Phase 0.6 complete!")


if __name__ == "__main__":
    main()
