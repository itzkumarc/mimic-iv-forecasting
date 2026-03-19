"""
02_qc_plots.py
==============
Quality-control visualizations for the processed time-series dataset.
Target: MAP only. Creatinine shown as a covariate for context.

Outputs (saved to outputs/qc/):
  1. sampling_density_heatmap.png  -- observations per variable per hour
  2. missingness_profile.png       -- % missing per variable (before vs after imputation)
  3. map_distribution_by_split.png -- box plots of MAP by train/val/test
  4. patient_trajectories.png      -- 5 representative test patients (MAP + key covariates)
  5. sampling_density_bar.png      -- total observation counts per variable
"""

import json
import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "outputs", "qc")
os.makedirs(OUT_DIR, exist_ok=True)

PALETTE = {"train": "#4C72B0", "val": "#DD8452", "test": "#55A868"}
TARGET_COL     = "MAP"
COVARIATE_COLS = ["HR", "RR", "SpO2", "SysBP", "DiaBP", "Temperature",
                  "Creatinine", "WBC", "Glucose", "Lactate"]
NOTE_COVS      = ["Vasopressor", "IV_Fluid_mL"]   # intervention columns
ALL_VARS       = [TARGET_COL] + COVARIATE_COLS + NOTE_COVS


def load_data():
    df   = pd.read_csv(os.path.join(DATA_DIR, "long_format.csv"))
    miss = pd.read_csv(os.path.join(DATA_DIR, "missingness_report.csv"))
    
    stats_path = os.path.join(DATA_DIR, "filtering_stats.json")
    if os.path.exists(stats_path):
        with open(stats_path, "r") as f:
            stats = json.load(f)
    else:
        stats = None
        
    import sys
    import importlib.util
    spec = importlib.util.spec_from_file_location("build_dataset", os.path.join(BASE_DIR, "src", "01_build_dataset.py"))
    build_dataset = importlib.util.module_from_spec(spec)
    sys.modules["build_dataset"] = build_dataset
    spec.loader.exec_module(build_dataset)
    
    # Load raw data pre-imputation for density mapping
    cohort, _ = build_dataset.load_cohort()
    chart_df = build_dataset.load_chart_vitals(cohort)
    lab_df   = build_dataset.load_lab_values(cohort)
    interv_df = build_dataset.load_interventions(cohort)
    df_raw = build_dataset.build_hourly_grid(cohort, chart_df, lab_df, interv_df)
    
    stays_with_map = df_raw[df_raw["MAP"].notna()]["stay_id"].unique()
    df_raw = df_raw[df_raw["stay_id"].isin(stays_with_map)].copy()
        
    return df, miss, stats, df_raw

# --------------------------------------------------------------------------
# Plot 0 -- Dataset Attrition (Filtering)
# --------------------------------------------------------------------------
def plot_dataset_attrition(stats):
    if not stats:
        print("  Warning: No filtering stats found. Skipping attrition plot.")
        return

    labels = [
        "Raw ICU Stays",
        "Adults (>=18y)",
        "Stay >= 6h",
        "Retained (MAP present)"
    ]
    vals = [
        stats["raw_icustays"],
        stats["after_age_filter"],
        stats["after_los_filter"],
        stats["after_map_presence_filter"]
    ]
    
    # Calculate omitted counts
    omitted = []
    prev = None
    for v in vals:
        if prev is None:
            omitted.append(0)
        else:
            omitted.append(prev - v)
        prev = v

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color="#4C72B0", alpha=0.8, width=0.6)
    
    # Add trend line
    ax.plot(x, vals, marker='o', color='#D65F5F', linewidth=2, markersize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Number of ICU Stays", fontsize=11)
    ax.set_title("Dataset Attrition (Inclusion/Exclusion Flow)", fontsize=13, fontweight="bold")
    
    # Add value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (max(vals)*0.02),
                f"{val:,}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        
        if i > 0 and omitted[i] > 0:
            ax.text(i, vals[i] + (vals[i-1]-vals[i])/2, f"-{omitted[i]:,} omitted", 
                    ha="left", va="center", fontsize=9, color="#D65F5F", fontweight="bold",
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    
    path = os.path.join(OUT_DIR, "dataset_attrition.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")



# --------------------------------------------------------------------------
# Plot 1 -- Sampling density heatmap
# --------------------------------------------------------------------------
def plot_sampling_density_heatmap(df):
    vars_of_interest = [v for v in ALL_VARS if v in df.columns]
    hour_bins = sorted(df["hour_bin"].unique())
    hour_sample = [h for h in hour_bins if h % 2 == 0]

    data_mat = []
    for var in vars_of_interest:
        row = [df[df["hour_bin"] == h][var].notna().mean() for h in hour_sample]
        data_mat.append(row)

    mat = np.array(data_mat)
    fig, ax = plt.subplots(figsize=(16, 5))
    im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(hour_sample)))
    ax.set_xticklabels([str(h) for h in hour_sample], fontsize=7, rotation=90)
    ax.set_yticks(range(len(vars_of_interest)))
    ax.set_yticklabels(vars_of_interest, fontsize=9)
    ax.set_xlabel("Hour relative to ICU admission", fontsize=11)
    ax.set_title("Sampling Density -- Fraction of Patient-Hours with a Recorded Value",
                 fontsize=13, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Fraction present")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sampling_density_heatmap.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------
# Plot 2 -- Missingness profile
# --------------------------------------------------------------------------
def plot_missingness_profile(miss_df):
    # Highlight MAP vs others
    miss_df = miss_df[miss_df["variable"].isin(ALL_VARS)].copy()
    miss_df = miss_df.sort_values("missing_before", ascending=False)

    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(len(miss_df))
    w = 0.35
    bar_colors_before = ["#D65F5F" if v == TARGET_COL else "#D65F5F"
                         for v in miss_df["variable"]]
    bar_colors_after  = ["#2ca02c" if v == TARGET_COL else "#4C72B0"
                         for v in miss_df["variable"]]

    bars1 = ax.bar(x - w/2, miss_df["missing_before"] * 100, w,
                   label="Before imputation", color=bar_colors_before, alpha=0.85)
    bars2 = ax.bar(x + w/2, miss_df["missing_after"]  * 100, w,
                   label="After imputation",  color=bar_colors_after,  alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(miss_df["variable"], rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("% Missing", fontsize=11)
    ax.set_ylim(0, 107)
    ax.set_title("Missingness Profile -- Before vs After Imputation\n"
                 "(green = MAP target after imputation; "
                 "note Creatinine ~94% raw missing -- used as covariate only)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle="--", alpha=0.7)
    ax.set_axisbelow(True)
    for bar in bars1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.0f}%",
                ha="center", va="bottom", fontsize=6.5, color="#8B0000")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "missingness_profile.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------
# Plot 3 -- MAP distribution by split
# --------------------------------------------------------------------------
def plot_map_distribution(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: box plot per split
    ax = axes[0]
    data, labels, colors = [], [], []
    for split in ["train", "val", "test"]:
        vals = df[df["split"] == split]["MAP"].dropna()
        data.append(vals.values)
        labels.append(f"{split}\n(n={len(vals):,})")
        colors.append(PALETTE[split])
    bp = ax.boxplot(data, patch_artist=True, notch=True,
                    medianprops=dict(color="white", linewidth=2))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("MAP (mmHg)", fontsize=10)
    ax.set_title("MAP Distribution by Split", fontsize=12, fontweight="bold")
    ax.set_ylim(30, 160)
    ax.axhline(65, color="red", linestyle="--", linewidth=1, alpha=0.7,
               label="Shock threshold (65 mmHg)")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)

    # Right: KDE of MAP per split
    ax2 = axes[1]
    for split, color in PALETTE.items():
        vals = df[df["split"] == split]["MAP"].dropna()
        vals.plot.kde(ax=ax2, label=split, color=color, linewidth=2)
    ax2.axvline(65, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
    ax2.set_xlim(20, 160)
    ax2.set_xlabel("MAP (mmHg)", fontsize=10)
    ax2.set_title("MAP Density by Split", fontsize=12, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax2.set_axisbelow(True)

    plt.suptitle("MAP Target Variable -- Split Consistency Check",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "map_distribution_by_split.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------
# Plot 4 -- Patient trajectories (MAP + HR + Vasopressor)
# --------------------------------------------------------------------------
def plot_patient_trajectories(df):
    test_df  = df[df["split"] == "test"]
    stay_ids = test_df["stay_id"].unique()
    np.random.seed(42)
    chosen   = np.random.choice(stay_ids, size=min(5, len(stay_ids)), replace=False)

    fig = plt.figure(figsize=(16, 16))
    gs  = gridspec.GridSpec(5, 2, hspace=0.6, wspace=0.32)

    for row_i, sid in enumerate(chosen):
        pat = df[df["stay_id"] == sid].sort_values("hour_bin")

        # Left: MAP trajectory
        ax1 = fig.add_subplot(gs[row_i, 0])
        ax1.plot(pat["hour_bin"], pat["MAP"], color="#1f77b4", linewidth=1.8, label="MAP")
        ax1.axhline(65, color="red", linestyle="--", alpha=0.6, linewidth=1,
                    label="65 mmHg threshold")
        ax1.fill_between(pat["hour_bin"], 65, pat["MAP"],
                         where=pat["MAP"] < 65,
                         alpha=0.18, color="red", label="Below threshold")
        ax1.axvline(0, color="gray", linestyle=":", alpha=0.7)
        ax1.set_ylabel("MAP (mmHg)", fontsize=8)
        ax1.set_xlabel("Hour from ICU admission", fontsize=7)
        ax1.set_title(f"stay_id={sid} -- MAP", fontsize=9, fontweight="bold")
        ax1.legend(fontsize=7)
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Right: HR and Vasopressor
        ax2 = fig.add_subplot(gs[row_i, 1])
        ax2.plot(pat["hour_bin"], pat["HR"], color="#2ca02c",
                 linewidth=1.5, label="HR (bpm)")
        ax3 = ax2.twinx()
        if "Vasopressor" in pat.columns:
            ax3.fill_between(pat["hour_bin"], 0, pat["Vasopressor"],
                             color="#d62728", alpha=0.35, label="Vasopressor ON")
            ax3.set_ylabel("Vasopressor (binary)", fontsize=7, color="#d62728")
            ax3.set_ylim(-0.1, 1.5)
        ax2.axvline(0, color="gray", linestyle=":", alpha=0.7)
        ax2.set_ylabel("HR (bpm)", fontsize=8)
        ax2.set_xlabel("Hour from ICU admission", fontsize=7)
        ax2.set_title(f"stay_id={sid} -- HR & Vasopressor", fontsize=9, fontweight="bold")
        ax2.legend(loc="upper left", fontsize=7)
        ax2.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(
        "Representative Patient Trajectories -- Test Set\n"
        "Left: MAP with 65 mmHg shock threshold | "
        "Right: HR + vasopressor use\n"
        "(vertical dashed = ICU admission time)",
        fontsize=11, fontweight="bold"
    )
    path = os.path.join(OUT_DIR, "patient_trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# --------------------------------------------------------------------------
# Plot 5 -- Sampling density bar chart
# --------------------------------------------------------------------------
def plot_sampling_density_bar(df):
    stay_count  = df["stay_id"].nunique()
    hour_count  = df["hour_bin"].nunique()
    total_slots = stay_count * hour_count

    vars_to_plot = [v for v in ALL_VARS if v in df.columns]
    percentages  = {}
    for col in vars_to_plot:
        if col in ("Vasopressor", "IV_Fluid_mL"):
            percentages[col] = (df[col] > 0).sum() / total_slots * 100
        else:
            percentages[col] = df[col].notna().sum() / total_slots * 100

    vars_sorted = sorted(percentages, key=lambda c: percentages[c], reverse=True)
    vals = [percentages[c] for c in vars_sorted]
    colors = []
    for v in vars_sorted:
        if v == TARGET_COL:
            colors.append("#1f77b4")        # blue = target
        elif v in NOTE_COVS:
            colors.append("#d62728")        # red  = intervention
        elif v == "Creatinine":
            colors.append("#ff7f0e")        # orange = sparse covariate
        else:
            colors.append("#78c679")        # green = regular covariate

    fig, ax = plt.subplots(figsize=(14, 5))
    bars = ax.bar(vars_sorted, vals, color=colors, alpha=0.85, edgecolor="white")
    ax.set_ylabel("% of patient-hours with observation", fontsize=11)
    ax.set_title(
        "Sampling Density per Variable\n"
        "(blue=MAP target | orange=Creatinine covariate [sparse, not a target] | "
        "green=covariates | red=interventions)",
        fontsize=11, fontweight="bold"
    )
    ax.set_ylim(0, 110)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{val:.0f}%", ha="center", va="bottom", fontsize=8)
    plt.xticks(rotation=40, ha="right", fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "sampling_density_bar.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("  QC Plots -- MAP Target")
    print("=" * 60)
    df, miss_df, stats, df_raw = load_data()
    print(f"Loaded Imputed: {df.shape}  stays={df['stay_id'].nunique()}")
    print(f"Loaded Raw:     {df_raw.shape}")

    print("\n[0/5] Dataset attrition (filtering) ...")
    plot_dataset_attrition(stats)
    print("[1/5] Sampling density heatmap (Using Raw Data) ...")
    plot_sampling_density_heatmap(df_raw)
    print("[2/5] Missingness profile ...")
    plot_missingness_profile(miss_df)
    print("[3/5] MAP distribution by split ...")
    plot_map_distribution(df)
    print("[4/5] Patient trajectories ...")
    plot_patient_trajectories(df)
    print("[5/5] Sampling density bar chart (Using Raw Data) ...")
    plot_sampling_density_bar(df_raw)
    print("\nAll QC plots saved to outputs/qc/")


if __name__ == "__main__":
    main()
