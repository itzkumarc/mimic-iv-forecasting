"""
01_build_dataset.py
===================
Builds an analysis-ready long-format hourly time-series dataset from MIMIC-IV demo.

Target (continuous, repeatedly measured):
  - MAP (Mean Arterial Pressure)  -- frequent vital sign (~hourly in ICU)

Note: Creatinine (94% missing) is retained as an exogenous COVARIATE, not a target.

Exogenous covariates:
  - HR, RR, SpO2, SysBP, DiaBP, Temperature (chart vitals)
  - WBC, Glucose, Lactate          (lab values)
  - Vasopressor (binary flag), IV_Fluid_mL (cumulative per hour)

Time alignment: anchored to ICU admission intime.
Window: -12h to +72h relative to intime.
Hourly resolution: values aggregated within each hour (mean for vitals,
last observation carried forward for labs in time_bin window).

Splits: by subject_id (patient-level), 70 / 15 / 15 stratified by mortality.
"""

import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# -- Paths --------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HOSP_DIR = os.path.join(BASE_DIR, "data", "mimic-iv-clinical-database-demo-2.2", "hosp")
ICU_DIR  = os.path.join(BASE_DIR, "data", "mimic-iv-clinical-database-demo-2.2", "icu")
OUT_DIR  = os.path.join(BASE_DIR, "data", "processed")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
np.random.seed(SEED)

# -- Item ID maps -------------------------------------------------------------
CHART_ITEMS = {
    "MAP":         [220052, 220181],
    "HR":          [220045],
    "RR":          [220210],
    "SpO2":        [220277],
    "SysBP":       [220179, 220050],
    "DiaBP":       [220180, 220051],
    "Temp_F":      [223761],
    "Temp_C":      [223762],
}
LAB_ITEMS = {
    "Creatinine":  [50912],
    "Lactate":     [50813],
    "WBC":         [51301],
    "Glucose":     [50931, 50809],
}
VASOPRESS_ITEMS  = [221906, 221289, 221662, 222315]   # NE, Epi, Dopa, VP
IVFLUID_ITEMS    = [225158, 225159, 220949, 225823]   # NaCl, LR, D5W, misc

ALL_CHART_IDS = [iid for v in CHART_ITEMS.values() for iid in v]
ALL_LAB_IDS   = [iid for v in LAB_ITEMS.values()  for iid in v]

FORECAST_TARGETS = ["MAP"]   # sole prediction target
COVARIATE_COLS   = ["HR", "RR", "SpO2", "SysBP", "DiaBP", "Temperature",
                    "Creatinine", "WBC", "Glucose", "Lactate",
                    "Vasopressor", "IV_Fluid_mL"]
ALL_VALUE_COLS   = FORECAST_TARGETS + COVARIATE_COLS

WINDOW_START_H = -12
WINDOW_END_H   = 72


def itemid_to_name(itemid):
    for name, ids in {**CHART_ITEMS, **LAB_ITEMS}.items():
        if itemid in ids:
            return name
    return None


def load_cohort():
    print("[1/7] Loading cohort tables ...")
    patients   = pd.read_csv(os.path.join(HOSP_DIR, "patients.csv"),
                             usecols=["subject_id", "anchor_age", "gender"])
    admissions = pd.read_csv(os.path.join(HOSP_DIR, "admissions.csv"),
                             usecols=["subject_id", "hadm_id", "hospital_expire_flag"])
    icustays   = pd.read_csv(os.path.join(ICU_DIR,  "icustays.csv"),
                             parse_dates=["intime", "outtime"])

    # Attrition tracking
    stats = {"raw_icustays": len(icustays)}

    cohort = (icustays
              .merge(admissions, on=["subject_id", "hadm_id"], how="left")
              .merge(patients,   on="subject_id",               how="left"))
    
    cohort = cohort[cohort["anchor_age"] >= 18].copy()
    stats["after_age_filter"] = len(cohort)

    cohort["los_h"] = (cohort["outtime"] - cohort["intime"]).dt.total_seconds() / 3600.0
    cohort = cohort[cohort["los_h"] >= 6].copy()   # at least 6h stay
    stats["after_los_filter"] = len(cohort)

    print(f"    Cohort: {len(cohort)} ICU stays  |  {cohort['subject_id'].nunique()} patients")
    return cohort, stats


def load_chart_vitals(cohort):
    print("[2/7] Loading chartevents (vitals) ...")
    stay_ids = cohort["stay_id"].tolist()
    ce = pd.read_csv(os.path.join(ICU_DIR, "chartevents.csv"),
                     usecols=["stay_id", "charttime", "itemid", "valuenum"],
                     parse_dates=["charttime"])
    ce = ce[(ce["itemid"].isin(ALL_CHART_IDS)) & (ce["stay_id"].isin(stay_ids))].copy()
    ce["variable"] = ce["itemid"].map(itemid_to_name)
    # convert F -> C
    mask_f = ce["variable"] == "Temp_F"
    ce.loc[mask_f, "valuenum"] = (ce.loc[mask_f, "valuenum"] - 32) * 5.0 / 9.0
    ce.loc[mask_f, "variable"] = "Temperature"
    ce.loc[ce["variable"] == "Temp_C", "variable"] = "Temperature"
    # sanity filter
    bounds = {"MAP": (20, 200), "HR": (20, 300), "RR": (4, 60),
              "SpO2": (50, 100), "SysBP": (40, 280), "DiaBP": (20, 200),
              "Temperature": (25, 45)}
    parts = []
    for var, (lo, hi) in bounds.items():
        sub = ce[ce["variable"] == var]
        sub = sub[(sub["valuenum"] >= lo) & (sub["valuenum"] <= hi)]
        parts.append(sub)
    ce = pd.concat(parts, ignore_index=True)
    print(f"    Chart rows after filter: {len(ce):,}")
    return ce


def load_lab_values(cohort):
    print("[3/7] Loading labevents ...")
    hadm_ids = cohort["hadm_id"].tolist()
    le = pd.read_csv(os.path.join(HOSP_DIR, "labevents.csv"),
                     usecols=["hadm_id", "charttime", "itemid", "valuenum"],
                     parse_dates=["charttime"])
    le = le[(le["itemid"].isin(ALL_LAB_IDS)) & (le["hadm_id"].isin(hadm_ids))].copy()
    le["variable"] = le["itemid"].map(itemid_to_name)
    bounds = {"Creatinine": (0.1, 30), "Lactate": (0.1, 30),
              "WBC": (0.1, 500), "Glucose": (20, 2000)}
    parts = []
    for var, (lo, hi) in bounds.items():
        sub = le[le["variable"] == var]
        sub = sub[(sub["valuenum"] >= lo) & (sub["valuenum"] <= hi)]
        parts.append(sub)
    le = pd.concat(parts, ignore_index=True)
    # attach stay_id via hadm_id
    hadm_stay = cohort[["hadm_id", "stay_id"]].drop_duplicates()
    le = le.merge(hadm_stay, on="hadm_id", how="inner")
    print(f"    Lab rows after filter: {len(le):,}")
    return le


def load_interventions(cohort):
    print("[4/7] Loading inputevents (vasopressors / IV fluids) ...")
    stay_ids = cohort["stay_id"].tolist()
    ie = pd.read_csv(os.path.join(ICU_DIR, "inputevents.csv"),
                     usecols=["stay_id", "starttime", "itemid", "amount", "amountuom"],
                     parse_dates=["starttime"])
    ie = ie[ie["stay_id"].isin(stay_ids)].copy()
    ie_vaso  = ie[ie["itemid"].isin(VASOPRESS_ITEMS)][["stay_id", "starttime", "amount"]].copy()
    ie_vaso["variable"]  = "Vasopressor_amt"
    ie_fluid = ie[ie["itemid"].isin(IVFLUID_ITEMS)][["stay_id", "starttime", "amount"]].copy()
    ie_fluid["variable"] = "IV_Fluid_mL"
    ie_all = pd.concat([ie_vaso, ie_fluid], ignore_index=True)
    ie_all.rename(columns={"starttime": "charttime", "amount": "valuenum"}, inplace=True)
    print(f"    Intervention rows: {len(ie_all):,}")
    return ie_all


def build_hourly_grid(cohort, chart_df, lab_df, interv_df):
    print("[5/7] Building hourly grid and aligning time series ...")
    intime_map = cohort.set_index("stay_id")["intime"].to_dict()

    def attach_hours(df):
        df = df.copy()
        df["hours"] = df.apply(
            lambda r: (r["charttime"] - intime_map[r["stay_id"]]).total_seconds() / 3600.0
            if r["stay_id"] in intime_map else np.nan, axis=1)
        return df[df["hours"].notna()]

    chart_h  = attach_hours(chart_df)
    lab_h    = attach_hours(lab_df)
    interv_h = attach_hours(interv_df)

    # Filter window
    for df in [chart_h, lab_h, interv_h]:
        df.drop(df[(df["hours"] < WINDOW_START_H) |
                   (df["hours"] > WINDOW_END_H)].index, inplace=True)

    # Aggregate to hourly bin (floor)
    chart_h["hour_bin"]  = chart_h["hours"].apply(np.floor).astype(int)
    lab_h["hour_bin"]    = lab_h["hours"].apply(np.floor).astype(int)
    interv_h["hour_bin"] = interv_h["hours"].apply(np.floor).astype(int)

    # Vitals: mean per bin; labs: last per bin; interventions: sum per bin
    agg_chart  = (chart_h.groupby(["stay_id", "hour_bin", "variable"])["valuenum"]
                  .mean().reset_index())
    agg_lab    = (lab_h.sort_values("charttime")
                  .groupby(["stay_id", "hour_bin", "variable"])["valuenum"]
                  .last().reset_index())
    # vasopressor: binary (any dose > 0), IV fluid: sum
    interv_vaso  = (interv_h[interv_h["variable"] == "Vasopressor_amt"]
                    .groupby(["stay_id", "hour_bin"])["valuenum"]
                    .sum().reset_index())
    interv_vaso["variable"]  = "Vasopressor"
    interv_vaso["valuenum"]  = (interv_vaso["valuenum"] > 0).astype(float)
    interv_fluid = (interv_h[interv_h["variable"] == "IV_Fluid_mL"]
                    .groupby(["stay_id", "hour_bin"])["valuenum"]
                    .sum().reset_index())
    interv_fluid["variable"] = "IV_Fluid_mL"

    long = pd.concat([agg_chart, agg_lab, interv_vaso, interv_fluid], ignore_index=True)
    pivot = long.pivot_table(index=["stay_id", "hour_bin"],
                             columns="variable", values="valuenum").reset_index()
    pivot.columns.name = None

    # Full hourly grid
    stay_ids  = cohort["stay_id"].unique()
    hour_bins = np.arange(WINDOW_START_H, WINDOW_END_H + 1, 1)
    grid = pd.MultiIndex.from_product([stay_ids, hour_bins],
                                      names=["stay_id", "hour_bin"]).to_frame(index=False)
    df = grid.merge(pivot, on=["stay_id", "hour_bin"], how="left")

    # Ensure all expected columns exist
    for col in ALL_VALUE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    df["Vasopressor"] = df["Vasopressor"].fillna(0.0)
    df["IV_Fluid_mL"] = df["IV_Fluid_mL"].fillna(0.0)

    # Merge static info
    static_cols = ["stay_id", "subject_id", "hadm_id",
                   "anchor_age", "gender", "hospital_expire_flag", "los_h"]
    static = cohort[static_cols].copy()
    static["gender_male"] = (static["gender"] == "M").astype(int)
    df = df.merge(static, on="stay_id", how="left")

    print(f"    Grid shape: {df.shape}  (stays={df['stay_id'].nunique()}, hours={len(hour_bins)})")
    return df


def impute(df):
    print("[6/7] Imputing missing values (tiered) ...")
    df = df.sort_values(["stay_id", "hour_bin"]).copy()

    # Record missingness BEFORE imputation (for QC)
    miss_before = df[ALL_VALUE_COLS].isnull().mean().rename("missing_before")

    # Tier 1: forward-fill within patient
    vital_cols = ["MAP", "HR", "RR", "SpO2", "SysBP", "DiaBP", "Temperature"]
    # Creatinine kept as covariate -- use liberal ffill given sparsity
    lab_cols   = ["Creatinine", "Lactate", "WBC", "Glucose"]
    for col in vital_cols:
        df[col] = df.groupby("stay_id")[col].transform(
            lambda s: s.ffill(limit=4).bfill(limit=2))
    for col in lab_cols:
        df[col] = df.groupby("stay_id")[col].transform(
            lambda s: s.ffill(limit=24).bfill(limit=12))

    # Tier 2: linear interpolation within patient for remaining short gaps
    for col in FORECAST_TARGETS + COVARIATE_COLS:
        if col in ("Vasopressor", "IV_Fluid_mL"):
            continue
        df[col] = df.groupby("stay_id")[col].transform(
            lambda s: s.interpolate(method="linear", limit=6, limit_direction="both"))

    # Tier 3: global median fill for persistent NaN (mostly sparse labs)
    for col in ALL_VALUE_COLS:
        med = df[col].median()
        df[col] = df[col].fillna(med)

    miss_after = df[ALL_VALUE_COLS].isnull().mean().rename("missing_after")
    miss_report = pd.concat([miss_before, miss_after], axis=1).reset_index()
    miss_report.columns = ["variable", "missing_before", "missing_after"]

    # Also compute per-variable per-hour missingness (pre-imputation) for QC plots
    # We'll recalculate from the raw merge later; here we save the summary
    miss_report.to_csv(os.path.join(OUT_DIR, "missingness_report.csv"), index=False)
    print("    Missingness report saved.")
    return df


def feature_engineering(df):
    print("\n[6.5/7] Feature Engineering (deltas and rolling windows) ...")
    df = df.sort_values(["stay_id", "hour_bin"]).copy()

    # Variables to create temporal features for
    TemporalVars = ["MAP", "HR", "SpO2"]
    
    new_cols = []
    for var in TemporalVars:
        # 1. Delta (hourly change)
        df[f"{var}_delta"] = df.groupby("stay_id")[var].diff().fillna(0.0)
        
        # 2. Rolling mean (3h window)
        df[f"{var}_roll3_mean"] = df.groupby("stay_id")[var].transform(
            lambda s: s.rolling(window=3, min_periods=1).mean())
        
        # 3. Rolling std (3h window) - measures volatility
        df[f"{var}_roll3_std"] = df.groupby("stay_id")[var].transform(
            lambda s: s.rolling(window=3, min_periods=1).std().fillna(0.0))
            
        new_cols += [f"{var}_delta", f"{var}_roll3_mean", f"{var}_roll3_std"]

    print(f"    Added {len(new_cols)} temporal features: {', '.join(new_cols)}")
    return df


def split_dataset(df):
    print("[7/7] Train/val/test split by subject_id (70/15/15, stratified by mortality) ...")
    subject_mort = (df.groupby("subject_id")["hospital_expire_flag"]
                    .max().reset_index()
                    .rename(columns={"hospital_expire_flag": "mortality"}))

    # Stratified split
    train_val, test = train_test_split(
        subject_mort, test_size=0.15, random_state=SEED,
        stratify=subject_mort["mortality"])
    train, val = train_test_split(
        train_val, test_size=0.15 / 0.85, random_state=SEED,
        stratify=train_val["mortality"])

    def label_split(subject_id):
        if subject_id in train["subject_id"].values:
            return "train"
        elif subject_id in val["subject_id"].values:
            return "val"
        return "test"

    df["split"] = df["subject_id"].apply(label_split)

    split_counts = df.groupby("split")["stay_id"].nunique()
    print(f"    Stays per split:\n{split_counts.to_string()}")

    # Verify no patient leakage
    for s1, s2 in [("train", "val"), ("train", "test"), ("val", "test")]:
        s1_subs = set(df[df["split"] == s1]["subject_id"])
        s2_subs = set(df[df["split"] == s2]["subject_id"])
        overlap = s1_subs & s2_subs
        assert len(overlap) == 0, f"Patient leakage between {s1} and {s2}: {overlap}"
    print("    No patient leakage detected OK")

    report = {
        "n_subjects":    {"train": int(train.shape[0]),
                          "val":   int(val.shape[0]),
                          "test":  int(test.shape[0])},
        "n_stays":       split_counts.to_dict(),
        "mortality_rate": df.groupby("split")["hospital_expire_flag"].mean().to_dict(),
        "patient_leakage": {"train_val": 0, "train_test": 0, "val_test": 0},
    }
    with open(os.path.join(OUT_DIR, "split_report.json"), "w") as f:
        json.dump(report, f, indent=2)

    return df

def main():
    print("=" * 60)
    print("  MIMIC-IV Continuous Time-Series Dataset Builder")
    print("=" * 60)

    cohort, stats = load_cohort()
    chart_df = load_chart_vitals(cohort)
    lab_df   = load_lab_values(cohort)
    interv_df = load_interventions(cohort)

    df = build_hourly_grid(cohort, chart_df, lab_df, interv_df)

    # Dataset Attrition Part 2: Omitted because of missing MAP in the window
    stays_with_map = df[df["MAP"].notna()]["stay_id"].unique()
    stats["after_map_presence_filter"] = len(stays_with_map)
    
    # Filter df to only include stays with at least one MAP measurement
    df = df[df["stay_id"].isin(stays_with_map)].copy()

    df = impute(df)
    df = feature_engineering(df)
    df = split_dataset(df)

    # Save filtering statistics
    stats_path = os.path.join(OUT_DIR, "filtering_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"    Filtering statistics saved to {stats_path}")

    # Save full long-format dataset and per-split CSVs
    df.to_csv(os.path.join(OUT_DIR, "long_format.csv"), index=False)
    for split in ["train", "val", "test"]:
        df[df["split"] == split].to_csv(
            os.path.join(OUT_DIR, f"{split}.csv"), index=False)

    print(f"\nOK Saved to {OUT_DIR}")
    print(f"  long_format.csv  : {df.shape}")
    print(f"  train.csv rows   : {(df['split']=='train').sum()}")
    print(f"  val.csv rows     : {(df['split']=='val').sum()}")
    print(f"  test.csv rows    : {(df['split']=='test').sum()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
