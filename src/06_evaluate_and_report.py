"""
06_evaluate_and_report.py
=========================
Aggregate results from all 3 models (Prophet, LSTM, TFT/GRU).
Target: MAP only.

Outputs -> outputs/results/:
  leaderboard.csv          -- MAE/RMSE/R2/MAPE per model
  leaderboard_plot.png     -- bar chart
  forecast_overlay.png     -- 5 test patients, all 3 models overlaid
  horizon_comparison.png   -- MAE per horizon step (h+1 ... h+12)
  error_distribution.png   -- per-patient MAE box plots
  summary.txt              -- plain-English interpretation
"""

import os
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "processed")
OUT_PROPHET = os.path.join(BASE_DIR, "outputs", "prophet")
OUT_LSTM    = os.path.join(BASE_DIR, "outputs", "lstm")
OUT_TFT     = os.path.join(BASE_DIR, "outputs", "tft")
OUT_ENS     = os.path.join(BASE_DIR, "outputs", "ensemble")
OUT_DIR     = os.path.join(BASE_DIR, "outputs", "results")
os.makedirs(OUT_DIR, exist_ok=True)

TARGET         = "MAP"
MODEL_DIRS     = {"Prophet": OUT_PROPHET, "LSTM": OUT_LSTM, "TFT/GRU": OUT_TFT, "Ensemble": OUT_ENS}
MODEL_COLORS   = {"Prophet": "#e377c2", "LSTM": "#1f77b4", "TFT/GRU": "#2ca02c", "Ensemble": "#d62728"}
METRICS_ORDER  = ["MAE", "RMSE", "R2", "MAPE"]
ENCODER_LEN    = 24
PREDICT_LEN    = 6


def load_array(path):
    return np.load(path) if os.path.exists(path) else None


# --------------------------------------------------------------------------
# 1. Leaderboard
# --------------------------------------------------------------------------
def build_leaderboard():
    rows = []
    for model, mdir in MODEL_DIRS.items():
        mp = os.path.join(mdir, "metrics.csv")
        if not os.path.exists(mp):
            print(f"  WARNING: {mp} missing -- skipping {model}")
            continue
        mdf = pd.read_csv(mp)
        if "metric" in mdf.columns and "value" in mdf.columns:
            for _, r in mdf.iterrows():
                rows.append({"Model": model, "Metric": r["metric"],
                             "Value": r["value"]})
        else:
            for metric in METRICS_ORDER:
                if metric in mdf.columns:
                    rows.append({"Model": model, "Metric": metric,
                                 "Value": mdf[metric].mean()})

    if not rows:
        print("ERROR: no metrics found."); return None

    lb = pd.DataFrame(rows)
    lb_pivot = lb.pivot_table(index="Model", columns="Metric",
                               values="Value").reset_index()
    lb_pivot.to_csv(os.path.join(OUT_DIR, "leaderboard.csv"), index=False)
    lb.to_csv(os.path.join(OUT_DIR, "leaderboard_long.csv"), index=False)
    print(f"  Leaderboard saved ({len(lb_pivot)} models).")
    return lb


# --------------------------------------------------------------------------
# 2. Leaderboard bar chart
# --------------------------------------------------------------------------
def plot_leaderboard(lb):
    if lb is None or lb.empty: return
    metrics       = ["MAE", "RMSE", "R2", "MAPE"]
    better_lower  = {"MAE": True, "RMSE": True, "R2": False, "MAPE": True}

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for ax, metric in zip(axes.flatten(), metrics):
        sub    = lb[lb["Metric"] == metric].copy()
        models = [m for m in MODEL_DIRS if m in sub["Model"].values]
        vals   = [sub[sub["Model"] == m]["Value"].values[0] for m in models]
        colors = [MODEL_COLORS.get(m, "#888") for m in models]
        bars   = ax.bar(models, vals, color=colors, alpha=0.85, edgecolor="white")
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width()/2,
                        bar.get_height() * (1.01 if val > 0 else 0.99),
                        f"{val:.3f}", ha="center", va="bottom", fontsize=9,
                        fontweight="bold")
        arrow = "(lower is better)" if better_lower[metric] else "(higher is better)"
        ax.set_title(f"{metric}  {arrow}", fontsize=11, fontweight="bold")
        ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="x", labelsize=10)

    plt.suptitle(f"Model Performance Leaderboard -- {TARGET} Forecasting",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "leaderboard_plot.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Leaderboard plot saved: {path}")


# --------------------------------------------------------------------------
# 3. Forecast overlay (5 patients, all 3 models)
# --------------------------------------------------------------------------
def plot_forecast_overlay(df_test):
    # Load data
    lstm_true = load_array(os.path.join(OUT_LSTM, "y_true_test.npy"))
    lstm_X    = load_array(os.path.join(OUT_LSTM, "X_test_raw.npy"))
    
    model_preds = {}
    for name, mdir in MODEL_DIRS.items():
        p = load_array(os.path.join(mdir, "preds_test.npy"))
        if p is not None:
            model_preds[name] = p

    # Mapping
    sub = df_test.sort_values(["stay_id", "hour_bin"])
    stay_map = {}
    w = 0
    for sid in sub["stay_id"].unique():
        nw = max(0, len(sub[sub["stay_id"] == sid]) - ENCODER_LEN - PREDICT_LEN + 1)
        if nw > 0:
            stay_map[sid] = w
            w += nw

    chosen = list(stay_map.keys())[:5]
    if not chosen: return

    # Individual plots directory
    patient_dir = os.path.join(OUT_DIR, "patients")
    os.makedirs(patient_dir, exist_ok=True)

    # Styling
    model_styles = {
        "Prophet":  dict(linestyle="--", marker="v", alpha=0.6, ms=5),
        "LSTM":     dict(linestyle="-",  marker="o", alpha=0.7, ms=5),
        "TFT/GRU":  dict(linestyle="-.", marker="s", alpha=0.7, ms=5),
        "Ensemble": dict(linestyle="-",  marker="D", alpha=0.9, linewidth=2.5, ms=6),
    }

    def plot_single_stay(ax, sid, wi, is_individual=False):
        hist_h = np.arange(-ENCODER_LEN + 1, 1)
        fore_h = np.arange(1, PREDICT_LEN + 1)

        # History
        if lstm_X is not None and wi < len(lstm_X):
            ax.plot(hist_h, lstm_X[wi, :, 0], color="#555555",
                    linewidth=2, label="History", alpha=0.8)

        # Observed
        if lstm_true is not None and wi < len(lstm_true):
            ax.plot(fore_h, lstm_true[wi], color="black", linewidth=3,
                    linestyle=":", label="Observed", zorder=5)

        # Models
        for model_name, preds_array in model_preds.items():
            if wi < len(preds_array):
                style = model_styles.get(model_name, {})
                color = MODEL_COLORS.get(model_name, "blue")
                ax.plot(fore_h, preds_array[wi], color=color, 
                        label=model_name, zorder=4, **style)

        ax.axhline(65, color="#e74c3c", linestyle="--", linewidth=1.2, alpha=0.7, label="MAP < 65")
        ax.axvline(0, color="black", linestyle="-", linewidth=1.5, alpha=0.3)
        
        ax.set_title(f"Patient Stay ID: {sid}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Hours from Origin (t=0)", fontsize=10)
        ax.set_ylabel("MAP (mmHg)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.3)
        
        if is_individual:
            ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1, 1))
        else:
            ax.legend(fontsize=8, ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.15))

    # 1. Combined Aggregate Plot
    fig, axes = plt.subplots(len(chosen), 1, figsize=(14, 4.5 * len(chosen)))
    if len(chosen) == 1: axes = [axes]
    
    for ax, sid in zip(axes, chosen):
        plot_single_stay(ax, sid, stay_map[sid])

    plt.suptitle(f"Time-Series Forecasting Overlay (Target: {TARGET})\n"
                 "Phase 4: Multi-Model Precision Comparison (6h Horizon)",
                 fontsize=16, fontweight="bold", y=0.98)
    
    # Custom tight_layout to handle suptitle and legend spacing
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    path = os.path.join(OUT_DIR, "forecast_overlay.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Overlay plot saved: {path}")

    # 2. Individual Plots
    for sid in chosen:
        fig_ind, ax_ind = plt.subplots(figsize=(10, 5))
        plot_single_stay(ax_ind, sid, stay_map[sid], is_individual=True)
        plt.tight_layout()
        path_ind = os.path.join(patient_dir, f"stay_{sid}.png")
        fig_ind.savefig(path_ind, dpi=120, bbox_inches="tight")
        plt.close(fig_ind)
    print(f"  Individual patient plots ({len(chosen)}) saved to {patient_dir}")


# --------------------------------------------------------------------------
# 4. Horizon comparison (MAE per step h+1 ... h+12)
# --------------------------------------------------------------------------
def plot_horizon_comparison():
    fig, ax = plt.subplots(figsize=(11, 5))
    for model_name, mdir in MODEL_DIRS.items():
        color = MODEL_COLORS.get(model_name, "blue")
        yt = load_array(os.path.join(mdir, "y_true_test.npy"))
        yp = load_array(os.path.join(mdir, "preds_test.npy"))
        if yt is not None and yp is not None:
            mask = ~np.isnan(yt) & ~np.isnan(yp)
            mae_per_step = []
            for h in range(yt.shape[1]):
                m = mask[:, h]
                if np.any(m):
                    mae_per_step.append(np.mean(np.abs(yt[m, h] - yp[m, h])))
                else:
                    mae_per_step.append(np.nan)
            ax.plot(np.arange(1, len(mae_per_step) + 1), mae_per_step,
                    marker="o", color=color, linewidth=2, label=model_name)

    ax.set_xlabel("Forecast step (hours ahead)", fontsize=11)
    ax.set_ylabel("Mean Absolute Error (mmHg)", fontsize=11)
    ax.set_title("MAP Forecast Error by Horizon Step\n"
                 "(error should generally increase with longer horizons)",
                 fontsize=12, fontweight="bold")
    ax.set_xticks(range(1, PREDICT_LEN + 1))
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "horizon_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Horizon comparison saved: {path}")


# --------------------------------------------------------------------------
# 5. Error distribution
# --------------------------------------------------------------------------
def plot_error_distributions():
    plot_data, labels, colors = [], [], []

    # All models
    for model_name in ["Prophet", "LSTM", "TFT/GRU", "Ensemble"]:
        mdir = MODEL_DIRS.get(model_name)
        color = MODEL_COLORS.get(model_name, "blue")
        if not mdir: continue
        
        yt = load_array(os.path.join(mdir, "y_true_test.npy"))
        yp = load_array(os.path.join(mdir, "preds_test.npy"))
        if yt is not None and yp is not None:
            # Drop NaNs
            errors = np.abs(yt - yp).mean(axis=1)
            valid_errors = errors[~np.isnan(errors)]
            if len(valid_errors) > 0:
                plot_data.append(valid_errors)
                labels.append(model_name)
                colors.append(color)

    if not plot_data: return

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(plot_data, patch_artist=True, notch=True,
                    medianprops=dict(color="white", linewidth=2),
                    boxprops=dict(edgecolor="black", alpha=0.8),
                    whiskerprops=dict(linestyle="--", color="#666"),
                    flierprops=dict(marker="o", markerfacecolor="red", alpha=0.3, markersize=4))
    
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=11, fontweight="bold")
    ax.set_ylabel("Mean Absolute Error per window (mmHg)", fontsize=10)
    ax.set_title("MAP Prediction Error Variance -- Clinical Distribution", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "error_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Error distribution saved: {path}")

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Evaluation & Reporting -- MAP Target")
    print("=" * 60)

    df      = pd.read_csv(os.path.join(DATA_DIR, "long_format.csv"))
    df_test = df[df["split"] == "test"].copy()

    print("[1/5] Leaderboard ...")
    lb = build_leaderboard()
    print("[2/5] Leaderboard plot ...")
    plot_leaderboard(lb)
    print("[3/5] Forecast overlay ...")
    plot_forecast_overlay(df_test)
    print("[4/5] Horizon comparison ...")
    plot_horizon_comparison()
    print("[5/5] Error distribution ...")
    plot_error_distributions()
    write_summary(lb)
    print("\nAll results saved to outputs/results/")


if __name__ == "__main__":
    main()
