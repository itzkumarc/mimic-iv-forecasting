"""
03_model_prophet.py
===================
Time-Series Model: Facebook Prophet
  - Target:    MAP (Mean Arterial Pressure)
  - Strategy:  Univariate forecasting using trend and seasonality
  - Outputs:   per-patient predictions, aggregate metrics, forecast plots
"""

import os
import warnings
from collections import defaultdict
import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

# Suppress Prophet logs
logging.getLogger('prophet').setLevel(logging.ERROR)
logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "outputs", "prophet")
os.makedirs(OUT_DIR, exist_ok=True)

HORIZON_H  = 6      # match deep learning pipeline
HISTORY_H  = 24     # minimum history required
TARGET     = "MAP"

def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
    yt, yp = y_true[mask], y_pred[mask]
    if len(yt) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
    mae    = np.mean(np.abs(yt - yp))
    rmse   = np.sqrt(np.mean((yt - yp) ** 2))
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - yt.mean()) ** 2)
    r2     = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((yt - yp) / np.abs(yt))[np.abs(yt) > 0]) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def main():
    print("=" * 60)
    print("  Model: Prophet -- target=MAP")
    print("=" * 60)

    df = pd.read_csv(os.path.join(DATA_DIR, "long_format.csv"))
    test_df  = df[df["split"] == "test"].copy()
    stay_ids = sorted(test_df["stay_id"].unique())
    print(f"Test stays: {len(stay_ids)}")

    all_results  = []
    plot_records = []
    
    all_y_true = []
    all_y_pred = []
    
    # Storage for evaluation script compatibility (npy files)
    all_preds_npy = []
    all_true_npy  = []

    for i, sid in enumerate(stay_ids):
        pat = test_df[test_df["stay_id"] == sid].sort_values("hour_bin")
        if len(pat) < HISTORY_H + HORIZON_H:
            continue
            
        history = pat.iloc[:-HORIZON_H].copy()
        future_true = pat.iloc[-HORIZON_H:].copy()
        
        # Prophet expects 'ds' (datestamp) and 'y' (value)
        # Create dummy timestamps based on hour_bin
        base_time = pd.Timestamp("2026-01-01")
        m_df = history[["hour_bin", TARGET]].rename(columns={"hour_bin": "ds", TARGET: "y"})
        m_df["ds"] = m_df["ds"].apply(lambda x: base_time + pd.Timedelta(hours=x))
        
        # Fit Prophet
        m = Prophet(
            changepoint_prior_scale=0.05,
            seasonality_prior_scale=10.0,
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False
        )
        m.add_country_holidays(country_name='US') # Optional, but shouldn't hurt
        m.fit(m_df)
        
        # Predict
        future = m.make_future_dataframe(periods=HORIZON_H, freq='h')
        forecast = m.predict(future)
        
        y_pred = forecast["yhat"].values[-HORIZON_H:]
        y_true = future_true[TARGET].values
        
        # Clip to reasonable MAP range
        y_pred = np.clip(y_pred, 20, 180)
        
        res_m = compute_metrics(y_true, y_pred)
        all_results.append({"stay_id": sid, **res_m})
        
        all_y_true.extend(y_true.tolist())
        all_y_pred.extend(y_pred.tolist())
        
        all_preds_npy.append(y_pred)
        all_true_npy.append(y_true)
        
        if len(plot_records) < 5:
            plot_records.append({
                "stay_id":       sid,
                "history_hours": history["hour_bin"].values,
                "history_vals":  history[TARGET].values,
                "hours_true":    future_true["hour_bin"].values,
                "y_true":        y_true,
                "y_pred":        y_pred
            })

    # Combined metrics
    pooled = compute_metrics(np.array(all_y_true), np.array(all_y_pred))
    n_pts  = len(all_y_true)
    
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(os.path.join(OUT_DIR, "predictions_per_patient.csv"), index=False)

    metric_rows = []
    for metric in ["MAE", "RMSE", "R2", "MAPE"]:
        metric_rows.append({
            "model": "Prophet", "target": TARGET, "metric": metric,
            "value": pooled[metric],
            "std":   res_df[metric].std(),
            "n_patients": len(res_df),
            "n_points": n_pts
        })
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)
    
    # Save npy files for evaluation script compatibility
    np.save(os.path.join(OUT_DIR, "preds_test.npy"), np.array(all_preds_npy))
    np.save(os.path.join(OUT_DIR, "y_true_test.npy"), np.array(all_true_npy))

    print("\n-- Prophet Metrics (MAP, pooled) --")
    for _, r in metrics_df.iterrows():
        print(f"  {r['metric']:6s}: {r['value']:.4f} (n={len(res_df)})")

    # -- Plots -------------------------------------------------------------
    n_rows = len(plot_records)
    if n_rows > 0:
        fig, axes = plt.subplots(n_rows, 1, figsize=(14, 4 * n_rows))
        if n_rows == 1: axes = [axes]
        for ax, rec in zip(axes, plot_records):
            ax.plot(rec["history_hours"], rec["history_vals"], color="#1f77b4", label="History")
            ax.plot(rec["hours_true"], rec["y_true"], color="black", linestyle="--", label="Observed")
            ax.plot(rec["hours_true"], rec["y_pred"], color="purple", label="Prophet Forecast")
            ax.axvline(rec["history_hours"][-1], color="gray", linestyle=":")
            ax.set_title(f"stay {rec['stay_id']}")
            ax.legend(fontsize=7)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "forecast_trajectories.png"), dpi=150)
        plt.close()

if __name__ == "__main__":
    main()
