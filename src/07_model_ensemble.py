"""
07_model_ensemble.py
====================
Ensemble of LSTM and TFT/GRU models.
Weights are optimized (or simple average) to maximize R2.
Outputs -> outputs/ensemble/:
  metrics.csv
  preds_test.npy
  forecast_trajectories.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_LSTM = os.path.join(BASE_DIR, "outputs", "lstm")
OUT_TFT  = os.path.join(BASE_DIR, "outputs", "tft")
OUT_DIR  = os.path.join(BASE_DIR, "outputs", "ensemble")
os.makedirs(OUT_DIR, exist_ok=True)

def load_array(path):
    return np.load(path) if os.path.exists(path) else None

def compute_metrics(y_true, y_pred):
    yt = y_true.flatten()
    yp = y_pred.flatten()
    mask = ~np.isnan(yt) & ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]
    
    mae = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    r2 = r2_score(yt, yp)
    mape = np.mean(np.abs((yt - yp) / yt)) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

def main():
    print("=" * 60)
    print("  Model: Multi-Model Ensemble (LSTM + GRU)")
    print("=" * 60)

    y_true = load_array(os.path.join(OUT_LSTM, "y_true_test.npy"))
    p_lstm = load_array(os.path.join(OUT_LSTM, "preds_test.npy"))
    p_tft  = load_array(os.path.join(OUT_TFT,  "preds_test.npy"))

    if y_true is None or p_lstm is None or p_tft is None:
        print("ERROR: Missing prediction files. Run LSTM and TFT models first.")
        return

    # Option 1: Simple average
    # Option 2: Weighted average (LSTM performed slightly better, so 0.6 / 0.4)
    w_lstm = 0.6
    w_tft  = 0.4
    p_ens  = (w_lstm * p_lstm) + (w_tft * p_tft)

    metrics = compute_metrics(y_true, p_ens)
    print("\n-- Ensemble Metrics (MAP, test set) --")
    for k, v in metrics.items():
        print(f"  {k:6s}: {v:.4f}")

    # Save results
    np.save(os.path.join(OUT_DIR, "preds_test.npy"), p_ens)
    pd.DataFrame([metrics]).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    # Plot sample
    plt.figure(figsize=(12, 6))
    for i in range(min(3, len(y_true))):
        plt.subplot(3, 1, i+1)
        plt.plot(y_true[i], "k--", label="Observed")
        plt.plot(p_lstm[i], "b:", alpha=0.5, label="LSTM")
        plt.plot(p_tft[i], "g:", alpha=0.5, label="GRU")
        plt.plot(p_ens[i], "r-", linewidth=2, label="Ensemble")
        plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "forecast_trajectories.png"), dpi=150)
    print(f"\nEnsemble complete. Results saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
