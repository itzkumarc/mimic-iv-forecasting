import os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_LSTM = os.path.join(BASE_DIR, "outputs", "lstm")
OUT_ENS  = os.path.join(BASE_DIR, "outputs", "ensemble")

def analyze_horizon(path, name):
    if not os.path.exists(path):
        print(f"Skipping {name}, path not found.")
        return
    
    y_true_path = os.path.join(path, "y_true_test.npy")
    if not os.path.exists(y_true_path):
        y_true_path = os.path.join(OUT_LSTM, "y_true_test.npy")
    
    y_true = np.load(y_true_path)
    y_pred = np.load(os.path.join(path, "preds_test.npy"))
    
    print(f"\n--- {name} Horizon-wise R2 Analysis ---")
    n_horizons = y_true.shape[1]
    horizons = [1, 3, 6, 12]
    for h in horizons:
        if h > n_horizons:
            continue
        # Index h-1 corresponds to hour h
        yt = y_true[:, h-1]
        yp = y_pred[:, h-1]
        mask = ~np.isnan(yt) & ~np.isnan(yp)
        r2 = r2_score(yt[mask], yp[mask])
        print(f"  Hour {h:2d} R2: {r2:.4f}")

def main():
    analyze_horizon(OUT_LSTM, "LSTM")
    analyze_horizon(OUT_ENS, "Ensemble")

if __name__ == "__main__":
    main()
