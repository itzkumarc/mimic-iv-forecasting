"""
05_model_tft.py
===============
Custom Model 2: Temporal Fusion Transformer / GRU fallback
  - Target:     MAP (single output)
  - Covariates: HR, RR, SpO2, SysBP, DiaBP, Temperature,
                Creatinine, WBC, Glucose, Lactate, IV_Fluid_mL
  - Categorical: Vasopressor (binary)
  - Encoder: 24h -> Prediction: 12h
"""

import os
import time
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

try:
    import pytorch_lightning as pl
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data import GroupNormalizer
    from pytorch_forecasting.metrics import MAE as PFMAE
    TFT_AVAILABLE = True
except ImportError:
    TFT_AVAILABLE = False
    print("WARNING: pytorch_forecasting not available -- using GRU fallback.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "outputs", "tft")
os.makedirs(OUT_DIR, exist_ok=True)

ENCODER_LEN = 24
PREDICT_LEN = 6
TARGET      = "MAP"
REAL_COVS   = ["HR", "RR", "SpO2", "SysBP", "DiaBP", "Temperature",
               "Creatinine", "WBC", "Glucose", "Lactate", "IV_Fluid_mL",
               "MAP_delta", "MAP_roll3_mean", "MAP_roll3_std",
               "HR_delta", "HR_roll3_mean", "HR_roll3_std",
               "SpO2_delta", "SpO2_roll3_mean", "SpO2_roll3_std"]
FEATURES    = [TARGET] + REAL_COVS + ["Vasopressor"]
N_FEAT      = len(FEATURES)
TGT_IDX     = 0
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    yt = np.array(y_true, dtype=float).flatten()
    yp = np.array(y_pred, dtype=float).flatten()
    mask = ~np.isnan(yt) & ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]
    if len(yt) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "MAPE": np.nan}
    mae  = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    ss_r = np.sum((yt - yp) ** 2)
    ss_t = np.sum((yt - yt.mean()) ** 2)
    r2   = 1 - ss_r / ss_t if ss_t > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((yt - yp) / np.abs(yt))[np.abs(yt) > 0]) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# --------------------------------------------------------------------------
# Shared forecast plots
# --------------------------------------------------------------------------
def plot_forecasts(X_te, y_true, y_pred, df_test, label="TFT/GRU", n=5):
    sub = df_test.sort_values(["stay_id", "hour_bin"])
    stay_ids = sub["stay_id"].unique()
    stay_map = {}
    w = 0
    for sid in stay_ids:
        nw = max(0, len(sub[sub["stay_id"] == sid]) - ENCODER_LEN - PREDICT_LEN + 1)
        if nw > 0:
            stay_map[sid] = w
            w += nw

    chosen = list(stay_map.keys())[:n]
    if not chosen:
        return

    fig, axes = plt.subplots(len(chosen), 1, figsize=(14, 4 * len(chosen)))
    if len(chosen) == 1:
        axes = [axes]

    for ax, sid in zip(axes, chosen):
        wi = stay_map[sid]
        hist_h = np.arange(-ENCODER_LEN + 1, 1)
        fore_h = np.arange(1, PREDICT_LEN + 1)
        hist_v = X_te[wi, :, TGT_IDX] if X_te is not None else np.full(ENCODER_LEN, np.nan)

        ax.plot(hist_h, hist_v, color="#1f77b4", linewidth=1.5, label="History")
        ax.plot(fore_h, y_true[wi], color="black", linewidth=2,
                linestyle="--", label="Observed")
        ax.plot(fore_h, y_pred[wi], color="#2ca02c", linewidth=2,
                marker="s", markersize=4, label=f"{label} forecast")
        ax.axhline(65, color="red", linestyle=":", linewidth=1, alpha=0.7)
        ax.axvline(0, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.set_title(f"stay {sid} -- MAP", fontsize=10, fontweight="bold")
        ax.set_xlabel("Hour relative to forecast origin", fontsize=8)
        ax.set_ylabel("MAP (mmHg)", fontsize=8)
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle(f"{label} -- MAP Forecasts on Test Patients",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "forecast_trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Forecast plot saved: {path}")


def plot_horizon_mae(y_true, y_pred):
    mae_per_step = np.mean(np.abs(y_true - y_pred), axis=0)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(1, PREDICT_LEN + 1), mae_per_step,
           color="#2ca02c", alpha=0.85)
    ax.set_xlabel("Forecast horizon (h)", fontsize=10)
    ax.set_ylabel("MAE (mmHg)", fontsize=10)
    ax.set_title("GRU/TFT -- MAP Forecast Error by Horizon Step", fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "horizon_mae.png"), dpi=150)
    plt.close()


# --------------------------------------------------------------------------
# GRU fallback
# --------------------------------------------------------------------------
class GRUModel(torch.nn.Module):
    def __init__(self, n_feat, hidden=128, n_layers=2, dropout=0.2, horizon=12):
        super().__init__()
        self.gru  = torch.nn.GRU(n_feat, hidden, n_layers,
                                  batch_first=True, dropout=dropout)
        self.head = torch.nn.Sequential(
            torch.nn.Linear(hidden, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, horizon),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1])    # (B, horizon)


def run_gru_fallback(df):
    from torch.utils.data import TensorDataset, DataLoader

    for col in FEATURES:
        if col not in df.columns:
            df[col] = 0.0

    def make_windows(split):
        sub = df[df["split"] == split].sort_values(["stay_id", "hour_bin"])
        Xs, ys = [], []
        for _, pat in sub.groupby("stay_id"):
            vals = pat[FEATURES].values.astype(np.float32)
            if len(vals) < ENCODER_LEN + PREDICT_LEN:
                continue
            for start in range(len(vals) - ENCODER_LEN - PREDICT_LEN + 1):
                Xs.append(vals[start : start + ENCODER_LEN])
                ys.append(vals[start + ENCODER_LEN : start + ENCODER_LEN + PREDICT_LEN, TGT_IDX])
        return (np.array(Xs, np.float32), np.array(ys, np.float32)) if Xs else (None, None)

    X_tr, y_tr = make_windows("train")
    X_vl, y_vl = make_windows("val")
    X_te, y_te = make_windows("test")
    if X_tr is None or X_te is None:
        return None, None, None

    f_mean = X_tr.mean((0, 1), keepdims=True)
    f_std  = X_tr.std((0, 1),  keepdims=True) + 1e-8
    X_tr_n, X_vl_n, X_te_n = (X_tr-f_mean)/f_std, (X_vl-f_mean)/f_std, (X_te-f_mean)/f_std
    t_mean = float(y_tr.mean())
    t_std  = float(y_tr.std()) + 1e-8
    y_tr_n = (y_tr - t_mean) / t_std
    y_vl_n = (y_vl - t_mean) / t_std
    y_te_n = (y_te - t_mean) / t_std

    model = GRUModel(N_FEAT, horizon=PREDICT_LEN).to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    crit  = torch.nn.MSELoss()
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)

    tr_ds = DataLoader(TensorDataset(torch.tensor(X_tr_n), torch.tensor(y_tr_n)),
                       batch_size=32, shuffle=True)
    vl_ds = DataLoader(TensorDataset(torch.tensor(X_vl_n), torch.tensor(y_vl_n)),
                       batch_size=64)
    te_ds = DataLoader(TensorDataset(torch.tensor(X_te_n), torch.tensor(y_te_n)),
                       batch_size=64)

    best_v, best_s, pat_cnt = np.inf, None, 0
    tr_losses, vl_losses = [], []
    MAX_EPOCHS_GRU = 150
    PATIENCE_GRU   = 20

    for epoch in range(1, MAX_EPOCHS_GRU + 1):
        # ---- proper training step ----
        model.train()
        tl = 0.0
        for xb, yb in tr_ds:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(xb)
        tl /= len(tr_ds.dataset)

        # ---- validation ----
        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in vl_ds:
                vl += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        vl /= len(vl_ds.dataset)

        tr_losses.append(tl)
        vl_losses.append(vl)
        sched.step(vl)

        if vl < best_v - 1e-5:
            best_v  = vl
            best_s  = {k: v.clone() for k, v in model.state_dict().items()}
            pat_cnt = 0
        else:
            pat_cnt += 1

        if epoch % 10 == 0:
            current_lr = opt.param_groups[0]["lr"]
            print(f"    Epoch {epoch:3d}: train={tl:.4f}  val={vl:.4f}  lr={current_lr:.2e}")
        if pat_cnt >= PATIENCE_GRU:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_s:
        model.load_state_dict(best_s)
    model.eval()
    preds_n = []
    with torch.no_grad():
        for xb, _ in te_ds:
            preds_n.append(model(xb.to(DEVICE)).cpu().numpy())
    preds_n = np.concatenate(preds_n, 0)
    y_pred_raw = preds_n * t_std + t_mean
    y_true_raw = y_te_n * t_std + t_mean

    # Training curves
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tr_losses, label="Train", color="#1f77b4")
    ax.plot(vl_losses, label="Val",   color="#d62728")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (normalised)")
    ax.set_title("GRU/TFT Training Curves -- MAP", fontweight="bold")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close()

    torch.save(model.state_dict(), os.path.join(OUT_DIR, "tft_weights.pt"))
    return X_te, y_true_raw, y_pred_raw


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"  Model: TFT/GRU -- target=MAP  (device={DEVICE})")
    if not TFT_AVAILABLE:
        print("  [GRU fallback active]")
    print("=" * 60)
    t0 = time.time()

    df      = pd.read_csv(os.path.join(DATA_DIR, "long_format.csv"))
    df_test = df[df["split"] == "test"].copy()

    if TFT_AVAILABLE:
        try:
            X_te, y_true_raw, y_pred_raw = run_tft(df)
        except Exception as exc:
            print(f"  TFT failed ({exc}); using GRU fallback ...")
            X_te, y_true_raw, y_pred_raw = run_gru_fallback(df)
    else:
        X_te, y_true_raw, y_pred_raw = run_gru_fallback(df)

    if y_true_raw is None:
        print("ERROR: insufficient data."); return

    np.save(os.path.join(OUT_DIR, "preds_test.npy"),  y_pred_raw)
    np.save(os.path.join(OUT_DIR, "y_true_test.npy"), y_true_raw)
    if X_te is not None:
        np.save(os.path.join(OUT_DIR, "X_test_raw.npy"), X_te)

    metrics = compute_metrics(y_true_raw, y_pred_raw)
    rows = [{"model": "TFT/GRU", "target": TARGET,
             "metric": k, "value": v} for k, v in metrics.items()]
    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    print("\n-- TFT/GRU Metrics (MAP, test set) --")
    for k, v in metrics.items():
        print(f"  {k:6s}: {v:.4f}")

    plot_forecasts(X_te, y_true_raw, y_pred_raw, df_test)
    plot_horizon_mae(y_true_raw, y_pred_raw)

    print(f"\nTFT/GRU complete ({time.time()-t0:.1f}s).")


if __name__ == "__main__":
    main()
