"""
04_model_lstm.py
================
Custom Model 1: Multi-variate multi-step LSTM (PyTorch)
  - Target:     MAP (single output)
  - Covariates: HR, RR, SpO2, SysBP, DiaBP, Temperature,
                Creatinine, WBC, Glucose, Lactate, Vasopressor, IV_Fluid_mL
  - Sliding window: 24h lookback -> 12h forecast
  - Architecture: 2-layer stacked LSTM + linear head
  - Training: Adam, MSE, early stopping on validation loss
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
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
OUT_DIR  = os.path.join(BASE_DIR, "outputs", "lstm")
os.makedirs(OUT_DIR, exist_ok=True)

# -- Hyper-parameters ---------------------------------------------------------
SEQ_LEN    = 24
HORIZON    = 6
BATCH_SIZE = 32
MAX_EPOCHS = 150
LR         = 1e-3
PATIENCE   = 20
HIDDEN_DIM = 128
N_LAYERS   = 2
DROPOUT    = 0.2

TARGET   = "MAP"
FEATURES = ["MAP", "HR", "RR", "SpO2", "SysBP", "DiaBP", "Temperature",
            "Creatinine", "WBC", "Glucose", "Lactate", "Vasopressor", "IV_Fluid_mL",
            "MAP_delta", "MAP_roll3_mean", "MAP_roll3_std",
            "HR_delta", "HR_roll3_mean", "HR_roll3_std",
            "SpO2_delta", "SpO2_roll3_mean", "SpO2_roll3_std"]
N_FEAT   = len(FEATURES)
DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TGT_IDX  = FEATURES.index(TARGET)   # index 0


# --------------------------------------------------------------------------
# Dataset construction
# --------------------------------------------------------------------------
def make_windows(df, split_name):
    sub = df[df["split"] == split_name].sort_values(["stay_id", "hour_bin"])
    for col in FEATURES:
        if col not in sub.columns:
            sub[col] = 0.0

    Xs, ys = [], []
    for sid, pat in sub.groupby("stay_id"):
        vals = pat[FEATURES].values.astype(np.float32)  # (T, F)
        if len(vals) < SEQ_LEN + HORIZON:
            continue
        for start in range(len(vals) - SEQ_LEN - HORIZON + 1):
            Xs.append(vals[start : start + SEQ_LEN])
            # only MAP for target
            ys.append(vals[start + SEQ_LEN : start + SEQ_LEN + HORIZON, TGT_IDX])

    if not Xs:
        return None, None
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)
    # shapes: (N, SEQ_LEN, N_FEAT), (N, HORIZON)


def normalise(X_tr, X_vl, X_te):
    mean = X_tr.mean(axis=(0, 1), keepdims=True)
    std  = X_tr.std(axis=(0, 1),  keepdims=True) + 1e-8
    return (X_tr-mean)/std, (X_vl-mean)/std, (X_te-mean)/std, mean, std


def normalise_targets(y_tr, y_vl, y_te):
    mean = float(y_tr.mean())
    std  = float(y_tr.std()) + 1e-8
    return (y_tr-mean)/std, (y_vl-mean)/std, (y_te-mean)/std, mean, std


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------
class MultiStepLSTM(nn.Module):
    def __init__(self, n_feat, hidden, n_layers, dropout, horizon):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers,
                            batch_first=True, dropout=dropout)
        
        # Dot-product attention
        self.query = nn.Linear(hidden, hidden)
        self.key   = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, horizon),
        )
        self.horizon = horizon

    def forward(self, x):
        # x: (B, seq_len, feat)
        lstm_out, (h_n, c_n) = self.lstm(x) # lstm_out: (B, seq_len, hidden)
        
        # Last hidden state as query
        q = self.query(h_n[-1]).unsqueeze(1) # (B, 1, hidden)
        k = self.key(lstm_out)               # (B, seq_len, hidden)
        v = self.value(lstm_out)             # (B, seq_len, hidden)
        
        # Attention scores
        attn_weights = torch.bmm(q, k.transpose(1, 2)) # (B, 1, seq_len)
        attn_weights = torch.softmax(attn_weights, dim=-1)
        
        # Context vector
        context = torch.bmm(attn_weights, v).squeeze(1) # (B, hidden)
        
        # Residual connection
        context = context + h_n[-1]
        
        return self.head(context)    # (B, horizon)


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
def train_model(model, tr_dl, vl_dl):
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    crit   = nn.MSELoss()
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5, min_lr=1e-5)
    best_v = np.inf
    best_s = None
    p_cnt  = 0
    tr_losses, vl_losses = [], []

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        tl = 0.0
        for xb, yb in tr_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(xb)
        tl /= len(tr_dl.dataset)
        tr_losses.append(tl)

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in vl_dl:
                vl += crit(model(xb.to(DEVICE)), yb.to(DEVICE)).item() * len(xb)
        vl /= len(vl_dl.dataset)
        vl_losses.append(vl)
        sched.step(vl)

        if vl < best_v - 1e-5:
            best_v = vl
            best_s = {k: v.clone() for k, v in model.state_dict().items()}
            p_cnt  = 0
        else:
            p_cnt += 1

        if epoch % 10 == 0:
            current_lr = opt.param_groups[0]["lr"]
            print(f"    Epoch {epoch:3d}: train={tl:.4f}  val={vl:.4f}  lr={current_lr:.2e}")
        if p_cnt >= PATIENCE:
            print(f"    Early stopping at epoch {epoch}")
            break

    if best_s:
        model.load_state_dict(best_s)
    return tr_losses, vl_losses


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """y_true, y_pred: (N, HORIZON) in original scale."""
    yt = y_true.flatten()
    yp = y_pred.flatten()
    mask = ~np.isnan(yt) & ~np.isnan(yp)
    yt, yp = yt[mask], yp[mask]
    mae  = np.mean(np.abs(yt - yp))
    rmse = np.sqrt(np.mean((yt - yp) ** 2))
    ss_r = np.sum((yt - yp) ** 2)
    ss_t = np.sum((yt - yt.mean()) ** 2)
    r2   = 1 - ss_r / ss_t if ss_t > 0 else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((yt - yp) / np.abs(yt))[np.abs(yt) > 0]) * 100
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_forecasts(X_te, y_true, y_pred, df_test, n=5):
    sub = df_test.sort_values(["stay_id", "hour_bin"])
    stay_ids = sub["stay_id"].unique()
    stay_map = {}
    w = 0
    for sid in stay_ids:
        pat = sub[sub["stay_id"] == sid]
        nw  = max(0, len(pat) - SEQ_LEN - HORIZON + 1)
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
        wi  = stay_map[sid]
        hist_h = np.arange(-SEQ_LEN + 1, 1)
        fore_h = np.arange(1, HORIZON + 1)
        hist_v = X_te[wi, :, TGT_IDX]

        ax.plot(hist_h, hist_v, color="#1f77b4", linewidth=1.5, label="History")
        ax.plot(fore_h, y_true[wi], color="black", linewidth=2,
                linestyle="--", label="Observed")
        ax.plot(fore_h, y_pred[wi], color="orange", linewidth=2,
                marker="o", markersize=4, label="LSTM forecast")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
        ax.axhline(65, color="red", linestyle=":", linewidth=1, alpha=0.7)
        ax.set_title(f"stay {sid} -- MAP", fontsize=10, fontweight="bold")
        ax.set_xlabel("Hour relative to forecast origin", fontsize=8)
        ax.set_ylabel("MAP (mmHg)", fontsize=8)
        ax.legend(fontsize=7, ncol=4)
        ax.grid(True, linestyle="--", alpha=0.5)

    plt.suptitle("LSTM -- MAP Forecasts on Test Patients",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "forecast_trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Forecast plot saved: {path}")


def plot_training(tr_losses, vl_losses):
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(tr_losses, label="Train", color="#1f77b4")
    ax.plot(vl_losses, label="Val",   color="#d62728")
    ax.set_xlabel("Epoch"); ax.set_ylabel("MSE (normalised)")
    ax.set_title("LSTM Training & Validation Loss -- MAP", fontweight="bold")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "training_curves.png"), dpi=150)
    plt.close()


def plot_horizon_mae(y_true, y_pred):
    """MAE per forecast horizon step."""
    mae_per_step = np.mean(np.abs(y_true - y_pred), axis=0)   # (HORIZON,)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(np.arange(1, HORIZON + 1), mae_per_step, color="#1f77b4", alpha=0.85)
    ax.set_xlabel("Forecast horizon (h)", fontsize=10)
    ax.set_ylabel("MAE (mmHg)", fontsize=10)
    ax.set_title("LSTM -- MAP Forecast Error by Horizon Step", fontweight="bold")
    ax.yaxis.grid(True, linestyle="--", alpha=0.6)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "horizon_mae.png"), dpi=150)
    plt.close()
    print(f"  Horizon MAE plot saved.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print("=" * 60)
    print(f"  Model: LSTM -- target=MAP  (device={DEVICE})")
    print("=" * 60)
    t0 = time.time()

    df = pd.read_csv(os.path.join(DATA_DIR, "long_format.csv"))
    df_test = df[df["split"] == "test"].copy()

    print("Building sliding-window datasets ...")
    X_tr, y_tr = make_windows(df, "train")
    X_vl, y_vl = make_windows(df, "val")
    X_te, y_te = make_windows(df, "test")

    if X_tr is None or X_te is None:
        print("ERROR: not enough data."); return

    print(f"  Windows  train={len(X_tr)}  val={len(X_vl)}  test={len(X_te)}")

    X_tr_n, X_vl_n, X_te_n, f_mean, f_std = normalise(X_tr, X_vl, X_te)
    y_tr_n, y_vl_n, y_te_n, t_mean, t_std = normalise_targets(y_tr, y_vl, y_te)
    np.save(os.path.join(OUT_DIR, "feat_mean.npy"), f_mean)
    np.save(os.path.join(OUT_DIR, "feat_std.npy"),  f_std)

    def dl(X, y, shuf=False):
        ds = TensorDataset(torch.tensor(X), torch.tensor(y))
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuf, num_workers=0)

    tr_dl = dl(X_tr_n, y_tr_n, shuf=True)
    vl_dl = dl(X_vl_n, y_vl_n)
    te_dl = dl(X_te_n, y_te_n)

    model = MultiStepLSTM(N_FEAT, HIDDEN_DIM, N_LAYERS, DROPOUT, HORIZON).to(DEVICE)
    print(f"  LSTM parameters: {sum(p.numel() for p in model.parameters()):,}")

    print("\nTraining ...")
    tr_l, vl_l = train_model(model, tr_dl, vl_dl)
    plot_training(tr_l, vl_l)
    torch.save(model.state_dict(), os.path.join(OUT_DIR, "lstm_weights.pt"))

    model.eval()
    preds_n = []
    with torch.no_grad():
        for xb, _ in te_dl:
            preds_n.append(model(xb.to(DEVICE)).cpu().numpy())
    preds_n = np.concatenate(preds_n, 0)  # (N, HORIZON)

    y_pred_raw = preds_n * t_std + t_mean
    y_true_raw = y_te_n * t_std + t_mean

    np.save(os.path.join(OUT_DIR, "preds_test.npy"),  y_pred_raw)
    np.save(os.path.join(OUT_DIR, "y_true_test.npy"), y_true_raw)
    np.save(os.path.join(OUT_DIR, "X_test_raw.npy"),  X_te)

    metrics = compute_metrics(y_true_raw, y_pred_raw)
    metric_rows = [{"model": "LSTM", "target": TARGET,
                    "metric": k, "value": v} for k, v in metrics.items()]
    pd.DataFrame(metric_rows).to_csv(os.path.join(OUT_DIR, "metrics.csv"), index=False)

    print("\n-- LSTM Metrics (MAP, test set) --")
    for k, v in metrics.items():
        print(f"  {k:6s}: {v:.4f}")

    plot_forecasts(X_te, y_true_raw, y_pred_raw, df_test)
    plot_horizon_mae(y_true_raw, y_pred_raw)

    print(f"\nLSTM complete ({time.time()-t0:.1f}s).")


if __name__ == "__main__":
    main()
