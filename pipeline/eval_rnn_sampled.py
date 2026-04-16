"""
One-off script: run RNN inference on dataset/stock_prices_20y.db for the
sampled-window tickers/dates used in the tree/lstm/llm comparison, then
generate matching metrics and plots.

Outputs:
    pipeline/results/sampled_rnn_2026-03-16_2026-04-15_rows.csv
  pipeline/results/sampled_rnn_2026-03-16_2026-04-15_daily.csv
  pipeline/results/sampled_summary_2026-03-16_2026-04-15.csv   (rnn row appended)
  pipeline/results/plots/sampled_rnn_2026-03-16_2026-04-15_metrics.png
  pipeline/results/plots/comparison_2026-03-16_2026-04-15_metrics.png
"""

from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  # noqa: F401 (needed for unpickling)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RNN_MODEL_PKL = Path("artifacts/model_outputs/models/20260415_233715-rnn_torch.pkl")
DB_PATH       = Path("dataset/stock_prices_20y.db")
TICKERS_CSV   = Path("pipeline/results/sampled_pipeline_tickers_2026-03-16_2026-04-15.csv")
DATE_FROM     = "2026-03-15"
DATE_TO       = "2026-04-15"
SLUG          = "2026-03-16_2026-04-15"

RESULTS_DIR = Path("pipeline/results")
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COLORS = {
    "tree": "#e07b39",
    "lstm": "#3a7ebf",
    "llm":  "#48a868",
    "rnn":  "#9b59b6",
}
METRIC_SPECS = [
    ("mae",             "MAE"),
    ("rmse",            "RMSE"),
    ("directional_acc", "Dir Acc"),
    ("mape_pct",        "MAPE %"),
]

DEVICE = torch.device("cpu")

# ---------------------------------------------------------------------------
# 1. Load model
# ---------------------------------------------------------------------------
with open(RNN_MODEL_PKL, "rb") as f:
    payload = pickle.load(f)

cfg         = payload["config"]
scaler      = payload["scaler"]
state_dict  = payload["model_state_dict"]
feature_cols = list(cfg["feature_cols"])
SEQ_LEN     = cfg["sequence_length"]
N_FEAT      = len(feature_cols)

print(f"Model loaded | seq_len={SEQ_LEN} | features={N_FEAT}")
print(f"Scaler type: {type(scaler)}")

# Re-instantiate RNN architecture from saved config
class RNNModel(nn.Module):
    def __init__(self, n_features: int, rnn_units: int, num_layers: int,
                 dense_units: int, dropout: float):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=n_features,
            hidden_size=rnn_units,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity="tanh",
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1  = nn.Linear(rnn_units, dense_units)
        self.relu = nn.ReLU()
        self.fc2  = nn.Linear(dense_units, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        last = out[:, -1, :]
        return self.fc2(self.relu(self.fc1(self.dropout(last)))).squeeze(-1)

model = RNNModel(
    n_features  = N_FEAT,
    rnn_units   = cfg["rnn_units"],
    num_layers  = cfg["num_layers"],
    dense_units = cfg["dense_units"],
    dropout     = cfg["dropout"],
).to(DEVICE)
model.load_state_dict(state_dict)
model.eval()
print("RNN model ready.")

# ---------------------------------------------------------------------------
# 2. Load price history from DB (need history before DATE_FROM for features)
# ---------------------------------------------------------------------------
tickers = pd.read_csv(TICKERS_CSV)["ticker"].tolist()
print(f"Sampled tickers: {len(tickers)}")

placeholders = ",".join("?" * len(tickers))
with sqlite3.connect(DB_PATH) as conn:
    raw = pd.read_sql_query(
        f"""
        SELECT ticker, date, open, high, low, close, volume
        FROM prices
        WHERE ticker IN ({placeholders})
          AND date <= ?
        ORDER BY ticker, date
        """,
        conn,
        params=tickers + [DATE_TO],
    )
raw["date"] = pd.to_datetime(raw["date"])
for c in ["open", "high", "low", "close"]:
    raw[c] = pd.to_numeric(raw[c], errors="coerce").astype("float64")
raw["volume"] = pd.to_numeric(raw["volume"], errors="coerce").fillna(0).astype("float64")
print(f"Loaded {len(raw):,} rows for {raw['ticker'].nunique()} tickers from DB")

# ---------------------------------------------------------------------------
# 3. Feature engineering (same as train_rnn_colab.ipynb)
# ---------------------------------------------------------------------------
def compute_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    close  = g["close"].astype("float64")
    high   = g["high"].astype("float64")
    low    = g["low"].astype("float64")
    open_  = g["open"].astype("float64")
    volume = g["volume"].astype("float64").clip(lower=1.0)
    lc = np.log(close)
    g["log_ret_1"]  = lc.diff(1)
    g["log_ret_3"]  = lc.diff(3)
    g["log_ret_5"]  = lc.diff(5)
    g["log_ret_10"] = lc.diff(10)
    g["log_ret_20"] = lc.diff(20)
    g["hl_range"]   = (high - low) / close
    g["oc_change"]  = (close - open_) / open_.clip(lower=1e-6)
    ma5  = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    g["close_vs_ma_5"]  = close / ma5.clip(lower=1e-6)  - 1.0
    g["close_vs_ma_10"] = close / ma10.clip(lower=1e-6) - 1.0
    g["close_vs_ma_20"] = close / ma20.clip(lower=1e-6) - 1.0
    g["close_vs_ma_50"] = close / ma50.clip(lower=1e-6) - 1.0
    g["ma_5_vs_ma_20"]  = ma5 / ma20.clip(lower=1e-6)   - 1.0
    g["rolling_vol_5"]  = g["log_ret_1"].rolling(5).std()
    g["rolling_vol_20"] = g["log_ret_1"].rolling(20).std()
    lv = np.log1p(volume)
    g["log_vol_chg_1"] = lv.diff(1)
    vm5  = volume.rolling(5).mean()
    vm20 = volume.rolling(20).mean()
    g["vol_vs_ma_5"]  = volume / vm5.clip(lower=1.0)  - 1.0
    g["vol_vs_ma_20"] = volume / vm20.clip(lower=1.0) - 1.0
    delta = close.diff()
    gain  = delta.clip(lower=0).ewm(alpha=1.0/14, min_periods=14).mean()
    loss  = (-delta.clip(upper=0)).ewm(alpha=1.0/14, min_periods=14).mean()
    g["rsi_14"] = (100.0 - 100.0 / (1.0 + gain / (loss + 1e-8))) / 100.0 - 0.5
    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    g["macd_norm"] = (ema12 - ema26) / close.clip(lower=1e-6)
    bbm = close.rolling(20).mean()
    bbs = close.rolling(20).std()
    g["bb_pos"] = (close - (bbm - 2*bbs)) / (4*bbs).clip(lower=1e-6) - 0.5
    prev_c = close.shift(1)
    tr = pd.concat([high-low,(high-prev_c).abs(),(low-prev_c).abs()], axis=1).max(axis=1)
    g["atr_14_norm"] = tr.ewm(alpha=1.0/14, min_periods=14).mean() / close.clip(lower=1e-6)
    return g.replace([np.inf, -np.inf], np.nan)

# Scaler mean/scale
mean_arr  = scaler.mean_.astype(np.float32)
scale_arr = scaler.scale_.astype(np.float32)
scale_arr[scale_arr == 0.0] = 1.0

# ---------------------------------------------------------------------------
# 4. Inference: for each ticker, build sequences and predict
# ---------------------------------------------------------------------------
target_dates = pd.date_range(DATE_FROM, DATE_TO, freq="B")  # business days

all_rows = []
for ticker, grp in raw.groupby("ticker", sort=False):
    feat = compute_features(grp)
    feat = feat.reset_index(drop=True)
    feat_vals = feat[feature_cols].to_numpy(dtype=np.float32)
    closes    = feat["close"].to_numpy(dtype=np.float64)
    dates     = feat["date"].to_numpy("datetime64[ns]")

    date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}

    for pred_date in target_dates:
        if pred_date not in date_to_idx:
            continue
        i = date_to_idx[pred_date]
        if i < SEQ_LEN:
            continue
        seq = feat_vals[i - SEQ_LEN : i]          # [SEQ_LEN, N_FEAT]
        if np.isnan(seq).any():
            continue
        actual_close = closes[i]
        prev_close   = closes[i - 1]
        if np.isnan(actual_close) or np.isnan(prev_close) or prev_close <= 0:
            continue

        # normalise
        seq_norm = (seq - mean_arr) / scale_arr
        x = torch.from_numpy(seq_norm).unsqueeze(0).to(DEVICE)  # [1, SEQ_LEN, N_FEAT]
        with torch.no_grad():
            log_ret_pred = float(model(x).item())

        pred_close = prev_close * np.exp(np.clip(log_ret_pred, -2.0, 2.0))
        all_rows.append({
            "date":         pred_date,
            "ticker":       ticker,
            "actual_close": actual_close,
            "prev_close":   prev_close,
            "pred_close":   pred_close,
            "directional_correct": float(
                np.sign(pred_close - prev_close) == np.sign(actual_close - prev_close)
            ),
        })

inf_df = pd.DataFrame(all_rows)
print(f"Inference rows : {len(inf_df):,}  |  tickers : {inf_df['ticker'].nunique()}")
if inf_df.empty:
    raise RuntimeError("No inference results produced.")

rows_path = RESULTS_DIR / f"sampled_rnn_{SLUG}_rows.csv"
row_export_df = inf_df.rename(columns={"date": "pred_date"}).copy()
row_export_df.insert(2, "model_name", "rnn")
row_export_df["abs_error"] = (row_export_df["pred_close"] - row_export_df["actual_close"]).abs()
row_export_df.to_csv(rows_path, index=False)
print(f"Saved {rows_path}")

# ---------------------------------------------------------------------------
# 5. Daily summary
# ---------------------------------------------------------------------------
rows = []
for pred_date, g in inf_df.groupby("date", sort=True):
    y_true = g["actual_close"].to_numpy(float)
    y_pred = g["pred_close"].to_numpy(float)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    da   = float(np.mean(g["directional_correct"].to_numpy(float)))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100.0)
    rows.append({
        "pred_date":       pd.Timestamp(pred_date).strftime("%Y-%m-%d"),
        "mae":             mae,
        "rmse":            rmse,
        "directional_acc": da,
        "mape_pct":        mape,
        "n_predictions":   len(g),
    })

daily_df = pd.DataFrame(rows)
daily_path = RESULTS_DIR / f"sampled_rnn_{SLUG}_daily.csv"
daily_df.to_csv(daily_path, index=False)
print(f"Saved {daily_path}")

# ---------------------------------------------------------------------------
# 6. Append RNN row to summary CSV
# ---------------------------------------------------------------------------
summary_row = {
    "model_name":           "rnn",
    "mean_mae":             float(daily_df["mae"].mean()),
    "mean_rmse":            float(daily_df["rmse"].mean()),
    "mean_directional_acc": float(daily_df["directional_acc"].mean()),
    "mean_mape_pct":        float(daily_df["mape_pct"].mean()),
}
summary_path = RESULTS_DIR / f"sampled_summary_{SLUG}.csv"
summary_df   = pd.read_csv(summary_path)
summary_df   = summary_df[summary_df["model_name"] != "rnn"]   # idempotent
summary_df   = pd.concat([summary_df, pd.DataFrame([summary_row])], ignore_index=True)
summary_df.to_csv(summary_path, index=False)
print(f"Updated {summary_path}")
print("\nSummary table:")
print(summary_df.to_string(index=False, float_format="%.6f"))

# ---------------------------------------------------------------------------
# 7. Individual RNN 4-panel plot
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
axes = axes.flatten()
x = pd.to_datetime(daily_df["pred_date"])
for ax, (col, title) in zip(axes, METRIC_SPECS):
    ax.plot(x, daily_df[col], marker="o", linewidth=1.8, markersize=4,
            color=COLORS["rnn"])
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
fig.suptitle("RNN Daily Metrics", fontsize=15)
rnn_plot = PLOTS_DIR / f"sampled_rnn_{SLUG}_metrics.png"
fig.savefig(rnn_plot, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved {rnn_plot}")

# ---------------------------------------------------------------------------
# 8. Overlapping comparison plot (all 4 models)
# ---------------------------------------------------------------------------
model_csvs: dict[str, Path | pd.DataFrame] = {
    "tree": RESULTS_DIR / f"sampled_tree_{SLUG}_daily.csv",
    "lstm": RESULTS_DIR / f"sampled_lstm_{SLUG}_daily.csv",
    "llm":  RESULTS_DIR / f"sampled_llm_{SLUG}_daily.csv",
    "rnn":  daily_df,
}
models_data: dict[str, pd.DataFrame] = {}
for name, src in model_csvs.items():
    if isinstance(src, pd.DataFrame):
        models_data[name] = src
    elif src.exists():
        models_data[name] = pd.read_csv(src)
    else:
        print(f"  WARNING: {src} not found, skipping {name}")

fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
axes = axes.flatten()
for ax, (col, title) in zip(axes, METRIC_SPECS):
    for model_name, mdf in models_data.items():
        xvals = pd.to_datetime(mdf["pred_date"])
        ax.plot(xvals, mdf[col],
                marker="o", linewidth=1.5, markersize=3,
                label=model_name.upper(),
                color=COLORS.get(model_name))
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

fig.suptitle("Model Comparison — Daily Metrics (2026-03-16 to 2026-04-15)", fontsize=14)
compare_plot = PLOTS_DIR / f"comparison_{SLUG}_metrics.png"
fig.savefig(compare_plot, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"Saved {compare_plot}")
print("\nDone!")
