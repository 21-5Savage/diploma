"""
Load trained model artifacts and predict next-day returns for all tickers
in the pipeline database.

Each model produces: ticker, date, model_name, pred_return, pred_direction (+1/-1)

Usage:
    python -m pipeline.predict
    python -m pipeline.predict --models lstm xgb
"""

import argparse
import os
import pickle
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from prophet.serialize import model_from_json

from pipeline.features import FEATURE_COLS, make_features_df

DB_PATH = os.environ.get("PIPELINE_DB", "pipeline/db/pipeline.db")
MODELS_DIR = os.environ.get("MODELS_DIR", "pipeline/models")
SEQUENCE_LENGTH = int(os.environ.get("LSTM_SEQ_LEN", "30"))


# ─────────────────── DB helpers ────────────────────────────────────────────

def init_predictions_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            run_at        TEXT NOT NULL,
            pred_date     TEXT NOT NULL,
            ticker        TEXT NOT NULL,
            model_name    TEXT NOT NULL,
            pred_return   REAL,
            pred_direction INTEGER,
            UNIQUE(pred_date, ticker, model_name)
        )
    """)
    conn.commit()


def load_prices(conn: sqlite3.Connection, tickers: list[str] | None = None) -> pd.DataFrame:
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        df = pd.read_sql_query(
            f"SELECT ticker,date,open,high,low,close,volume FROM prices WHERE ticker IN ({placeholders}) ORDER BY ticker,date",
            conn, params=tickers
        )
    else:
        df = pd.read_sql_query(
            "SELECT ticker,date,open,high,low,close,volume FROM prices ORDER BY ticker,date",
            conn
        )
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


# ─────────────────── Model loaders ─────────────────────────────────────────

def _find_latest_artifact(prefix: str, ext: str = ".pkl") -> str | None:
    """Find the most recently modified matching file in MODELS_DIR or artifacts/."""
    candidates = []
    for d in [MODELS_DIR, "artifacts"]:
        if not os.path.isdir(d):
            continue
        for fn in os.listdir(d):
            if prefix in fn and fn.endswith(ext):
                candidates.append(os.path.join(d, fn))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_pkl(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────── LSTM / RNN predictor ──────────────────────────────────

class _LSTMModel(torch.nn.Module):
    def __init__(self, n_features, units, layers, dense, dropout):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_features, units, layers, batch_first=True,
                                  dropout=dropout if layers > 1 else 0.0)
        self.drop = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(units, dense)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(dense, dense // 2)
        self.fc3 = torch.nn.Linear(dense // 2, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        return self.fc3(self.act(self.fc2(self.act(self.fc1(h))))).squeeze(-1)


class _RNNModel(torch.nn.Module):
    def __init__(self, n_features, units, layers, dense, dropout):
        super().__init__()
        self.rnn = torch.nn.RNN(n_features, units, layers, batch_first=True, nonlinearity="tanh",
                                dropout=dropout if layers > 1 else 0.0)
        self.drop = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(units, dense)
        self.act = torch.nn.GELU()
        self.fc2 = torch.nn.Linear(dense, dense // 2)
        self.fc3 = torch.nn.Linear(dense // 2, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        h = self.drop(out[:, -1, :])
        return self.fc3(self.act(self.fc2(self.act(self.fc1(h))))).squeeze(-1)


def predict_with_seq_model(feat_df: pd.DataFrame, payload: dict, model_class) -> pd.DataFrame:
    """Run LSTM or RNN on the last SEQUENCE_LENGTH rows per ticker."""
    cfg = payload["config"]
    feature_cols = list(cfg.get("feature_cols", FEATURE_COLS))
    seq_len = cfg.get("sequence_length", SEQUENCE_LENGTH)
    scaler = payload["scaler"]
    mean_ = scaler.mean_.astype(np.float32)
    scale_ = scaler.scale_.astype(np.float32)
    scale_[scale_ == 0.0] = 1.0

    n_features = len(feature_cols)
    units = cfg.get("lstm_units", cfg.get("rnn_units", 128))
    layers = cfg.get("num_layers", 2)
    dense = cfg.get("dense_units", 64)
    dropout = cfg.get("dropout", 0.3)

    model = model_class(n_features, units, layers, dense, dropout)
    model.load_state_dict(payload["model_state_dict"])
    model.eval()

    records = []
    for ticker, g in feat_df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < seq_len:
            continue
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        if np.isnan(feat).any():
            continue
        x_seq = feat[-seq_len:]  # (seq_len, n_features)
        x_seq = (x_seq - mean_) / scale_
        x_tensor = torch.from_numpy(x_seq).unsqueeze(0)  # (1, seq_len, n_features)
        with torch.no_grad():
            pred = model(x_tensor).item()
        records.append({
            "ticker": ticker,
            "pred_return": float(pred),
            "pred_direction": int(np.sign(pred)),
        })
    return pd.DataFrame(records)


def predict_with_tabular_model(feat_df: pd.DataFrame, payload: dict) -> pd.DataFrame:
    model = payload["model"]
    feature_cols = payload.get("feature_cols", FEATURE_COLS)
    records = []
    for ticker, g in feat_df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        row = g[feature_cols].iloc[-1].values.astype(np.float32)
        if np.isnan(row).any():
            continue
        pred = float(model.predict(row.reshape(1, -1))[0])
        records.append({
            "ticker": ticker,
            "pred_return": pred,
            "pred_direction": int(np.sign(pred)),
        })
    return pd.DataFrame(records)


def predict_with_prophet(feat_df: pd.DataFrame, payload: dict, pred_date: str) -> pd.DataFrame:
    """Use persisted per-ticker Prophet models to predict the next-day return."""
    models_by_ticker = payload.get("models_by_ticker", {})
    if not models_by_ticker:
        return pd.DataFrame(columns=["ticker", "pred_return", "pred_direction"])

    cfg = payload.get("config", {})
    use_log_target = bool(cfg.get("use_log_target", False))
    regressor_cols = payload.get(
        "regressor_cols",
        ["log_volume", "log_ret_1", "hl_range", "rsi_14"],
    )

    records = []
    for ticker, g in feat_df.groupby("ticker", sort=False):
        model_json = models_by_ticker.get(ticker)
        if not model_json:
            continue

        g = g.sort_values("date").reset_index(drop=True)
        if g.empty:
            continue

        last_row = g.iloc[-1]
        last_close = float(last_row["close"])
        if not np.isfinite(last_close) or last_close <= 0.0:
            continue

        last_date = pd.Timestamp(last_row["date"])
        future_date = pd.bdate_range(last_date, periods=2)[-1]
        future_df = pd.DataFrame({"ds": [future_date]})

        for col in regressor_cols:
            value = last_row[col] if col in g.columns else 0.0
            future_df[col] = [float(value) if np.isfinite(value) else 0.0]

        try:
            model = model_from_json(model_json)
            forecast = model.predict(future_df)
        except Exception:
            continue

        pred_value = float(forecast["yhat"].iloc[-1])
        if use_log_target:
            pred_value = float(np.exp(np.clip(pred_value, -100.0, 20.0)))

        pred_return = float(np.log(max(pred_value, 1e-6) / max(last_close, 1e-6)))
        records.append({
            "ticker": ticker,
            "pred_return": pred_return,
            "pred_direction": int(np.sign(pred_return)),
        })

    return pd.DataFrame(records)


# ─────────────────── Main ──────────────────────────────────────────────────

def run_predictions(models_to_run: list[str]) -> None:
    conn = sqlite3.connect(DB_PATH)
    init_predictions_table(conn)

    print("Loading price data from pipeline DB...")
    raw_df = load_prices(conn)
    if raw_df.empty:
        print("No price data found. Run fetch_data first.")
        conn.close()
        return

    print(f"Computing features for {raw_df['ticker'].nunique()} tickers...")
    feat_df = make_features_df(raw_df)

    pred_date = feat_df.groupby("ticker")["date"].max().max()
    pred_date_str = pred_date.strftime("%Y-%m-%d")
    run_at = datetime.utcnow().isoformat()
    print(f"Prediction date: {pred_date_str}")

    model_registry = {
        "lstm": ("lstm_torch", _LSTMModel, predict_with_seq_model),
        "rnn": ("rnn_torch", _RNNModel, predict_with_seq_model),
        "tree": ("decision_tree", None, predict_with_tabular_model),
        "xgb": ("xgb_tree", None, predict_with_tabular_model),
        "prophet": ("prophet", None, predict_with_prophet),
    }

    for name in models_to_run:
        if name not in model_registry:
            print(f"Unknown model: {name}, skipping")
            continue
        prefix, model_class, predictor = model_registry[name]
        artifact = _find_latest_artifact(prefix)
        if artifact is None:
            print(f"No artifact found for '{name}' (prefix='{prefix}'), skipping")
            continue
        print(f"\n[{name}] Loading {artifact}")
        try:
            payload = _load_pkl(artifact)
        except Exception as exc:
            print(f"  Failed to load: {exc}")
            continue

        try:
            if name in ("lstm", "rnn"):
                preds_df = predictor(feat_df, payload, model_class)
            elif name in ("tree", "xgb"):
                preds_df = predictor(feat_df, payload)
            else:
                preds_df = predictor(feat_df, payload, pred_date_str)
        except Exception as exc:
            print(f"  Prediction failed: {exc}")
            continue

        if preds_df.empty:
            print(f"  No predictions generated for {name}")
            continue

        preds_df["pred_date"] = pred_date_str
        preds_df["model_name"] = name
        preds_df["run_at"] = run_at

        rows = preds_df[["run_at", "pred_date", "ticker", "model_name", "pred_return", "pred_direction"]].values.tolist()
        conn.executemany(
            """INSERT INTO predictions (run_at, pred_date, ticker, model_name, pred_return, pred_direction)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(pred_date, ticker, model_name) DO UPDATE SET
                   pred_return=excluded.pred_return,
                   pred_direction=excluded.pred_direction,
                   run_at=excluded.run_at""",
            rows,
        )
        conn.commit()
        print(f"  Saved {len(rows)} predictions for {name}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["lstm", "rnn", "tree"],
                        choices=["lstm", "rnn", "tree", "xgb", "prophet"])
    args = parser.parse_args()
    run_predictions(args.models)
