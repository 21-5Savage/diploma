"""
    python -m pipeline.predict
    python -m pipeline.predict --models lstm rnn tree prophet
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


def get_available_prediction_dates(
    raw_df: pd.DataFrame,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[pd.Timestamp]:
    dates = pd.Series(pd.to_datetime(raw_df["date"]).sort_values().unique())
    if date_from is not None:
        dates = dates[dates >= pd.Timestamp(date_from)]
    if date_to is not None:
        dates = dates[dates <= pd.Timestamp(date_to)]
    return list(pd.to_datetime(dates))


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


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    return (100.0 - 100.0 / (1.0 + gain / (loss + 1e-8))) / 100.0 - 0.5


def make_legacy_lstm_features_df(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"].astype("float64")
        high = g["high"].astype("float64")
        low = g["low"].astype("float64")
        open_ = g["open"].astype("float64")
        volume = g["volume"].astype("float64").clip(lower=1.0)

        log_c = np.log(close)
        g["ret_1"] = log_c.diff(1)
        g["ret_5"] = log_c.diff(5)
        g["hl_range"] = (high - low) / close
        g["oc_change"] = (close - open_) / open_.clip(lower=1e-6)
        g["close_vs_ma_5"] = close / close.rolling(5).mean().clip(lower=1e-6) - 1.0
        g["close_vs_ma_10"] = close / close.rolling(10).mean().clip(lower=1e-6) - 1.0
        log_vol = np.log1p(volume)
        g["vol_chg_1"] = log_vol.diff(1)
        g["vol_vs_ma_5"] = volume / volume.rolling(5).mean().clip(lower=1.0) - 1.0
        out.append(g)
    if not out:
        return df
    return pd.concat(out, ignore_index=True).replace([np.inf, -np.inf], np.nan)


def make_tree_features_df(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"].astype("float64")
        high = g["high"].astype("float64")
        low = g["low"].astype("float64")
        open_ = g["open"].astype("float64")
        volume = g["volume"].astype("float64").clip(lower=1.0)

        log_c = np.log(close)
        g["log_ret_1"] = log_c.diff(1)
        g["log_ret_3"] = log_c.diff(3)
        g["log_ret_5"] = log_c.diff(5)
        g["log_ret_10"] = log_c.diff(10)
        g["log_ret_20"] = log_c.diff(20)
        g["log_ret_60"] = log_c.diff(60)
        g["hl_range"] = (high - low) / close
        g["oc_change"] = (close - open_) / open_.clip(lower=1e-6)

        for d in [5, 10, 20, 50, 200]:
            ma = close.rolling(d).mean()
            g[f"close_vs_ma_{d}"] = close / ma.clip(lower=1e-6) - 1.0

        g["ma_5_vs_20"] = close.rolling(5).mean() / close.rolling(20).mean().clip(lower=1e-6) - 1.0
        g["ma_20_vs_50"] = close.rolling(20).mean() / close.rolling(50).mean().clip(lower=1e-6) - 1.0

        for d in [5, 10, 20]:
            g[f"rolling_vol_{d}"] = g["log_ret_1"].rolling(d).std()
        g["vol_of_vol_20"] = g["rolling_vol_5"].rolling(20).std()

        log_vol = np.log1p(volume)
        g["log_vol_chg_1"] = log_vol.diff(1)
        g["log_vol_chg_5"] = log_vol.diff(5)
        for d in [5, 20]:
            g[f"vol_vs_ma_{d}"] = volume / volume.rolling(d).mean().clip(lower=1.0) - 1.0

        g["rsi_14"] = _compute_rsi(close, 14)
        g["rsi_7"] = _compute_rsi(close, 7)

        ema12 = close.ewm(span=12, min_periods=12).mean()
        ema26 = close.ewm(span=26, min_periods=26).mean()
        macd = ema12 - ema26
        g["macd_norm"] = macd / close.clip(lower=1e-6)
        g["macd_signal_norm"] = macd.ewm(span=9, min_periods=9).mean() / close.clip(lower=1e-6)

        bb_mean = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        g["bb_pos"] = (close - (bb_mean - 2 * bb_std)) / (4 * bb_std).clip(lower=1e-6) - 0.5
        g["bb_width"] = 4 * bb_std / bb_mean.clip(lower=1e-6)

        prev_c = close.shift(1)
        tr = pd.concat([high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1).max(axis=1)
        g["atr_14_norm"] = tr.ewm(alpha=1.0 / 14, min_periods=14).mean() / close.clip(lower=1e-6)
        g["skew_20"] = g["log_ret_1"].rolling(20).skew()
        g["kurt_20"] = g["log_ret_1"].rolling(20).kurt()
        out.append(g)

    if not out:
        return df
    return pd.concat(out, ignore_index=True).replace([np.inf, -np.inf], np.nan)


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


class _LegacyLSTMModel(torch.nn.Module):
    def __init__(self, n_features, units, layers, dense, dropout):
        super().__init__()
        self.lstm = torch.nn.LSTM(
            n_features, units, layers, batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.drop = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(units, dense)
        self.act = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(dense, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        h = self.drop(out[:, -1, :])
        return self.fc2(self.act(self.fc1(h))).squeeze(-1)


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

    state_dict = payload["model_state_dict"]
    if model_class is _LSTMModel and "fc3.weight" not in state_dict:
        model = _LegacyLSTMModel(n_features, units, layers, dense, dropout)
    else:
        model = model_class(n_features, units, layers, dense, dropout)
    model.load_state_dict(state_dict)
    model.eval()

    records = []
    for ticker, g in feat_df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < seq_len:
            continue
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        x_seq = feat[-seq_len:]  # (seq_len, n_features)
        if np.isnan(x_seq).any():
            continue
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


def choose_feature_frame(
    model_name: str,
    payload: dict,
    default_feat_df: pd.DataFrame,
    tree_feat_df: pd.DataFrame | None,
    legacy_lstm_feat_df: pd.DataFrame | None,
) -> pd.DataFrame:
    if model_name == "xgb":
        return tree_feat_df if tree_feat_df is not None else default_feat_df
    if model_name == "tree":
        feature_cols = payload.get("feature_cols", [])
        if feature_cols and all(col in default_feat_df.columns for col in feature_cols):
            return default_feat_df
        return tree_feat_df if tree_feat_df is not None else default_feat_df
    if model_name == "lstm":
        cfg = payload.get("config", {})
        feature_cols = list(cfg.get("feature_cols", FEATURE_COLS))
        if feature_cols and all(col in default_feat_df.columns for col in feature_cols):
            return default_feat_df
        return legacy_lstm_feat_df if legacy_lstm_feat_df is not None else default_feat_df
    return default_feat_df


# ─────────────────── Main ──────────────────────────────────────────────────

def run_predictions(
    models_to_run: list[str],
    date_from: str | None = None,
    date_to: str | None = None,
    tickers: list[str] | None = None,
) -> None:
    conn = sqlite3.connect(DB_PATH, timeout=60)
    init_predictions_table(conn)

    print("Loading price data from pipeline DB...")
    raw_df = load_prices(conn, tickers=tickers)
    if raw_df.empty:
        print("No price data found. Run fetch_data first.")
        conn.close()
        return

    print(f"Computing features for {raw_df['ticker'].nunique()} tickers...")
    feat_df = make_features_df(raw_df)
    available_dates = get_available_prediction_dates(raw_df, date_from, date_to)
    if not available_dates:
        print("No available prediction dates in the requested range.")
        conn.close()
        return

    if date_from is None and date_to is None:
        available_dates = [max(available_dates)]
    else:
        print(
            f"Historical prediction window: {available_dates[0].strftime('%Y-%m-%d')} "
            f"-> {available_dates[-1].strftime('%Y-%m-%d')} "
            f"({len(available_dates)} trading dates)"
        )

    model_registry = {
        "lstm": ("lstm_torch", _LSTMModel, predict_with_seq_model),
        "rnn": ("rnn_torch", _RNNModel, predict_with_seq_model),
        "tree": ("decision_tree", None, predict_with_tabular_model),
        "xgb": ("xgb_tree", None, predict_with_tabular_model),
        "prophet": ("prophet", None, predict_with_prophet),
    }

    model_payloads: dict[str, tuple[dict, object | None, object]] = {}
    need_tree_features = False
    need_legacy_lstm_features = False
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
        model_payloads[name] = (payload, model_class, predictor)
        if name in ("xgb", "tree"):
            need_tree_features = True
        if name == "lstm":
            state_dict = payload.get("model_state_dict", {})
            cfg = payload.get("config", {})
            feature_cols = list(cfg.get("feature_cols", []))
            if "fc3.weight" not in state_dict or (feature_cols and not all(col in feat_df.columns for col in feature_cols)):
                need_legacy_lstm_features = True

    if not model_payloads:
        print("No usable model artifacts were found.")
        conn.close()
        return

    tree_feat_df = make_tree_features_df(raw_df) if need_tree_features else None
    legacy_lstm_feat_df = make_legacy_lstm_features_df(raw_df) if need_legacy_lstm_features else None

    for i, pred_date in enumerate(available_dates, start=1):
        pred_date_str = pred_date.strftime("%Y-%m-%d")
        run_at = datetime.utcnow().isoformat()
        if feat_df[feat_df["date"] < pred_date].empty:
            print(f"[{i}/{len(available_dates)}] {pred_date_str}: no prior history, skipping")
            continue
        print(f"\n[{i}/{len(available_dates)}] Prediction date: {pred_date_str}")

        for name, (payload, model_class, predictor) in model_payloads.items():
            model_feat_df = choose_feature_frame(
                name, payload, feat_df, tree_feat_df, legacy_lstm_feat_df
            )
            hist_df = model_feat_df[model_feat_df["date"] < pred_date].copy()
            if hist_df.empty:
                print(f"  [{name}] No prior history available")
                continue
            try:
                if name in ("lstm", "rnn"):
                    preds_df = predictor(hist_df, payload, model_class)
                elif name in ("tree", "xgb"):
                    preds_df = predictor(hist_df, payload)
                else:
                    preds_df = predictor(hist_df, payload, pred_date_str)
            except Exception as exc:
                print(f"  [{name}] Prediction failed: {exc}")
                continue

            if preds_df.empty:
                print(f"  [{name}] No predictions generated")
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
            print(f"  [{name}] Saved {len(rows)} predictions")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=["lstm", "rnn", "tree", "prophet"],
                        choices=["lstm", "rnn", "tree", "xgb", "prophet"])
    parser.add_argument("--date-from", default=None, help="Historical prediction start date (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=None, help="Historical prediction end date (YYYY-MM-DD)")
    parser.add_argument("--tickers", nargs="*", default=None, help="Optional ticker filter")
    args = parser.parse_args()
    run_predictions(args.models, date_from=args.date_from, date_to=args.date_to, tickers=args.tickers)
