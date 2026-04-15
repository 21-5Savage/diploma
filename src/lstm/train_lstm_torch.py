"""
PyTorch LSTM for next-day stock return prediction.

Target  : log(close_t+1 / close_t)  — log return
Metrics : R², RMSE, MAE, Directional Accuracy
          DA = sign(pred_return) == sign(actual_return)

Run:
    python -m src.lstm.train_lstm_torch
"""

import gc
import json
import os
import pickle
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

import datetime
import random
import string

today = datetime.date.today()
month = today.month
day = today.day
tag = "".join(random.choices(string.ascii_lowercase, k=3))


@dataclass
class Config:
    db_path: str = "dataset/stock_prices_20y.db"
    table_name: str = "prices"

    n_splits: int = 5
    test_size: float = 0.2
    target_horizon: int = 1
    # "return"  →  log(close_{t+1}/close_t)  — recommended for proper R²/DA
    predict_target: str = "return"
    random_state: int = 42

    sequence_length: int = 30
    batch_size: int = 2048
    epochs: int = 60
    patience: int = 10

    lstm_units: int = 128
    num_layers: int = 2
    dense_units: int = 64
    dropout: float = 0.3
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4

    model_output_path: str = f"artifacts/{month}-{day}-{tag}-lstm_torch.pkl"
    config_output_path: str = f"artifacts/{month}-{day}-{tag}-lstm_torch_config.json"

    max_tickers: int = 0  # 0 = use all tickers

    feature_cols: tuple = (
        # Returns / momentum
        "log_ret_1",
        "log_ret_3",
        "log_ret_5",
        "log_ret_10",
        "log_ret_20",
        # Intraday structure
        "hl_range",
        "oc_change",
        # Moving-average relative position
        "close_vs_ma_5",
        "close_vs_ma_10",
        "close_vs_ma_20",
        "close_vs_ma_50",
        "ma_5_vs_ma_20",
        # Volatility (rolling std of log-returns)
        "rolling_vol_5",
        "rolling_vol_20",
        # Volume
        "log_vol_chg_1",
        "vol_vs_ma_5",
        "vol_vs_ma_20",
        # RSI-14
        "rsi_14",
        # MACD (fast EMA - slow EMA, normalised by close)
        "macd_norm",
        # Bollinger band position
        "bb_pos",
        # ATR-14 as fraction of close
        "atr_14_norm",
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def load_prices_from_sqlite(db_path: str, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT ticker, date, open, high, low, close, adj_close, volume
        FROM {table_name}
        ORDER BY ticker, date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    float_cols = ["open", "high", "low", "close", "adj_close"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("float32")
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = gain / (loss + 1e-8)
    return 100.0 - 100.0 / (1.0 + rs)


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()


def make_sequence_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute rich feature set per ticker; returns concatenated DataFrame."""
    out = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"].astype("float64")
        high = g["high"].astype("float64")
        low = g["low"].astype("float64")
        open_ = g["open"].astype("float64")
        volume = g["volume"].astype("float64").clip(lower=1.0)

        # Log returns
        log_close = np.log(close)
        g["log_ret_1"] = log_close.diff(1)
        g["log_ret_3"] = log_close.diff(3)
        g["log_ret_5"] = log_close.diff(5)
        g["log_ret_10"] = log_close.diff(10)
        g["log_ret_20"] = log_close.diff(20)

        # Intraday
        g["hl_range"] = (high - low) / close
        g["oc_change"] = (close - open_) / open_.clip(lower=1e-6)

        # MA relative
        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma50 = close.rolling(50).mean()
        g["close_vs_ma_5"] = close / ma5.clip(lower=1e-6) - 1.0
        g["close_vs_ma_10"] = close / ma10.clip(lower=1e-6) - 1.0
        g["close_vs_ma_20"] = close / ma20.clip(lower=1e-6) - 1.0
        g["close_vs_ma_50"] = close / ma50.clip(lower=1e-6) - 1.0
        g["ma_5_vs_ma_20"] = ma5 / ma20.clip(lower=1e-6) - 1.0

        # Volatility
        g["rolling_vol_5"] = g["log_ret_1"].rolling(5).std()
        g["rolling_vol_20"] = g["log_ret_1"].rolling(20).std()

        # Volume
        log_vol = np.log1p(volume)
        g["log_vol_chg_1"] = log_vol.diff(1)
        vol_ma5 = volume.rolling(5).mean()
        vol_ma20 = volume.rolling(20).mean()
        g["vol_vs_ma_5"] = volume / vol_ma5.clip(lower=1.0) - 1.0
        g["vol_vs_ma_20"] = volume / vol_ma20.clip(lower=1.0) - 1.0

        # RSI-14
        g["rsi_14"] = _compute_rsi(close, 14) / 100.0 - 0.5  # centered at 0

        # MACD (EMA12 - EMA26), normalised by close
        ema12 = close.ewm(span=12, min_periods=12).mean()
        ema26 = close.ewm(span=26, min_periods=26).mean()
        g["macd_norm"] = (ema12 - ema26) / close.clip(lower=1e-6)

        # Bollinger bands position: (close - lower) / (upper - lower)
        bb_mean = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = bb_mean + 2 * bb_std
        bb_lower = bb_mean - 2 * bb_std
        bb_width = (bb_upper - bb_lower).clip(lower=1e-6)
        g["bb_pos"] = (close - bb_lower) / bb_width - 0.5  # centred

        # ATR-14 as fraction of close
        g["atr_14_norm"] = _compute_atr(high, low, close, 14) / close.clip(lower=1e-6)

        out.append(g)

    feat_df = pd.concat(out, ignore_index=True)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan)
    return feat_df


def add_target(df: pd.DataFrame, horizon: int, predict_target: str) -> pd.DataFrame:
    """
    predict_target='return'  → log(close_{t+horizon} / close_t)
    predict_target='price'   → close_{t+horizon}  (absolute)
    """
    out = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"]
        if predict_target == "return":
            g["target"] = np.log(close.shift(-horizon) / close).astype("float32")
        elif predict_target == "price":
            g["target"] = close.shift(-horizon).astype("float32")
        else:
            raise ValueError("predict_target must be 'return' or 'price'")
        out.append(g)
    return pd.concat(out, ignore_index=True)


def split_train_test_per_ticker(
    df: pd.DataFrame, test_size: float, embargo: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    for _, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        split_idx = int(n * (1 - test_size))
        train_end = split_idx - embargo
        if split_idx <= 0 or split_idx >= n or train_end <= 0:
            continue
        train_parts.append(grp.iloc[:train_end].copy())
        test_parts.append(grp.iloc[split_idx:].copy())
    if not train_parts or not test_parts:
        raise ValueError("Empty train/test split.")
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def build_sequences(
    df: pd.DataFrame, feature_cols: List[str], sequence_length: int
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_list, y_list, meta_list = [], [], []
    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        if len(g) < sequence_length:
            continue
        feat = g[feature_cols].to_numpy(dtype=np.float32)
        target = g["target"].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        closes = g["close"].to_numpy(dtype=np.float32)
        for i in range(sequence_length - 1, len(g)):
            if np.isnan(target[i]):
                continue
            x_seq = feat[i - sequence_length + 1 : i + 1]
            if np.isnan(x_seq).any():
                continue
            X_list.append(x_seq)
            y_list.append(target[i])
            meta_list.append((ticker, dates[i], closes[i]))
    if not X_list:
        raise ValueError("No sequences created.")
    return (
        np.asarray(X_list, dtype=np.float32),
        np.asarray(y_list, dtype=np.float32),
        pd.DataFrame(meta_list, columns=["ticker", "date", "close"]),
    )


def build_grouped_time_folds(
    meta: pd.DataFrame, n_splits: int, embargo: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    meta = meta.sort_values(["ticker", "date"]).reset_index(drop=True)
    folds = []
    for fold_num in range(n_splits):
        tr_idx_all, va_idx_all = [], []
        for _, grp in meta.groupby("ticker", sort=False):
            idx = grp.index.to_numpy()
            n = len(idx)
            if n < (n_splits + 1) * max(embargo, 1) + 5:
                continue
            boundaries = np.linspace(0, n, n_splits + 2, dtype=int)
            va_start = boundaries[fold_num + 1]
            va_end = boundaries[fold_num + 2]
            purged_train_end = va_start - embargo
            if purged_train_end <= 0 or va_end <= va_start:
                continue
            tr_idx_all.extend(idx[:purged_train_end].tolist())
            va_idx_all.extend(idx[va_start:va_end].tolist())
        if tr_idx_all and va_idx_all:
            folds.append((np.asarray(tr_idx_all, dtype=np.int64), np.asarray(va_idx_all, dtype=np.int64)))
    if not folds:
        raise ValueError("No CV folds built.")
    return folds


class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray, scaler: StandardScaler):
        self.X = X
        self.y = y
        self.indices = indices
        self.mean = scaler.mean_.astype(np.float32)
        self.scale = scaler.scale_.astype(np.float32)
        self.scale[self.scale == 0.0] = 1.0

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.X[i].copy()
        x = (x - self.mean) / self.scale
        return torch.from_numpy(x), torch.tensor(self.y[i], dtype=torch.float32)


class LSTMModel(nn.Module):
    def __init__(self, n_features: int, cfg: Config):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=cfg.lstm_units,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc1 = nn.Linear(cfg.lstm_units, cfg.dense_units)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.dense_units, cfg.dense_units // 2)
        self.fc3 = nn.Linear(cfg.dense_units // 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        out = self.dropout(last)
        out = self.act(self.fc1(out))
        out = self.act(self.fc2(out))
        return self.fc3(out).squeeze(-1)


def fit_scaler(X_all: np.ndarray, indices: np.ndarray, chunk_rows: int = 50000) -> StandardScaler:
    scaler = StandardScaler()
    n_features = X_all.shape[2]
    for start in range(0, len(indices), chunk_rows):
        batch_idx = indices[start : start + chunk_rows]
        x_chunk = X_all[batch_idx].reshape(-1, n_features)
        scaler.partial_fit(x_chunk)
    return scaler


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    For return predictions: DA = sign(pred)==sign(actual).
    For price predictions: DA is meaningless here — caller must pre-convert.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "directional_accuracy": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        # For log-return target: sign(pred)==sign(actual) is the correct DA
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(yb)
        n += len(yb)
    return total_loss / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds = []
    for xb, _ in loader:
        xb = xb.to(device)
        preds.append(model(xb).cpu().numpy())
    return np.concatenate(preds)


def train_model(model, train_loader, val_loader, cfg, device):
    criterion = nn.HuberLoss(delta=1.0)  # more robust than MSE for financial returns
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.learning_rate * 0.05
    )

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        scheduler.step()

        # Validation
        model.eval()
        val_loss = 0.0
        n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_loss += criterion(pred, yb).item() * len(yb)
                n += len(yb)
        val_loss /= n

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 5 == 0 or patience_counter == 0:
            print(f"  Epoch {epoch:3d} | train={train_loss:.6f} | val={val_loss:.6f} | lr={scheduler.get_last_lr()[0]:.2e}", flush=True)

        if patience_counter >= cfg.patience:
            print(f"  Early stop at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def main():
    cfg = Config()
    set_seed(cfg.random_state)
    device = torch.device("cpu")

    print("Loading data...")
    raw_df = load_prices_from_sqlite(cfg.db_path, cfg.table_name)

    # Subsample tickers for CPU training speed
    all_tickers = list(raw_df["ticker"].unique())
    if cfg.max_tickers > 0 and len(all_tickers) > cfg.max_tickers:
        rng = np.random.default_rng(cfg.random_state)
        all_tickers = rng.choice(all_tickers, cfg.max_tickers, replace=False).tolist()
        raw_df = raw_df[raw_df["ticker"].isin(all_tickers)].reset_index(drop=True)
        print(f"Subsampled to {cfg.max_tickers} tickers")
    else:
        print(f"Using all {len(all_tickers)} tickers")

    print("Building features...")
    feat_df = make_sequence_features(raw_df)

    print("Splitting train/test per ticker with embargo...")
    train_feat_df, test_feat_df = split_train_test_per_ticker(feat_df, cfg.test_size, cfg.target_horizon)

    print("Creating targets inside each split...")
    train_df = add_target(train_feat_df, cfg.target_horizon, cfg.predict_target)
    test_df = add_target(test_feat_df, cfg.target_horizon, cfg.predict_target)

    print("Building train sequences...")
    X_train_raw, y_train, meta_train = build_sequences(train_df, list(cfg.feature_cols), cfg.sequence_length)
    print(f"Train sequences: {len(X_train_raw):,} | Tickers: {meta_train['ticker'].nunique()}")

    print("Building test sequences...")
    X_test_raw, y_test, meta_test = build_sequences(test_df, list(cfg.feature_cols), cfg.sequence_length)
    print(f"Test sequences : {len(X_test_raw):,} | Tickers: {meta_test['ticker'].nunique()}")

    print("Building grouped time-series CV folds...")
    folds = build_grouped_time_folds(meta_train, cfg.n_splits, cfg.target_horizon)
    print(f"CV folds: {len(folds)}")

    n_features = len(cfg.feature_cols)
    cv_results = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_i}/{len(folds)} ===")
        print(f"  Train: {len(tr_idx):,} | Valid: {len(va_idx):,}")

        scaler = fit_scaler(X_train_raw, tr_idx)
        train_ds = SequenceDataset(X_train_raw, y_train, tr_idx, scaler)
        val_ds = SequenceDataset(X_train_raw, y_train, va_idx, scaler)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

        model = LSTMModel(n_features, cfg).to(device)
        model = train_model(model, train_loader, val_loader, cfg, device)

        y_va_pred = predict(model, val_loader, device)
        y_va_true = y_train[va_idx]
        metrics = regression_metrics(y_va_true, y_va_pred)
        cv_results.append(metrics)
        print(f"  Fold {fold_i} | RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} R2={metrics['r2']:.6f} DA={metrics['directional_accuracy']:.4f}")

        del model, train_ds, val_ds, train_loader, val_loader, scaler
        gc.collect()

    print("\n=== CV Summary ===")
    cv_df = pd.DataFrame(cv_results)
    print(cv_df.mean(numeric_only=True).to_string())

    print("\nTraining final model on all training data...")
    all_idx = np.arange(len(X_train_raw), dtype=np.int64)
    scaler = fit_scaler(X_train_raw, all_idx)

    # Use last 10% of training as validation for early stopping
    n_train = len(all_idx)
    split = int(n_train * 0.9)
    final_tr_idx = all_idx[:split]
    final_va_idx = all_idx[split:]

    train_ds = SequenceDataset(X_train_raw, y_train, final_tr_idx, scaler)
    val_ds = SequenceDataset(X_train_raw, y_train, final_va_idx, scaler)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    final_model = LSTMModel(n_features, cfg).to(device)
    final_model = train_model(final_model, train_loader, val_loader, cfg, device)

    print("\nEvaluating on held-out test set...")
    test_idx = np.arange(len(X_test_raw), dtype=np.int64)
    test_ds = SequenceDataset(X_test_raw, y_test, test_idx, scaler)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    y_test_pred = predict(final_model, test_loader, device)
    test_metrics = regression_metrics(y_test, y_test_pred)
    print(f"Test | RMSE={test_metrics['rmse']:.6f} MAE={test_metrics['mae']:.6f} R2={test_metrics['r2']:.6f} DA={test_metrics['directional_accuracy']:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    payload = {
        "model_state_dict": final_model.state_dict(),
        "scaler": scaler,
        "config": asdict(cfg),
        "cv_results": cv_results,
        "test_metrics": test_metrics,
    }
    with open(cfg.model_output_path, "wb") as f:
        pickle.dump(payload, f)

    config_dict = asdict(cfg)
    config_dict["feature_cols"] = list(config_dict["feature_cols"])
    with open(cfg.config_output_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Save predictions
    out = meta_test.copy()
    out["y_true"] = y_test
    out["y_pred"] = y_test_pred
    pred_path = f"results/{month}-{day}-{tag}-lstm_torch_predictions.csv"
    out.to_csv(pred_path, index=False)

    result_txt = f"""PyTorch LSTM Results (target={cfg.predict_target}):
RMSE={test_metrics['rmse']:.6f}
MAE={test_metrics['mae']:.6f}
R2={test_metrics['r2']:.6f}
DA={test_metrics['directional_accuracy']:.4f}

CV Mean R2={cv_df['r2'].mean():.6f}
CV Mean DA={cv_df['directional_accuracy'].mean():.4f}
"""
    with open(f"{pred_path}.txt", "w") as f:
        f.write(result_txt)

    print(f"\nSaved model to: {cfg.model_output_path}")
    print(f"Saved predictions to: {pred_path}")


if __name__ == "__main__":
    main()
