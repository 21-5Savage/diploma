"""
XGBoost gradient-boosted decision tree for next-day stock return prediction.

Target  : log(close_t+1 / close_t)  — log return
Metrics : R², RMSE, MAE, Directional Accuracy
          DA = sign(pred_return) == sign(actual_return)

Features are the same rich technical-indicator set used by LSTM/RNN
but flattened into a single vector (no sequences needed for tree models).
We include rolling statistics over a lookback window as separate features.

Run:
    python -m src.tree.train_tree
"""

import json
import os
import pickle
import sqlite3
import warnings
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import datetime
import random
import string

warnings.filterwarnings("ignore")

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
    random_state: int = 42

    # XGBoost hyperparams
    n_estimators: int = 1000
    learning_rate: float = 0.03
    max_depth: int = 6
    min_child_weight: int = 50
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 50

    max_tickers: int = 0  # 0 = use all tickers

    model_output_path: str = f"artifacts/{month}-{day}-{tag}-xgb_tree.pkl"
    config_output_path: str = f"artifacts/{month}-{day}-{tag}-xgb_tree_config.json"


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
    for c in ["open", "high", "low", "close", "adj_close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("float64")
    return df


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    return (100.0 - 100.0 / (1.0 + gain / (loss + 1e-8))) / 100.0 - 0.5


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute point-in-time features for each row (no look-ahead)."""
    out = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"]
        high = g["high"]
        low = g["low"]
        open_ = g["open"]
        volume = g["volume"].clip(lower=1.0)

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

        # Rolling skew and kurtosis of returns
        g["skew_20"] = g["log_ret_1"].rolling(20).skew()
        g["kurt_20"] = g["log_ret_1"].rolling(20).kurt()

        # Target: log-return at horizon
        g["target"] = np.log(close.shift(-1) / close)

        out.append(g)

    result = pd.concat(out, ignore_index=True)
    result = result.replace([np.inf, -np.inf], np.nan)
    return result


FEATURE_COLS = [
    "log_ret_1", "log_ret_3", "log_ret_5", "log_ret_10", "log_ret_20", "log_ret_60",
    "hl_range", "oc_change",
    "close_vs_ma_5", "close_vs_ma_10", "close_vs_ma_20", "close_vs_ma_50", "close_vs_ma_200",
    "ma_5_vs_20", "ma_20_vs_50",
    "rolling_vol_5", "rolling_vol_10", "rolling_vol_20", "vol_of_vol_20",
    "log_vol_chg_1", "log_vol_chg_5", "vol_vs_ma_5", "vol_vs_ma_20",
    "rsi_14", "rsi_7",
    "macd_norm", "macd_signal_norm",
    "bb_pos", "bb_width",
    "atr_14_norm",
    "skew_20", "kurt_20",
]


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


def build_grouped_time_folds(
    df: pd.DataFrame, n_splits: int, embargo: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Return index arrays for grouped time-series CV."""
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    folds = []
    for fold_num in range(n_splits):
        tr_idx, va_idx = [], []
        for _, grp in df.groupby("ticker", sort=False):
            idx = grp.index.to_numpy()
            n = len(idx)
            if n < (n_splits + 1) * max(embargo, 1) + 5:
                continue
            boundaries = np.linspace(0, n, n_splits + 2, dtype=int)
            va_start = boundaries[fold_num + 1]
            va_end = boundaries[fold_num + 2]
            purge_end = va_start - embargo
            if purge_end <= 0 or va_end <= va_start:
                continue
            tr_idx.extend(idx[:purge_end].tolist())
            va_idx.extend(idx[va_start:va_end].tolist())
        if tr_idx and va_idx:
            folds.append((np.array(tr_idx), np.array(va_idx)))
    if not folds:
        raise ValueError("No CV folds built.")
    return folds


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "directional_accuracy": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def main():
    cfg = Config()
    np.random.seed(cfg.random_state)

    print("Loading data...")
    raw_df = load_prices_from_sqlite(cfg.db_path, cfg.table_name)

    all_tickers = list(raw_df["ticker"].unique())
    if cfg.max_tickers > 0 and len(all_tickers) > cfg.max_tickers:
        rng = np.random.default_rng(cfg.random_state)
        all_tickers = rng.choice(all_tickers, cfg.max_tickers, replace=False).tolist()
        raw_df = raw_df[raw_df["ticker"].isin(all_tickers)].reset_index(drop=True)
        print(f"Subsampled to {cfg.max_tickers} tickers")
    else:
        print(f"Using all {len(all_tickers)} tickers")

    print("Computing features...")
    feat_df = make_features(raw_df)

    print("Splitting train/test...")
    train_df, test_df = split_train_test_per_ticker(feat_df, cfg.test_size, cfg.target_horizon)
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    # Drop rows with NaN in features or target
    feature_cols = [c for c in FEATURE_COLS if c in train_df.columns]
    train_clean = train_df[feature_cols + ["target", "ticker", "date", "close"]].dropna().copy()
    test_clean = test_df[feature_cols + ["target", "ticker", "date", "close"]].dropna().copy()
    print(f"Train (no-NaN): {len(train_clean):,} | Test (no-NaN): {len(test_clean):,}")

    X_train = train_clean[feature_cols].values.astype(np.float32)
    y_train = train_clean["target"].values.astype(np.float32)
    X_test = test_clean[feature_cols].values.astype(np.float32)
    y_test = test_clean["target"].values.astype(np.float32)

    print("\nBuilding CV folds...")
    folds = build_grouped_time_folds(train_clean, cfg.n_splits, cfg.target_horizon)
    print(f"CV folds: {len(folds)}")

    cv_results = []
    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_i}/{len(folds)} ===")
        X_tr = X_train[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train[va_idx]
        y_va = y_train[va_idx]
        print(f"  Train: {len(X_tr):,} | Val: {len(X_va):,}")

        model = xgb.XGBRegressor(
            n_estimators=cfg.n_estimators,
            learning_rate=cfg.learning_rate,
            max_depth=cfg.max_depth,
            min_child_weight=cfg.min_child_weight,
            subsample=cfg.subsample,
            colsample_bytree=cfg.colsample_bytree,
            reg_alpha=cfg.reg_alpha,
            reg_lambda=cfg.reg_lambda,
            early_stopping_rounds=cfg.early_stopping_rounds,
            eval_metric="rmse",
            random_state=cfg.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_va, y_va)],
            verbose=100,
        )
        y_va_pred = model.predict(X_va)
        metrics = regression_metrics(y_va, y_va_pred)
        cv_results.append(metrics)
        print(f"  Fold {fold_i} | RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} R2={metrics['r2']:.6f} DA={metrics['directional_accuracy']:.4f}")

    print("\n=== CV Summary ===")
    cv_df = pd.DataFrame(cv_results)
    print(cv_df.mean(numeric_only=True).to_string())

    print("\nTraining final model on all training data...")
    n_train = len(X_train)
    split = int(n_train * 0.9)
    final_model = xgb.XGBRegressor(
        n_estimators=cfg.n_estimators,
        learning_rate=cfg.learning_rate,
        max_depth=cfg.max_depth,
        min_child_weight=cfg.min_child_weight,
        subsample=cfg.subsample,
        colsample_bytree=cfg.colsample_bytree,
        reg_alpha=cfg.reg_alpha,
        reg_lambda=cfg.reg_lambda,
        early_stopping_rounds=cfg.early_stopping_rounds,
        eval_metric="rmse",
        random_state=cfg.random_state,
        n_jobs=-1,
        verbosity=0,
    )
    final_model.fit(
        X_train[:split], y_train[:split],
        eval_set=[(X_train[split:], y_train[split:])],
        verbose=200,
    )

    print("\nEvaluating on held-out test set...")
    y_test_pred = final_model.predict(X_test)
    test_metrics = regression_metrics(y_test, y_test_pred)
    print(f"Test | RMSE={test_metrics['rmse']:.6f} MAE={test_metrics['mae']:.6f} R2={test_metrics['r2']:.6f} DA={test_metrics['directional_accuracy']:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    payload = {
        "model": final_model,
        "feature_cols": feature_cols,
        "config": asdict(cfg),
        "cv_results": cv_results,
        "test_metrics": test_metrics,
    }
    with open(cfg.model_output_path, "wb") as f:
        pickle.dump(payload, f)

    config_dict = asdict(cfg)
    with open(cfg.config_output_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    out_df = test_clean[["ticker", "date", "close"]].copy()
    out_df["y_true"] = y_test
    out_df["y_pred"] = y_test_pred
    pred_path = f"results/{month}-{day}-{tag}-xgb_tree_predictions.csv"
    out_df.to_csv(pred_path, index=False)

    result_txt = (
        f"XGBoost Decision Tree Results (target=return):\n"
        f"RMSE={test_metrics['rmse']:.6f}\n"
        f"MAE={test_metrics['mae']:.6f}\n"
        f"R2={test_metrics['r2']:.6f}\n"
        f"DA={test_metrics['directional_accuracy']:.4f}\n\n"
        f"CV Mean R2={cv_df['r2'].mean():.6f}\n"
        f"CV Mean DA={cv_df['directional_accuracy'].mean():.4f}\n"
    )
    with open(f"{pred_path}.txt", "w") as f:
        f.write(result_txt)

    print(f"\nSaved model   : {cfg.model_output_path}")
    print(f"Saved results : {pred_path}")

    # Feature importance top-20
    importances = pd.Series(
        final_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
    print("\nTop-20 feature importances:")
    print(importances.head(20).to_string())


if __name__ == "__main__":
    main()
