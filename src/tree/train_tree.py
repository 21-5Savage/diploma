"""
Simple decision tree regressor for next-day stock return prediction.

Target  : log(close_t+1 / close_t)  — log return
Metrics : R², RMSE, MAE, Directional Accuracy
          DA = sign(pred_return) == sign(actual_return)

This trainer intentionally uses the shared `pipeline.features` feature set so
the control tree does not get a richer handcrafted-feature advantage over the
sequence models.

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
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from pipeline.features import FEATURE_COLS, make_features_df

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

    # Decision tree control-model hyperparams
    max_depth: int = 8
    min_samples_leaf: int = 100
    min_samples_split: int = 400

    max_tickers: int = 0  # 0 = use all tickers
    max_train_rows_per_ticker: int = 500
    max_test_rows_per_ticker: int = 120

    model_output_path: str = f"artifacts/{month}-{day}-{tag}-decision_tree.pkl"
    config_output_path: str = f"artifacts/{month}-{day}-{tag}-decision_tree_config.json"


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


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the shared point-in-time feature set plus the tree target."""
    feat_df = make_features_df(df).copy()
    feat_df["target"] = np.log(
        feat_df.groupby("ticker", sort=False)["close"].shift(-1) / feat_df["close"]
    )
    return feat_df.replace([np.inf, -np.inf], np.nan)


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


def trim_recent_rows_per_ticker(df: pd.DataFrame, max_rows_per_ticker: int) -> pd.DataFrame:
    if max_rows_per_ticker <= 0:
        return df
    parts = []
    for _, grp in df.groupby("ticker", sort=False):
        parts.append(grp.sort_values("date").tail(max_rows_per_ticker).copy())
    return pd.concat(parts, ignore_index=True) if parts else df


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
    train_clean = trim_recent_rows_per_ticker(train_clean, cfg.max_train_rows_per_ticker)
    test_clean = trim_recent_rows_per_ticker(test_clean, cfg.max_test_rows_per_ticker)
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

        model = DecisionTreeRegressor(
            max_depth=cfg.max_depth,
            min_samples_leaf=cfg.min_samples_leaf,
            min_samples_split=cfg.min_samples_split,
            random_state=cfg.random_state,
        )
        model.fit(X_tr, y_tr)
        y_va_pred = model.predict(X_va)
        metrics = regression_metrics(y_va, y_va_pred)
        cv_results.append(metrics)
        print(f"  Fold {fold_i} | RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} R2={metrics['r2']:.6f} DA={metrics['directional_accuracy']:.4f}")

    print("\n=== CV Summary ===")
    cv_df = pd.DataFrame(cv_results)
    print(cv_df.mean(numeric_only=True).to_string())

    print("\nTraining final model on all training data...")
    final_model = DecisionTreeRegressor(
        max_depth=cfg.max_depth,
        min_samples_leaf=cfg.min_samples_leaf,
        min_samples_split=cfg.min_samples_split,
        random_state=cfg.random_state,
    )
    final_model.fit(X_train, y_train)

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
    pred_path = f"results/{month}-{day}-{tag}-decision_tree_predictions.csv"
    out_df.to_csv(pred_path, index=False)

    result_txt = (
        f"Decision Tree Results (target=return):\n"
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

    importances = pd.Series(final_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print("\nTop-20 feature importances:")
    print(importances.head(20).to_string())


if __name__ == "__main__":
    main()
