"""
python -m src.tree.train_tree
"""

import os
import pickle
import sqlite3
from dataclasses import asdict, dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor

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
    target_horizon: int = 5
    predict_target: str = "return"   # "return" or "price"
    random_state: int = 42
    model_output_path: str = f"artifacts/{month}-{day}-{tag}-decision_tree.pkl"
    dt_params: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.dt_params:
            self.dt_params = {
                "max_depth": 5,
                "min_samples_leaf": 20,
                "random_state": self.random_state,
            }


FEATURE_COLS = [
    "open", "high", "low", "close", "adj_close", "volume",
    "ret_1", "ret_5", "ret_10",
    "close_lag_1", "close_lag_5", "close_lag_10",
    "volume_lag_1", "volume_lag_5",
    "close_ma_5", "close_ma_10", "close_ma_20",
    "close_std_5", "close_std_10",
    "vol_ma_5",
    "hl_range", "oc_change",
    "close_vs_ma_5", "close_vs_ma_10",
]


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
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build strictly backward-looking features only.
    No target is created here to avoid leakage across split boundaries.
    """
    df = df.sort_values(["ticker", "date"]).copy()

    float_cols = ["open", "high", "low", "close", "adj_close"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce", downcast="integer")

    out = []

    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()

        close = g["close"]
        volume = g["volume"]

        g["ret_1"] = close.pct_change(1)
        g["ret_5"] = close.pct_change(5)
        g["ret_10"] = close.pct_change(10)

        g["close_lag_1"] = close.shift(1)
        g["close_lag_5"] = close.shift(5)
        g["close_lag_10"] = close.shift(10)

        g["volume_lag_1"] = volume.shift(1)
        g["volume_lag_5"] = volume.shift(5)

        ma_5 = close.rolling(5).mean()
        ma_10 = close.rolling(10).mean()
        ma_20 = close.rolling(20).mean()

        std_5 = close.rolling(5).std()
        std_10 = close.rolling(10).std()

        vol_ma_5 = volume.rolling(5).mean()

        g["close_ma_5"] = ma_5
        g["close_ma_10"] = ma_10
        g["close_ma_20"] = ma_20
        g["close_std_5"] = std_5
        g["close_std_10"] = std_10
        g["vol_ma_5"] = vol_ma_5

        g["hl_range"] = (g["high"] - g["low"]) / g["close"]
        g["oc_change"] = (g["close"] - g["open"]) / g["open"]

        g["close_vs_ma_5"] = g["close"] / ma_5 - 1.0
        g["close_vs_ma_10"] = g["close"] / ma_10 - 1.0

        out.append(g)

    df = pd.concat(out, ignore_index=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def add_target_per_split(df: pd.DataFrame, horizon: int, predict_target: str) -> pd.DataFrame:
    """
    Create target INSIDE an already-created split only.
    This prevents labels from crossing train/test boundaries.
    """
    out = []

    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"]

        if predict_target == "return":
            g["target"] = close.shift(-horizon) / close - 1.0
        elif predict_target == "price":
            g["target"] = close.shift(-horizon)
        else:
            raise ValueError("predict_target must be 'return' or 'price'")

        out.append(g)

    return pd.concat(out, ignore_index=True)


def split_train_test_per_ticker(
    df: pd.DataFrame,
    test_size: float,
    embargo: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split each ticker chronologically and drop an embargo window from the
    end of train so train labels cannot look into the test window.
    """
    train_parts = []
    test_parts = []

    for _, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        split_idx = int(n * (1 - test_size))
        train_end = split_idx - embargo

        if split_idx <= 0 or split_idx >= n:
            continue
        if train_end <= 0:
            continue

        train_parts.append(grp.iloc[:train_end].copy())
        test_parts.append(grp.iloc[split_idx:].copy())

    if not train_parts or not test_parts:
        raise ValueError("No valid train/test splits were created.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def build_grouped_time_folds(
    train_df: pd.DataFrame,
    n_splits: int,
    embargo: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Build expanding-window grouped time-series folds with a purge/embargo.
    Validation is always later in time than training for each ticker.
    Training rows whose targets would reach into validation are excluded.
    """
    train_df = train_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    folds = []

    for fold_num in range(n_splits):
        tr_idx_all = []
        va_idx_all = []

        for _, grp in train_df.groupby("ticker", sort=False):
            grp = grp.sort_values("date")
            idx = grp.index.to_numpy()
            n = len(idx)

            # Need enough room for initial train + n_splits validation chunks
            if n < (n_splits + 1) * max(embargo, 1) + 5:
                continue

            boundaries = np.linspace(0, n, n_splits + 2, dtype=int)
            va_start = boundaries[fold_num + 1]
            va_end = boundaries[fold_num + 2]

            purged_train_end = va_start - embargo
            if purged_train_end <= 0 or va_end <= va_start:
                continue

            tr_local = idx[:purged_train_end]
            va_local = idx[va_start:va_end]

            if len(tr_local) == 0 or len(va_local) == 0:
                continue

            tr_idx_all.extend(tr_local.tolist())
            va_idx_all.extend(va_local.tolist())

        if tr_idx_all and va_idx_all:
            folds.append((np.array(tr_idx_all), np.array(va_idx_all)))

    return folds


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    direction_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": direction_acc,
    }


def main():
    cfg = Config()

    print("Loading data...")
    raw_df = load_prices_from_sqlite(cfg.db_path, cfg.table_name)

    print("Building past-only features...")
    feat_df = make_features(raw_df)

    missing = [c for c in FEATURE_COLS if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    print("Splitting train/test per ticker with embargo...")
    train_feat_df, test_feat_df = split_train_test_per_ticker(
        feat_df,
        test_size=cfg.test_size,
        embargo=cfg.target_horizon,
    )

    print("Creating targets inside each split...")
    train_df = add_target_per_split(
        train_feat_df,
        horizon=cfg.target_horizon,
        predict_target=cfg.predict_target,
    )
    test_df = add_target_per_split(
        test_feat_df,
        horizon=cfg.target_horizon,
        predict_target=cfg.predict_target,
    )

    train_df = train_df.dropna(subset=FEATURE_COLS + ["target"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    test_df = test_df.dropna(subset=FEATURE_COLS + ["target"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows : {len(test_df):,}")
    print(f"Tickers   : {train_df['ticker'].nunique()}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"].values

    print("Building purged grouped time-series CV folds...")
    folds = build_grouped_time_folds(
        train_df,
        n_splits=cfg.n_splits,
        embargo=cfg.target_horizon,
    )
    print(f"Number of CV folds built: {len(folds)}")

    cv_results = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_i}/{len(folds)} ===")
        print(f"Train rows: {len(tr_idx):,} | Valid rows: {len(va_idx):,}")

        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train[va_idx]

        model = DecisionTreeRegressor(**cfg.dt_params)
        model.fit(X_tr, y_tr)

        pred_va = model.predict(X_va)
        metrics = regression_metrics(y_va, pred_va)
        cv_results.append(metrics)

        print(
            f"Fold {fold_i} | "
            f"RMSE={metrics['rmse']:.6f}, "
            f"MAE={metrics['mae']:.6f}, "
            f"R2={metrics['r2']:.6f}, "
            f"DirAcc={metrics['directional_accuracy']:.4f}"
        )

    if cv_results:
        print("\n=== CV Summary ===")
        cv_df = pd.DataFrame(cv_results)
        print(cv_df.mean(numeric_only=True).to_string())
    else:
        print("\nNo valid CV folds were built.")

    print("\nTraining final model on all training data...")
    final_model = DecisionTreeRegressor(**cfg.dt_params)
    final_model.fit(X_train, y_train)

    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n=== Top 20 Feature Importances ===")
    print(fi.head(20).to_string(index=False))

    print("\nEvaluating on held-out test set...")
    X_test = test_df[FEATURE_COLS]
    y_test = test_df["target"].values
    y_pred = final_model.predict(X_test)
    test_metrics = regression_metrics(y_test, y_pred)
    print(
        "Test | "
        f"RMSE={test_metrics['rmse']:.6f}, "
        f"MAE={test_metrics['mae']:.6f}, "
        f"R2={test_metrics['r2']:.6f}, "
        f"DirAcc={test_metrics['directional_accuracy']:.4f}"
    )

    os.makedirs(os.path.dirname(cfg.model_output_path), exist_ok=True)

    payload = {
        "model": final_model,
        "feature_cols": FEATURE_COLS,
        "config": asdict(cfg),
    }

    with open(cfg.model_output_path, "wb") as f:
        pickle.dump(payload, f)

    print(f"\nSaved model to: {cfg.model_output_path}")


if __name__ == "__main__":
    main()
