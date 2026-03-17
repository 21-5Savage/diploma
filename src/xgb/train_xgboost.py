"""
python -m src.models.xgb.train_xgboost
"""

import os
import pickle
import sqlite3
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


@dataclass
class Config:
    db_path: str = "dataset/stock_prices_20y.db"
    table_name: str = "prices"
    n_splits: int = 5
    test_size: float = 0.2
    target_horizon: int = 5
    predict_target: str = "return"   # "return" or "price"
    random_state: int = 42
    model_output_path: str = "artifacts/xgb_model.pkl"

    xgb_params: dict = None

    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                "n_estimators": 200,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "objective": "reg:squarederror",
                "random_state": self.random_state,
                "n_jobs": 2,
                "tree_method": "hist",
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


def make_features(df: pd.DataFrame, horizon: int, predict_target: str) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).copy()

    float_cols = ["open", "high", "low", "close", "adj_close"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], downcast="float")
    df["volume"] = pd.to_numeric(df["volume"], downcast="integer")

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

        if predict_target == "return":
            g["target"] = close.shift(-horizon) / close - 1.0
        elif predict_target == "price":
            g["target"] = close.shift(-horizon)
        else:
            raise ValueError("predict_target must be 'return' or 'price'")

        out.append(g)

    df = pd.concat(out, ignore_index=True)
    return df


def split_train_test_per_ticker(df: pd.DataFrame, test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for _, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        split_idx = int(len(grp) * (1 - test_size))

        if split_idx <= 0 or split_idx >= len(grp):
            continue

        train_parts.append(grp.iloc[:split_idx].copy())
        test_parts.append(grp.iloc[split_idx:].copy())

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def build_grouped_time_folds(train_df: pd.DataFrame, n_splits: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    train_df = train_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    folds = []

    for fold_num in range(n_splits):
        tr_idx_all = []
        va_idx_all = []

        for _, grp in train_df.groupby("ticker"):
            idx = grp.index.to_numpy()
            n = len(idx)

            if n < (n_splits + 1):
                continue

            fold_sizes = np.full(n_splits, n // (n_splits + 1), dtype=int)
            remainder = n % (n_splits + 1)

            for i in range(min(remainder, n_splits)):
                fold_sizes[i] += 1

            boundaries = []
            start = n // (n_splits + 1) + (1 if remainder > n_splits else 0)
            current = start

            for fs in fold_sizes:
                end = min(current + fs, n)
                boundaries.append((current, end))
                current = end

            if fold_num >= len(boundaries):
                continue

            va_start, va_end = boundaries[fold_num]
            if va_start >= va_end:
                continue

            tr_local = idx[:va_start]
            va_local = idx[va_start:va_end]

            if len(tr_local) == 0 or len(va_local) == 0:
                continue

            tr_idx_all.extend(tr_local.tolist())
            va_idx_all.extend(va_local.tolist())

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
    df = load_prices_from_sqlite(cfg.db_path, cfg.table_name)

    print("Building features...")
    df = make_features(df, horizon=cfg.target_horizon, predict_target=cfg.predict_target)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df = df.dropna(subset=FEATURE_COLS + ["target"])

    print("Splitting train/test per ticker...")
    train_df, test_df = split_train_test_per_ticker(df, test_size=cfg.test_size)

    train_df = train_df.sort_values(["ticker", "date"]).reset_index(drop=True)
    test_df = test_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows : {len(test_df):,}")
    print(f"Tickers   : {train_df['ticker'].nunique()}")

    X_train = train_df[FEATURE_COLS]
    y_train = train_df["target"].values

    print("Building grouped time-series CV folds...")
    folds = build_grouped_time_folds(train_df, n_splits=cfg.n_splits)
    print(f"Number of CV folds built: {len(folds)}")

    cv_results = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_i}/{cfg.n_splits} ===")
        print(f"Train rows: {len(tr_idx):,} | Valid rows: {len(va_idx):,}")

        X_tr = X_train.iloc[tr_idx]
        y_tr = y_train[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_train[va_idx]

        model = XGBRegressor(**cfg.xgb_params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)

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

    print("\n=== CV Summary ===")
    cv_df = pd.DataFrame(cv_results)
    print(cv_df.mean(numeric_only=True).to_string())

    print("\nTraining final model on all training data...")
    final_model = XGBRegressor(**cfg.xgb_params)
    final_model.fit(X_train, y_train, verbose=False)

    fi = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": final_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    print("\n=== Top 20 Feature Importances ===")
    print(fi.head(20).to_string(index=False))

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
"""
python -m src.models.xgb.train_xgboost
"""
