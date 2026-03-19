"""
python -m src.tree.test_tree [model_input_path]
"""

import argparse
import os
import pickle
import sqlite3
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


@dataclass
class Config:
    db_path: str = "dataset/stock_prices_20y.db"
    table_name: str = "prices"
    test_size: float = 0.2
    model_input_path: str = "artifacts/tree_model.pkl"
    predictions_output_path: str = "results/tree_test_predictions.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run tree model evaluation against the test split."
    )
    parser.add_argument(
        "model_input_path",
        nargs="?",
        default=Config.model_input_path,
        help="Path to the saved model pickle file.",
    )
    return parser.parse_args()


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
    No target is created here.
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
    Create target only inside the already-created split.
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
    args = parse_args()
    cfg = Config(model_input_path=args.model_input_path)

    print("Loading model...")
    with open(cfg.model_input_path, "rb") as f:
        payload = pickle.load(f)

    model = payload["model"]
    feature_cols = payload["feature_cols"]
    saved_cfg = payload["config"]

    target_horizon = saved_cfg["target_horizon"]
    predict_target = saved_cfg["predict_target"]
    test_size = saved_cfg["test_size"]

    print("Loading data...")
    raw_df = load_prices_from_sqlite(cfg.db_path, cfg.table_name)

    print("Building past-only features...")
    feat_df = make_features(raw_df)

    missing = [c for c in feature_cols if c not in feat_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    print("Rebuilding leak-safe train/test split...")
    _, test_feat_df = split_train_test_per_ticker(
        feat_df,
        test_size=test_size,
        embargo=target_horizon,
    )

    print("Creating targets inside test split...")
    test_df = add_target_per_split(
        test_feat_df,
        horizon=target_horizon,
        predict_target=predict_target,
    )

    test_df = test_df.dropna(subset=feature_cols + ["target"]).sort_values(["ticker", "date"]).reset_index(drop=True)

    X_test = test_df[feature_cols]
    y_test = test_df["target"].values

    print(f"Test rows: {len(test_df):,}")
    print(f"Tickers  : {test_df['ticker'].nunique()}")

    print("Running predictions...")
    y_pred = model.predict(X_test)

    metrics = regression_metrics(y_test, y_pred)
    result = f"""\nDecision Tree Results:
RMSE={metrics['rmse']:.6f}
MAE={metrics['mae']:.6f}
R2={metrics['r2']:.6f}
DA={metrics['directional_accuracy']:.4f}
"""

    print(result)

    # Baseline: predict zero return / no move
    y_pred_baseline = np.zeros_like(y_test, dtype=float)
    baseline_metrics = regression_metrics(y_test, y_pred_baseline)

    np.random.seed(42)
    y_pred_rand = np.random.choice([-1.0, 1.0], size=len(y_test))
    baseline_dir = float(np.mean(np.sign(y_test) == np.sign(y_pred_rand)))

    baseline = f"""Baseline Results:
RMSE={baseline_metrics['rmse']:.6f}
MAE={baseline_metrics['mae']:.6f}
R2={baseline_metrics['r2']:.6f}
DirAcc={baseline_dir:.4f}
"""

    print("Baseline (predict 0)")
    print(baseline)

    os.makedirs(os.path.dirname(cfg.predictions_output_path), exist_ok=True)

    with open(f"{cfg.predictions_output_path}.txt", "w") as file:
        file.write(result)
        file.write("\n")
        file.write(baseline)

    out = test_df[["ticker", "date", "close"]].copy()
    out["y_true"] = y_test
    out["y_pred"] = y_pred
    out.to_csv(cfg.predictions_output_path, index=False)

    print(f"\nSaved predictions to: {cfg.predictions_output_path}")


if __name__ == "__main__":
    main()
