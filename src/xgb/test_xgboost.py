"""
python -m src.xgb.test_xgboost
"""

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
    model_input_path: str = "artifacts/xgb_model.pkl"
    predictions_output_path: str = "artifacts/xgboost_test_predictions.csv"


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

    print("Loading model...")
    with open(cfg.model_input_path, "rb") as f:
        payload = pickle.load(f)

    model = payload["model"]
    feature_cols = payload["feature_cols"]
    saved_cfg = payload["config"]

    target_horizon = saved_cfg["target_horizon"]
    predict_target = saved_cfg["predict_target"]

    print("Loading data...")
    df = load_prices_from_sqlite(cfg.db_path, cfg.table_name)

    print("Building features...")
    df = make_features(df, horizon=target_horizon, predict_target=predict_target)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df = df.dropna(subset=feature_cols + ["target"])

    print("Rebuilding train/test split...")
    _, test_df = split_train_test_per_ticker(df, test_size=cfg.test_size)
    test_df = test_df.sort_values(["ticker", "date"]).reset_index(drop=True)

    X_test = test_df[feature_cols]
    y_test = test_df["target"].values

    print(f"Test rows: {len(test_df):,}")
    print(f"Tickers  : {test_df['ticker'].nunique()}")

    print("Running predictions...")
    y_pred = model.predict(X_test)

    metrics = regression_metrics(y_test, y_pred)
    result = f"""RMSE={metrics['rmse']:.6f}\n\n
        MAE={metrics['mae']:.6f}\n
        R2={metrics['r2']:.6f}\n
        DA={metrics['directional_accuracy']:.4f}\n"""

    print("\nResults:")
    print(result)
    
    with open(f"{cfg.predictions_output_path}.txt", "x") as file:
        file.write(result)

    os.makedirs(os.path.dirname(cfg.predictions_output_path), exist_ok=True)

    out = test_df[["ticker", "date", "close"]].copy()
    out["y_true"] = y_test
    out["y_pred"] = y_pred
    out.to_csv(cfg.predictions_output_path, index=False)

    print(f"\nSaved predictions to: {cfg.predictions_output_path}")


if __name__ == "__main__":
    main()
"""
python -m src.xgb.test_xgboost
"""
