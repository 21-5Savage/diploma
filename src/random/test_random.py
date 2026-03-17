"""
Random baseline for next-day direction guessing.

Run:
    python -m src.random.test_random
"""

import sqlite3
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class Config:
    db_path: str = "dataset/stock_prices_20y.db"
    table_name: str = "prices"
    random_state: int = 42
    output_csv: str = "random_test_predictions.csv"


def load_prices_from_sqlite(db_path: str, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT ticker, date, close
        FROM {table_name}
        ORDER BY ticker, date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()

    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


def build_next_day_direction_target(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    next_close = out.groupby("ticker")["close"].shift(-1)
    out["target_return_1d"] = next_close / out["close"] - 1.0
    out = out.dropna(subset=["target_return_1d"]).copy()
    out["y_true_dir"] = (out["target_return_1d"] > 0).astype(int)
    return out


def split_train_test_per_ticker(df: pd.DataFrame, test_size: float = 0.2) -> pd.DataFrame:
    test_parts = []
    for _, grp in df.groupby("ticker"):
        grp = grp.sort_values("date").reset_index(drop=True)
        split_idx = int(len(grp) * (1 - test_size))
        if split_idx <= 0 or split_idx >= len(grp):
            continue
        test_parts.append(grp.iloc[split_idx:].copy())

    if not test_parts:
        return pd.DataFrame(columns=df.columns)
    return pd.concat(test_parts, ignore_index=True)


def main() -> None:
    cfg = Config()
    rng = np.random.default_rng(cfg.random_state)

    print("Loading data...")
    df = load_prices_from_sqlite(cfg.db_path, cfg.table_name)
    df = build_next_day_direction_target(df)

    test_df = split_train_test_per_ticker(df, test_size=0.2)
    if test_df.empty:
        raise RuntimeError("No test rows after split. Check dataset contents.")

    # Random coin-flip guesses: 1 = up, 0 = down/non-up
    test_df["y_pred_dir"] = rng.integers(0, 2, size=len(test_df))

    overall_acc = (test_df["y_pred_dir"] == test_df["y_true_dir"]).mean()
    up_rate_true = test_df["y_true_dir"].mean()
    up_rate_pred = test_df["y_pred_dir"].mean()

    print("\n=== Random Next-Day Direction Baseline ===")
    print(f"Test rows             : {len(test_df):,}")
    print(f"Tickers               : {test_df['ticker'].nunique():,}")
    print(f"Directional Accuracy  : {overall_acc:.4f}")
    print(f"True Up Rate          : {up_rate_true:.4f}")
    print(f"Predicted Up Rate     : {up_rate_pred:.4f}")

    per_ticker = (
        test_df.assign(correct=(test_df["y_true_dir"] == test_df["y_pred_dir"]).astype(float))
        .groupby("ticker", as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "directional_accuracy"})
        .sort_values("directional_accuracy", ascending=False)
    )

    print("\nTop 20 tickers by random directional accuracy (noise baseline):")
    print(per_ticker.head(20).to_string(index=False))

    out = test_df[["ticker", "date", "close", "target_return_1d", "y_true_dir", "y_pred_dir"]].copy()
    out.to_csv(cfg.output_csv, index=False)
    print(f"\nSaved predictions to {cfg.output_csv}")


if __name__ == "__main__":
    main()
