"""
Run sampled-window evaluation and plotting for tree/lstm/llm.

Outputs:
  - sampled ticker lists
  - daily metrics CSVs
  - summary CSV
  - per-method 4-panel metric dashboards (MAE/RMSE/DA/MAPE by date)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


DB_PATH = os.environ.get("PIPELINE_DB", "pipeline/db/pipeline.db")
RESULTS_DIR = Path("pipeline/results")
PLOTS_DIR = RESULTS_DIR / "plots"


def get_trading_dates(conn: sqlite3.Connection, date_from: str, date_to: str) -> list[str]:
    rows = conn.execute(
        """
        SELECT DISTINCT date
        FROM prices
        WHERE date >= ? AND date <= ?
        ORDER BY date
        """,
        [date_from, date_to],
    ).fetchall()
    return [r[0] for r in rows]


def sample_tickers(
    conn: sqlite3.Connection,
    date_from: str,
    date_to: str,
    sample_size: int,
    random_state: int,
    min_history_days: int = 60,
) -> list[str]:
    trading_dates = get_trading_dates(conn, date_from, date_to)
    if not trading_dates:
        return []
    n_days = len(trading_dates)
    eligible = pd.read_sql_query(
        """
        SELECT ticker
        FROM prices
        GROUP BY ticker
        HAVING COUNT(DISTINCT CASE WHEN date >= ? AND date <= ? THEN date END) = ?
           AND SUM(CASE WHEN date < ? THEN 1 ELSE 0 END) >= ?
        ORDER BY ticker
        """,
        conn,
        params=[date_from, date_to, n_days, date_from, min_history_days],
    )
    if eligible.empty:
        return []
    rng = np.random.default_rng(random_state)
    tickers = eligible["ticker"].tolist()
    if len(tickers) <= sample_size:
        return tickers
    return sorted(rng.choice(tickers, size=sample_size, replace=False).tolist())


def load_actuals(conn: sqlite3.Connection, tickers: list[str], date_from: str, date_to: str) -> pd.DataFrame:
    placeholders = ",".join("?" * len(tickers))
    df = pd.read_sql_query(
        f"""
        SELECT ticker, date, close
        FROM prices
        WHERE ticker IN ({placeholders}) AND date >= ? AND date <= ?
        ORDER BY ticker, date
        """,
        conn,
        params=tickers + [date_from, date_to],
    )
    if df.empty:
        return df
    df["pred_date"] = pd.to_datetime(df["date"])
    df["actual_close"] = pd.to_numeric(df["close"], errors="coerce")
    prev = pd.read_sql_query(
        f"""
        SELECT p.ticker, p.date,
               (
                 SELECT q.close FROM prices q
                 WHERE q.ticker = p.ticker AND q.date < p.date
                 ORDER BY q.date DESC
                 LIMIT 1
               ) AS prev_close
        FROM prices p
        WHERE p.ticker IN ({placeholders}) AND p.date >= ? AND p.date <= ?
        ORDER BY p.ticker, p.date
        """,
        conn,
        params=tickers + [date_from, date_to],
    )
    prev["pred_date"] = pd.to_datetime(prev["date"])
    prev["prev_close"] = pd.to_numeric(prev["prev_close"], errors="coerce")
    return df.merge(prev[["ticker", "pred_date", "prev_close"]], on=["ticker", "pred_date"], how="left")[
        ["ticker", "pred_date", "actual_close", "prev_close"]
    ]


def load_pipeline_predictions(
    conn: sqlite3.Connection,
    tickers: list[str],
    model_name: str,
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    placeholders = ",".join("?" * len(tickers))
    df = pd.read_sql_query(
        f"""
        SELECT pred_date, ticker, model_name, pred_return
        FROM predictions
        WHERE ticker IN ({placeholders})
          AND model_name = ?
          AND pred_date >= ?
          AND pred_date <= ?
        ORDER BY pred_date, ticker
        """,
        conn,
        params=tickers + [model_name, date_from, date_to],
    )
    if df.empty:
        return df
    df["pred_date"] = pd.to_datetime(df["pred_date"])
    df["pred_return"] = pd.to_numeric(df["pred_return"], errors="coerce")
    return df


def build_pipeline_method_frame(
    conn: sqlite3.Connection,
    tickers: list[str],
    model_name: str,
    date_from: str,
    date_to: str,
) -> pd.DataFrame:
    actuals = load_actuals(conn, tickers, date_from, date_to)
    preds = load_pipeline_predictions(conn, tickers, model_name, date_from, date_to)
    if actuals.empty or preds.empty:
        return pd.DataFrame()
    merged = preds.merge(actuals, on=["ticker", "pred_date"], how="inner")
    merged = merged.dropna(subset=["pred_return", "actual_close", "prev_close"]).copy()
    if merged.empty:
        return merged
    merged["pred_close"] = merged["prev_close"] * np.exp(merged["pred_return"].clip(-2.0, 2.0))
    merged["abs_error"] = (merged["pred_close"] - merged["actual_close"]).abs()
    merged["directional_correct"] = (
        np.sign(merged["pred_close"] - merged["prev_close"])
        == np.sign(merged["actual_close"] - merged["prev_close"])
    ).astype(float)
    merged["model_name"] = model_name
    return merged[["pred_date", "ticker", "model_name", "actual_close", "pred_close", "abs_error", "directional_correct"]]


def build_llm_method_frame(llm_csv: str, tickers: list[str], date_from: str, date_to: str) -> pd.DataFrame:
    df = pd.read_csv(llm_csv)
    required = {"ticker", "date", "y_true", "y_pred", "prev_close"}
    if not required.issubset(df.columns):
        return pd.DataFrame()
    df = df[df["ticker"].isin(tickers)].copy()
    df["pred_date"] = pd.to_datetime(df["date"])
    df = df[(df["pred_date"] >= pd.Timestamp(date_from)) & (df["pred_date"] <= pd.Timestamp(date_to))].copy()
    if df.empty:
        return df
    df["actual_close"] = pd.to_numeric(df["y_true"], errors="coerce")
    df["pred_close"] = pd.to_numeric(df["y_pred"], errors="coerce")
    df["prev_close"] = pd.to_numeric(df["prev_close"], errors="coerce")
    df = df.dropna(subset=["actual_close", "pred_close", "prev_close"]).copy()
    df["abs_error"] = (df["pred_close"] - df["actual_close"]).abs()
    df["directional_correct"] = (
        np.sign(df["pred_close"] - df["prev_close"])
        == np.sign(df["actual_close"] - df["prev_close"])
    ).astype(float)
    df["model_name"] = "llm"
    return df[["pred_date", "ticker", "model_name", "actual_close", "pred_close", "abs_error", "directional_correct"]]


def summarise_daily(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pred_date, g in df.groupby("pred_date", sort=True):
        y_true = g["actual_close"].to_numpy(dtype=float)
        y_pred = g["pred_close"].to_numpy(dtype=float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        da = float(np.mean(g["directional_correct"].to_numpy(dtype=float)))
        denom = np.clip(np.abs(y_true), 1e-6, None)
        mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
        rows.append(
            {
                "pred_date": pd.Timestamp(pred_date).strftime("%Y-%m-%d"),
                "mae": mae,
                "rmse": rmse,
                "directional_acc": da,
                "mape_pct": mape,
                "n_predictions": len(g),
            }
        )
    return pd.DataFrame(rows)


def summarise_overall(model_name: str, daily_df: pd.DataFrame) -> dict:
    return {
        "model_name": model_name,
        "mean_mae": float(daily_df["mae"].mean()),
        "mean_rmse": float(daily_df["rmse"].mean()),
        "mean_directional_acc": float(daily_df["directional_acc"].mean()),
        "mean_mape_pct": float(daily_df["mape_pct"].mean()),
        "n_days": int(len(daily_df)),
        "mean_predictions_per_day": float(daily_df["n_predictions"].mean()),
    }


def plot_method_metrics(model_name: str, daily_df: pd.DataFrame, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.flatten()
    specs = [
        ("mae", "MAE"),
        ("rmse", "RMSE"),
        ("directional_acc", "Dir Acc"),
        ("mape_pct", "MAPE %"),
    ]
    x = pd.to_datetime(daily_df["pred_date"])
    for ax, (col, title) in zip(axes, specs):
        ax.plot(x, daily_df[col], marker="o", linewidth=1.8)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"{model_name.upper()} Daily Metrics", fontsize=15)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date-from", required=True)
    parser.add_argument("--date-to", required=True)
    parser.add_argument("--pipeline-sample-size", type=int, default=100)
    parser.add_argument("--llm-sample-size", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--llm-csv", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    pipeline_tickers = sample_tickers(
        conn, args.date_from, args.date_to, args.pipeline_sample_size, args.random_state
    )
    llm_tickers = sample_tickers(
        conn, args.date_from, args.date_to, args.llm_sample_size, args.random_state + 1
    )
    if not pipeline_tickers or not llm_tickers:
        print("Not enough eligible tickers in the requested window.")
        conn.close()
        return

    pd.Series(pipeline_tickers, name="ticker").to_csv(
        RESULTS_DIR / f"sampled_pipeline_tickers_{args.date_from}_{args.date_to}.csv", index=False
    )
    pd.Series(llm_tickers, name="ticker").to_csv(
        RESULTS_DIR / f"sampled_llm_tickers_{args.date_from}_{args.date_to}.csv", index=False
    )

    outputs = []
    summary_rows = []

    for model_name in ["tree", "lstm"]:
        model_df = build_pipeline_method_frame(
            conn, pipeline_tickers, model_name, args.date_from, args.date_to
        )
        if model_df.empty:
            continue
        daily_df = summarise_daily(model_df)
        daily_path = RESULTS_DIR / f"sampled_{model_name}_{args.date_from}_{args.date_to}_daily.csv"
        daily_df.to_csv(daily_path, index=False)
        outputs.append(daily_path)
        plot_path = PLOTS_DIR / f"sampled_{model_name}_{args.date_from}_{args.date_to}_metrics.png"
        plot_method_metrics(model_name, daily_df, plot_path)
        outputs.append(plot_path)
        summary_rows.append(summarise_overall(model_name, daily_df))

    conn.close()

    llm_df = build_llm_method_frame(args.llm_csv, llm_tickers, args.date_from, args.date_to)
    if not llm_df.empty:
        daily_df = summarise_daily(llm_df)
        daily_path = RESULTS_DIR / f"sampled_llm_{args.date_from}_{args.date_to}_daily.csv"
        daily_df.to_csv(daily_path, index=False)
        outputs.append(daily_path)
        plot_path = PLOTS_DIR / f"sampled_llm_{args.date_from}_{args.date_to}_metrics.png"
        plot_method_metrics("llm", daily_df, plot_path)
        outputs.append(plot_path)
        summary_rows.append(summarise_overall("llm", daily_df))

    summary_df = pd.DataFrame(summary_rows)
    summary_path = RESULTS_DIR / f"sampled_summary_{args.date_from}_{args.date_to}.csv"
    summary_df.to_csv(summary_path, index=False)
    outputs.append(summary_path)

    print("Wrote sampled-window outputs:")
    for path in outputs:
        print(f"  {path}")
    if not summary_df.empty:
        print("\nSummary:")
        print(summary_df.to_string(index=False, float_format="%.6f"))


if __name__ == "__main__":
    main()
