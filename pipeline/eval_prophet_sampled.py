"""
Run Prophet inference on the shared 100-ticker sample for the eval window.

For each ticker:
  - Load all DB data
  - Add features (same as train_prophet.py)
  - Train Prophet on data BEFORE the eval window start
  - Predict on eval window dates from DB
  - Produce sampled_prophet_*_rows.csv and sampled_prophet_*_daily.csv

Usage:
    python pipeline/eval_prophet_sampled.py
"""
from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from prophet import Prophet

warnings.filterwarnings("ignore")

DB_PATH = Path("dataset/stock_prices_20y.db")
RESULTS_DIR = Path("pipeline/results")
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

DATE_FROM = "2026-03-16"
DATE_TO = "2026-04-15"
TICKERS_CSV = RESULTS_DIR / f"sampled_pipeline_tickers_{DATE_FROM}_{DATE_TO}.csv"
SUMMARY_CSV = RESULTS_DIR / f"sampled_summary_{DATE_FROM}_{DATE_TO}.csv"

# Prophet config (mirrors train_prophet.py)
CHANGEPOINT_PRIOR_SCALE = 0.2
SEASONALITY_PRIOR_SCALE = 15.0
SEASONALITY_MODE = "multiplicative"


# ── helpers ──────────────────────────────────────────────────────────────────
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = gain / (loss + 1e-8)
    return 100.0 - 100.0 / (1.0 + rs)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    g = df.sort_values("date").copy()
    close = g["close"].astype("float64")
    volume = g["volume"].astype("float64").clip(lower=1.0)
    g["log_ret_1"] = np.log(close / close.shift(1))
    g["log_volume"] = np.log1p(volume)
    g["hl_range"] = (g["high"].astype("float64") - g["low"].astype("float64")) / close
    g["rsi_14"] = compute_rsi(close, 14) / 100.0
    return g.replace([np.inf, -np.inf], np.nan)


def fit_prophet(train_df: pd.DataFrame) -> Prophet:
    close = train_df["close"].astype("float64").clip(lower=1e-6)
    prophet_df = pd.DataFrame({"ds": train_df["date"].values, "y": close.values})
    for col in ["log_volume", "log_ret_1", "hl_range", "rsi_14"]:
        if col in train_df.columns:
            vals = train_df[col].values.astype("float64")
            prophet_df[col] = np.where(np.isfinite(vals), vals, 0.0)
    prophet_df = prophet_df.dropna(subset=["ds", "y"])
    model = Prophet(
        changepoint_prior_scale=CHANGEPOINT_PRIOR_SCALE,
        seasonality_prior_scale=SEASONALITY_PRIOR_SCALE,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode=SEASONALITY_MODE,
    )
    for col in ["log_volume", "log_ret_1", "hl_range", "rsi_14"]:
        model.add_regressor(col, standardize=True)
    model.fit(prophet_df)
    return model


def predict_prophet(model: Prophet, eval_df: pd.DataFrame) -> np.ndarray:
    future = pd.DataFrame({"ds": eval_df["date"].values})
    for col in ["log_volume", "log_ret_1", "hl_range", "rsi_14"]:
        if col in eval_df.columns:
            vals = eval_df[col].values.astype("float64")
            future[col] = np.where(np.isfinite(vals), vals, 0.0)
        else:
            future[col] = 0.0
    forecast = model.predict(future)
    return forecast["yhat"].values.astype(np.float32)


def summarise_daily(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pred_date, g in df.groupby("pred_date", sort=True):
        y_true = g["actual_close"].to_numpy(float)
        y_pred = g["pred_close"].to_numpy(float)
        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        da = float(np.mean(g["directional_correct"].to_numpy(float)))
        mape = float(np.mean(np.abs((y_true - y_pred) / np.clip(np.abs(y_true), 1e-6, None))) * 100)
        rows.append({"pred_date": pred_date, "mae": mae, "rmse": rmse,
                     "directional_acc": da, "mape_pct": mape, "n_predictions": len(g)})
    return pd.DataFrame(rows)


def plot_metrics(daily_df: pd.DataFrame, plot_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle(f"Prophet – sampled daily metrics ({DATE_FROM} → {DATE_TO})")
    metrics = [("mae", "MAE ($)"), ("rmse", "RMSE ($)"),
               ("directional_acc", "Directional Accuracy"), ("mape_pct", "MAPE (%)")]
    for ax, (col, label) in zip(axes.flat, metrics):
        ax.plot(daily_df["pred_date"], daily_df[col], marker="o", ms=4)
        ax.set_title(label)
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    tickers = pd.read_csv(TICKERS_CSV)["ticker"].tolist()
    print(f"Running Prophet inference for {len(tickers)} tickers "
          f"({DATE_FROM} → {DATE_TO})")

    rows = []
    skipped = []

    with sqlite3.connect(DB_PATH) as conn:
        for i, ticker in enumerate(tickers, 1):
            df = pd.read_sql_query(
                "SELECT ticker, date, open, high, low, close, volume "
                "FROM prices WHERE ticker = ? ORDER BY date",
                conn, params=[ticker]
            )
            if df.empty:
                skipped.append(ticker)
                continue

            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
            df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("float32")

            df = add_features(df)

            train_df = df[df["date"] < DATE_FROM].dropna(subset=["close"])
            eval_df = df[(df["date"] >= DATE_FROM) & (df["date"] <= DATE_TO)].dropna(subset=["close"])

            if len(train_df) < 60 or eval_df.empty:
                skipped.append(ticker)
                continue

            try:
                model = fit_prophet(train_df)
                preds = predict_prophet(model, eval_df)
            except Exception as e:
                print(f"  [{i}/{len(tickers)}] {ticker}: FAILED – {e}")
                skipped.append(ticker)
                continue

            actual = eval_df["close"].values.astype(float)
            prev = np.concatenate([[float(train_df["close"].iloc[-1])], actual[:-1]])
            dir_correct = (np.sign(preds - prev) == np.sign(actual - prev)).astype(float)

            for j, row in enumerate(eval_df.itertuples()):
                rows.append({
                    "pred_date": row.date.date(),
                    "ticker": ticker,
                    "model_name": "prophet",
                    "prev_close": float(prev[j]),
                    "actual_close": float(actual[j]),
                    "pred_close": float(preds[j]),
                    "abs_error": float(abs(preds[j] - actual[j])),
                    "directional_correct": float(dir_correct[j]),
                })

            print(f"  [{i}/{len(tickers)}] {ticker}: {len(eval_df)} predictions done")

    if not rows:
        print("No predictions produced. Exiting.")
        return

    rows_df = pd.DataFrame(rows)
    rows_df["pred_date"] = pd.to_datetime(rows_df["pred_date"])

    rows_path = RESULTS_DIR / f"sampled_prophet_{DATE_FROM}_{DATE_TO}_rows.csv"
    rows_df.to_csv(rows_path, index=False)
    print(f"\nRows saved → {rows_path}  ({len(rows_df)} rows, {rows_df['ticker'].nunique()} tickers)")

    daily_df = summarise_daily(rows_df)
    daily_path = RESULTS_DIR / f"sampled_prophet_{DATE_FROM}_{DATE_TO}_daily.csv"
    daily_df.to_csv(daily_path, index=False)
    print(f"Daily saved → {daily_path}")

    plot_path = PLOTS_DIR / f"sampled_prophet_{DATE_FROM}_{DATE_TO}_metrics.png"
    plot_metrics(daily_df, plot_path)
    print(f"Plot saved → {plot_path}")

    # ── update summary CSV ────────────────────────────────────────────────────
    summary_row = {
        "model_name": "prophet",
        "mean_mae": round(daily_df["mae"].mean(), 3),
        "mean_rmse": round(daily_df["rmse"].mean(), 3),
        "mean_directional_acc": round(daily_df["directional_acc"].mean(), 3),
        "mean_mape_pct": round(daily_df["mape_pct"].mean(), 3),
    }
    if SUMMARY_CSV.exists():
        summary = pd.read_csv(SUMMARY_CSV)
        summary = summary[summary["model_name"] != "prophet"]
        summary = pd.concat([summary, pd.DataFrame([summary_row])], ignore_index=True)
    else:
        summary = pd.DataFrame([summary_row])
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"\nSummary row: {summary_row}")
    print(f"Summary saved → {SUMMARY_CSV}")

    if skipped:
        print(f"\nSkipped ({len(skipped)}): {skipped}")


if __name__ == "__main__":
    main()
