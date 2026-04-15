"""
Gemini 2.5 Flash-based stock prediction.

The approach: for each test day, send the preceding N days of OHLCV data
to Gemini 2.5 Flash with a structured prompt and ask it to predict
tomorrow's closing price. Compute R², RMSE, MAE, directional accuracy.

Cross-validation is done as walk-forward backtesting: each "fold" is a
consecutive time window, always predicting out-of-sample.

Usage:
    export GEMINI_API_KEY="your_key_here"
    python -m src.llm.test_llm

    # Predict the 61st day close from the previous 60 trading days,
    # making 10 Gemini calls per ticker:
    python -m src.llm.test_llm --tickers AAPL --context_days 60 --n_eval 10
"""

import argparse
import ast
import json
import os
import re
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from google import genai
from google.genai import types
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

    model_name: str = "gemini-2.5-flash"

    # How many past trading days to include in the prompt
    context_days: int = 60

    # Number of test samples per ticker (walk-forward)
    n_eval_per_ticker: int = 10

    # Number of tickers to evaluate
    max_tickers: int = 20

    # CV folds (time-based walk-forward windows)
    n_splits: int = 5

    # Rate limiting: seconds between API calls
    request_delay: float = 1.0

    # Retry config
    max_retries: int = 3
    retry_delay: float = 5.0

    # Test split
    test_size: float = 0.2

    random_state: int = 42

    output_path: str = f"results/{month}-{day}-{tag}-gemini_predictions.csv"
    config_output_path: str = f"artifacts/{month}-{day}-{tag}-gemini_config.json"


def load_prices_from_sqlite(db_path: str, table_name: str, tickers: Optional[List[str]] = None) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    if tickers:
        placeholders = ",".join("?" * len(tickers))
        query = f"""
            SELECT ticker, date, open, high, low, close, volume
            FROM {table_name}
            WHERE ticker IN ({placeholders})
            ORDER BY ticker, date
        """
        df = pd.read_sql_query(query, conn, params=tickers)
    else:
        query = f"""
            SELECT ticker, date, open, high, low, close, volume
            FROM {table_name}
            ORDER BY ticker, date
        """
        df = pd.read_sql_query(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("float64")
    return df


def build_prompt(ticker: str, history: pd.DataFrame, predict_date: str) -> str:
    """Build a structured prompt from recent OHLCV history."""
    lines = []
    for _, row in history.iterrows():
        vol_k = row["volume"] / 1000
        lines.append(
            f"  {row['date'].strftime('%Y-%m-%d')}: "
            f"Open={row['open']:.2f}  High={row['high']:.2f}  "
            f"Low={row['low']:.2f}  Close={row['close']:.2f}  "
            f"Volume={vol_k:.0f}K"
        )

    history_text = "\n".join(lines)
    last_close = history.iloc[-1]["close"]

    return f"""You are a quantitative financial analyst. Your task is to predict the next trading day closing price for {ticker}.

Recent price history (last {len(history)} trading days):
{history_text}

Last known closing price: {last_close:.2f}

Predict the closing price for {ticker} on {predict_date}.

Rules:
- Reply with ONLY a single number (the predicted closing price), no explanation, no units, no text.
- The number must be a positive float, e.g.: 143.72
"""


def call_gemini(client: genai.Client, model_name: str, prompt: str, cfg: Config) -> Optional[float]:
    """Call Gemini API and parse the numeric response."""
    for attempt in range(cfg.max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=32,
                ),
            )

            text = (getattr(response, "text", None) or "").strip()

            if not text:
                # Some responses don't populate .text directly; fall back to a
                # string form so we can still attempt numeric extraction.
                text = str(response).strip()

            if not text:
                return None

            cleaned = text.replace(",", "").replace("$", "")

            # Prefer plain numeric replies, but fall back to the last positive
            # float if Gemini includes extra formatting or explanation.
            candidates = re.findall(r"(?<!\w)(\d+(?:\.\d+)?)(?!\w)", cleaned)
            if candidates:
                val = float(candidates[-1])
                if val > 0:
                    return val

            try:
                literal_val = ast.literal_eval(cleaned)
                if isinstance(literal_val, (int, float)) and literal_val > 0:
                    return float(literal_val)
            except (SyntaxError, ValueError):
                pass

            return None
        except Exception as e:
            if attempt < cfg.max_retries - 1:
                print(f"    Retry {attempt + 1}/{cfg.max_retries}: {e}")
                time.sleep(cfg.retry_delay)
            else:
                print(f"    Failed after {cfg.max_retries} attempts: {e}")
                return None


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "directional_accuracy": np.nan, "n": len(y_true)}

    # Directional accuracy: did model correctly predict up/down vs prior day?
    dir_true = np.sign(np.diff(y_true, prepend=y_true[0]))
    dir_pred = np.sign(y_pred - np.concatenate([[y_true[0]], y_true[:-1]]))
    da = float(np.mean(dir_true[1:] == dir_pred[1:]))

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": da,
        "n": int(len(y_true)),
    }


def build_cv_fold_indices(n: int, n_splits: int, context_days: int, test_size: float) -> List[List[int]]:
    """Walk-forward fold indices within the test region."""
    test_start = int(n * (1 - test_size))
    test_rows = list(range(max(test_start, context_days), n))
    if not test_rows:
        return []
    chunks = np.array_split(test_rows, n_splits)
    return [list(c) for c in chunks if len(c) > 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--n_eval", type=int, default=None)
    parser.add_argument("--max_tickers", type=int, default=None)
    parser.add_argument("--context_days", type=int, default=None)
    parser.add_argument("--n_splits", type=int, default=None)
    parser.add_argument("--request_delay", type=float, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config()

    if args.n_eval is not None:
        cfg.n_eval_per_ticker = args.n_eval
    if args.max_tickers is not None:
        cfg.max_tickers = args.max_tickers
    if args.context_days is not None:
        cfg.context_days = args.context_days
    if args.n_splits is not None:
        cfg.n_splits = args.n_splits
    if args.request_delay is not None:
        cfg.request_delay = args.request_delay

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GEMINI_API_KEY environment variable not set.\n"
            "Set it with: export GEMINI_API_KEY='your_key_here'"
        )

    client = genai.Client(api_key=api_key)
    print(f"Using model: {cfg.model_name}")
    print(
        f"Configuration: context_days={cfg.context_days}, "
        f"n_eval={cfg.n_eval_per_ticker}, n_splits={cfg.n_splits}, "
        f"request_delay={cfg.request_delay}"
    )

    print("Loading price data...")
    raw_df = load_prices_from_sqlite(cfg.db_path, cfg.table_name, tickers=args.tickers)

    all_tickers = raw_df["ticker"].unique()
    if args.tickers is None:
        rng = np.random.default_rng(cfg.random_state)
        selected = rng.choice(all_tickers, size=min(cfg.max_tickers, len(all_tickers)), replace=False)
    else:
        selected = np.array(args.tickers)

    print(f"Evaluating {len(selected)} tickers: {', '.join(selected)}")

    # === Walk-forward cross-validation ===
    print(f"\nRunning {cfg.n_splits}-fold walk-forward CV...\n")

    fold_results: List[List[dict]] = [[] for _ in range(cfg.n_splits)]

    for ticker in selected:
        t_df = raw_df[raw_df["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        n = len(t_df)
        if n < cfg.context_days + 10:
            print(f"  Skipping {ticker}: only {n} rows")
            continue

        fold_indices = build_cv_fold_indices(n, cfg.n_splits, cfg.context_days, cfg.test_size)
        n_per_fold = max(1, cfg.n_eval_per_ticker // cfg.n_splits)

        for fold_i, eval_idx in enumerate(fold_indices):
            # Evenly subsample within fold
            if len(eval_idx) > n_per_fold:
                step = len(eval_idx) // n_per_fold
                eval_idx = eval_idx[::step][:n_per_fold]

            print(f"  [{ticker}] Fold {fold_i + 1}/{cfg.n_splits} — {len(eval_idx)} points")

            for idx in eval_idx:
                if idx < cfg.context_days or idx >= n:
                    continue

                history = t_df.iloc[idx - cfg.context_days : idx]
                row = t_df.iloc[idx]
                predict_date = row["date"].strftime("%Y-%m-%d")
                actual = float(row["close"])

                if np.isnan(actual):
                    continue

                prompt = build_prompt(ticker, history, predict_date)
                pred = call_gemini(client, cfg.model_name, prompt, cfg)
                time.sleep(cfg.request_delay)

                if pred is not None:
                    fold_results[fold_i].append({
                        "ticker": ticker,
                        "date": predict_date,
                        "close": actual,
                        "y_true": actual,
                        "y_pred": pred,
                    })
                    print(f"    {predict_date}: actual={actual:.2f}  pred={pred:.2f}  diff={pred - actual:+.2f}")

    # === Per-fold metrics ===
    print("\n=== CV Results ===")
    cv_metrics = []
    for fold_i, rows in enumerate(fold_results):
        if not rows:
            continue
        df_fold = pd.DataFrame(rows)
        m = regression_metrics(df_fold["y_true"].values, df_fold["y_pred"].values)
        cv_metrics.append(m)
        print(
            f"Fold {fold_i + 1} | n={m['n']}  "
            f"RMSE={m['rmse']:.4f}  MAE={m['mae']:.4f}  "
            f"R²={m['r2']:.4f}  DA={m['directional_accuracy']:.4f}"
        )

    cv_df = pd.DataFrame(cv_metrics) if cv_metrics else pd.DataFrame()
    if not cv_df.empty:
        print("\n=== CV Mean ===")
        print(cv_df.mean(numeric_only=True).to_string())

    all_rows = [r for fold in fold_results for r in fold]
    if not all_rows:
        print("No predictions collected.")
        return

    all_df = pd.DataFrame(all_rows)
    overall = regression_metrics(all_df["y_true"].values, all_df["y_pred"].values)
    print(f"\n=== Overall ===")
    print(
        f"n={overall['n']}  RMSE={overall['rmse']:.4f}  MAE={overall['mae']:.4f}  "
        f"R²={overall['r2']:.4f}  DA={overall['directional_accuracy']:.4f}"
    )

    os.makedirs("results", exist_ok=True)
    os.makedirs("artifacts", exist_ok=True)

    all_df.to_csv(cfg.output_path, index=False)

    cv_r2 = f"{cv_df['r2'].mean():.4f}" if not cv_df.empty else "N/A"
    result_txt = f"""Gemini 2.5 Flash Stock Prediction Results:
Model: {cfg.model_name}
Tickers: {len(selected)}
Context days: {cfg.context_days}

Overall:
  RMSE={overall['rmse']:.4f}
  MAE={overall['mae']:.4f}
  R2={overall['r2']:.4f}
  DA={overall['directional_accuracy']:.4f}
  n={overall['n']}

CV Mean R2={cv_r2}
"""
    with open(f"{cfg.output_path}.txt", "w") as f:
        f.write(result_txt)

    with open(cfg.config_output_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print(f"\nSaved predictions to: {cfg.output_path}")


if __name__ == "__main__":
    main()
