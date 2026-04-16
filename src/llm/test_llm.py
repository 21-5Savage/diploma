"""
Gemini 2.5 Flash-based stock prediction (structured output version).
"""

import argparse
import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import datetime
import random
import string

today = datetime.date.today()
month = today.month
day = today.day
tag = "".join(random.choices(string.ascii_lowercase, k=3))


class PricePrediction(BaseModel):
    predicted_close: float


@dataclass
class Config:
    db_path: str = "dataset/stock_prices_20y.db"
    table_name: str = "prices"
    model_name: str = "gemini-2.5-flash"

    context_days: int = 60
    n_eval_per_ticker: int = 10
    max_tickers: int = 30
    n_splits: int = 5
    request_delay: float = 1.0

    max_retries: int = 3
    retry_delay: float = 5.0

    test_size: float = 0.2
    random_state: int = 42

    date_from: Optional[str] = None
    date_to: Optional[str] = None

    output_path: str = f"results/{month}-{day}-{tag}-gemini_predictions.csv"
    config_output_path: str = f"artifacts/{month}-{day}-{tag}-gemini_config.json"


# =========================
# Data loading
# =========================
def load_prices_from_sqlite(db_path, table_name, tickers=None):
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
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0)
    return df.sort_values(["ticker", "date"]).reset_index(drop=True)


# =========================
# Prompt
# =========================
def build_prompt(ticker, history, predict_date):
    lines = []
    for _, row in history.iterrows():
        vol_k = row["volume"] / 1000
        lines.append(
            f"{row['date'].strftime('%Y-%m-%d')}: "
            f"O={row['open']:.2f} H={row['high']:.2f} "
            f"L={row['low']:.2f} C={row['close']:.2f} V={vol_k:.0f}K"
        )

    history_text = "\n".join(lines)
    last_close = history.iloc[-1]["close"]

    return f"""
You are a quantitative financial analyst.

Predict the next trading day closing price for {ticker}.

Recent history:
{history_text}

Last close: {last_close:.2f}
Target date: {predict_date}

RULES:
- you are forbidden from guessing same as last close price.
Return JSON:
{{
  "predicted_close": float
}}
"""


# =========================
# Gemini call (STRUCTURED)
# =========================
def call_gemini(client, model_name, prompt, cfg, last_close):
    for attempt in range(cfg.max_retries):
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                    response_schema=PricePrediction,
                ),
            )

            parsed = getattr(response, "parsed", None)

            if parsed:
                val = (
                    parsed.predicted_close
                    if isinstance(parsed, PricePrediction)
                    else parsed["predicted_close"]
                )

                val = float(val)

                # sanity check
                if val <= 0:
                    return None
                if val > last_close * 5 or val < last_close * 0.2:
                    return None

                return val

            # fallback JSON parse
            text = (getattr(response, "text", "") or "").strip()
            if text:
                data = json.loads(text)
                val = float(data["predicted_close"])
                return val

            return None

        except Exception as e:
            if attempt < cfg.max_retries - 1:
                print(f"Retry {attempt + 1}: {e}")
                time.sleep(cfg.retry_delay)
            else:
                print(f"Failed: {e}")
                return None


# =========================
# Metrics
# =========================
def regression_metrics(y_true, y_pred):
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]

    if len(y_true) < 2:
        return {}

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="+", default=None)
    parser.add_argument("--date-from", default=None)
    parser.add_argument("--date-to", default=None)
    parser.add_argument("--context-days", type=int, default=None)
    parser.add_argument("--n-eval", type=int, default=None)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--request-delay", type=float, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--retry-delay", type=float, default=None)
    return parser.parse_args()


# =========================
# Main
# =========================
def main():
    args = parse_args()
    cfg = Config()
    if args.context_days is not None:
        cfg.context_days = args.context_days
    if args.n_eval is not None:
        cfg.n_eval_per_ticker = args.n_eval
    if args.max_tickers is not None:
        cfg.max_tickers = args.max_tickers
    if args.request_delay is not None:
        cfg.request_delay = args.request_delay
    if args.max_retries is not None:
        cfg.max_retries = args.max_retries
    if args.retry_delay is not None:
        cfg.retry_delay = args.retry_delay
    cfg.date_from = args.date_from
    cfg.date_to = args.date_to

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Missing GEMINI_API_KEY")

    client = genai.Client(api_key=api_key)

    print("Loading data...")
    df = load_prices_from_sqlite(cfg.db_path, cfg.table_name, tickers=args.tickers)

    if args.tickers:
        tickers = np.array(args.tickers)
    else:
        tickers = df["ticker"].unique()[: cfg.max_tickers]

    print(
        f"Running Gemini on {len(tickers)} tickers | "
        f"context_days={cfg.context_days} | request_delay={cfg.request_delay}"
    )
    if cfg.date_from or cfg.date_to:
        print(f"Date window: {cfg.date_from or 'start'} -> {cfg.date_to or 'end'}")

    results = []

    for ticker in tickers:
        t_df = df[df["ticker"] == ticker].reset_index(drop=True)

        valid_idx = []
        for i in range(cfg.context_days, len(t_df)):
            row_date = t_df.iloc[i]["date"].strftime("%Y-%m-%d")
            if cfg.date_from and row_date < cfg.date_from:
                continue
            if cfg.date_to and row_date > cfg.date_to:
                continue
            valid_idx.append(i)

        if not valid_idx:
            continue

        if cfg.n_eval_per_ticker > 0 and len(valid_idx) > cfg.n_eval_per_ticker:
            step = max(1, len(valid_idx) // cfg.n_eval_per_ticker)
            valid_idx = valid_idx[::step][: cfg.n_eval_per_ticker]

        print(f"[{ticker}] evaluating {len(valid_idx)} points")

        for i in valid_idx:
            history = t_df.iloc[i - cfg.context_days : i]
            row = t_df.iloc[i]

            actual = float(row["close"])
            prev_close = float(history.iloc[-1]["close"])

            prompt = build_prompt(ticker, history, row["date"])

            pred = call_gemini(
                client, cfg.model_name, prompt, cfg, prev_close
            )

            time.sleep(cfg.request_delay)

            if pred is None:
                continue

            print(
                f"{ticker} {row['date'].date()} | "
                f"actual={actual:.2f} pred={pred:.2f}"
            )

            results.append(
                {
                    "ticker": ticker,
                    "date": row["date"],
                    "prev_close": prev_close,
                    "y_true": actual,
                    "y_pred": pred,
                }
            )

    df_res = pd.DataFrame(results)
    if df_res.empty:
        print("No predictions collected.")
        return

    metrics = regression_metrics(
        df_res["y_true"].values, df_res["y_pred"].values
    )

    print("\nMetrics:", metrics)

    os.makedirs("results", exist_ok=True)
    df_res.to_csv(cfg.output_path, index=False)

    with open(cfg.config_output_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("Saved:", cfg.output_path)


if __name__ == "__main__":
    main()
