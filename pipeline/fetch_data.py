"""
Fetch latest OHLCV data from yfinance for the configured tickers.
Saves data to the pipeline SQLite database.

Usage:
    python -m pipeline.fetch_data
    python -m pipeline.fetch_data --tickers AAPL MSFT GOOGL --lookback 90
"""

import argparse
import os
import sqlite3
import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

DB_PATH = os.environ.get("PIPELINE_DB", "pipeline/db/pipeline.db")
DEFAULT_TICKERS_FILE = "dataset/tickers.csv"

# Default set used when no tickers file or arg provided
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "JPM", "V", "WMT",
    "XOM", "UNH", "PG", "HD", "MA", "CVX", "MRK", "LLY", "ABBV", "PEP",
]


def get_tickers(tickers_arg: list | None) -> list[str]:
    if tickers_arg:
        return [t.upper() for t in tickers_arg]
    if os.path.exists(DEFAULT_TICKERS_FILE):
        df = pd.read_csv(DEFAULT_TICKERS_FILE)
        col = [c for c in df.columns if c.lower() in ("ticker", "symbol")][0]
        return df[col].dropna().str.upper().tolist()
    return FALLBACK_TICKERS


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker TEXT NOT NULL,
            date   TEXT NOT NULL,
            open   REAL,
            high   REAL,
            low    REAL,
            close  REAL,
            volume REAL,
            PRIMARY KEY (ticker, date)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS fetch_log (
            run_at      TEXT NOT NULL,
            status      TEXT NOT NULL,
            tickers_ok  INTEGER,
            tickers_err INTEGER,
            message     TEXT
        )
    """)
    conn.commit()


def fetch_ticker(ticker: str, start: str, end: str, retries: int = 3) -> pd.DataFrame | None:
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
            if df.empty:
                return None
            df = df.reset_index()
            # yfinance returns multi-level columns when multiple tickers; handle both
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ["_".join(c).strip("_") for c in df.columns]
            df.columns = [c.lower().replace(" ", "_") for c in df.columns]
            # Normalise column names
            col_map = {}
            for c in df.columns:
                if "date" in c:
                    col_map[c] = "date"
                elif "open" in c:
                    col_map[c] = "open"
                elif "high" in c:
                    col_map[c] = "high"
                elif "low" in c:
                    col_map[c] = "low"
                elif "close" in c:
                    col_map[c] = "close"
                elif "volume" in c:
                    col_map[c] = "volume"
            df = df.rename(columns=col_map)
            required = {"date", "open", "high", "low", "close", "volume"}
            if not required.issubset(df.columns):
                return None
            df = df[list(required)].copy()
            df["ticker"] = ticker
            df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
            return df
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                print(f"  ERROR {ticker}: {exc}")
                return None
    return None


def upsert_prices(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    rows = df[["ticker", "date", "open", "high", "low", "close", "volume"]].values.tolist()
    conn.executemany(
        """INSERT INTO prices (ticker, date, open, high, low, close, volume)
           VALUES (?,?,?,?,?,?,?)
           ON CONFLICT(ticker, date) DO UPDATE SET
               open=excluded.open, high=excluded.high, low=excluded.low,
               close=excluded.close, volume=excluded.volume""",
        rows,
    )
    return len(rows)


def main(tickers: list[str], lookback: int) -> None:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    init_db(conn)

    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=lookback)).strftime("%Y-%m-%d")
    print(f"Fetching {len(tickers)} tickers | {start_date} → {end_date}")

    ok, err = 0, 0
    for i, ticker in enumerate(tickers, 1):
        df = fetch_ticker(ticker, start_date, end_date)
        if df is not None and len(df) > 0:
            rows = upsert_prices(conn, df)
            ok += 1
            if i % 20 == 0:
                print(f"  [{i}/{len(tickers)}] {ticker}: {rows} rows upserted")
        else:
            err += 1
            if i % 20 == 0:
                print(f"  [{i}/{len(tickers)}] {ticker}: no data")
        conn.commit()

    conn.execute(
        "INSERT INTO fetch_log VALUES (?,?,?,?,?)",
        [datetime.utcnow().isoformat(), "ok", ok, err, f"lookback={lookback}d"],
    )
    conn.commit()
    conn.close()
    print(f"\nDone: {ok} tickers saved, {err} errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--lookback", type=int, default=252, help="Days of history to fetch")
    args = parser.parse_args()
    main(get_tickers(args.tickers), args.lookback)
