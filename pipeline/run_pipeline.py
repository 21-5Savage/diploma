"""
Daily pipeline orchestrator.

Runs every trading day (Mon–Fri):
  1. Fetch latest OHLCV data (yfinance)
  2. Run predictions from all available models
  3. Evaluate prior predictions against actual prices
  4. Write a daily summary report

Usage:
    python -m pipeline.run_pipeline
    python -m pipeline.run_pipeline --skip-fetch
"""

import argparse
import os
import sys
from datetime import datetime

# Ensure the workspace root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.fetch_data import get_tickers, main as fetch_main
from pipeline.predict import run_predictions
from pipeline.evaluate import main as evaluate_main


def is_trading_day() -> bool:
    """Mon–Fri are potential trading days (holidays not checked)."""
    return datetime.today().weekday() < 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the daily stock prediction pipeline")
    parser.add_argument("--skip-fetch", action="store_true", help="Skip data fetch step")
    parser.add_argument("--force", action="store_true", help="Run even on weekends")
    parser.add_argument("--lookback", type=int, default=90,
                        help="Days of history to fetch (default 90)")
    parser.add_argument("--models", nargs="*", default=["lstm", "rnn", "tree"],
                        choices=["lstm", "rnn", "tree", "xgb", "prophet"])
    args = parser.parse_args()

    if not args.force and not is_trading_day():
        print(f"{datetime.today().strftime('%A')} is not a trading day. Use --force to override.")
        sys.exit(0)

    print(f"\n{'='*60}")
    print(f" Stock Prediction Pipeline  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    # Step 1 — Fetch data
    if not args.skip_fetch:
        print("\n[1/3] Fetching latest price data...")
        tickers = get_tickers(None)
        fetch_main(tickers, lookback=args.lookback)
    else:
        print("\n[1/3] Fetch skipped")

    # Step 2 — Predict
    print("\n[2/3] Running predictions...")
    run_predictions(args.models)

    # Step 3 — Evaluate
    print("\n[3/3] Evaluating predictions...")
    evaluate_main()

    print(f"\n{'='*60}")
    print(" Pipeline complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
