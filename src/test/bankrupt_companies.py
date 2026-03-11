from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL_TEMPLATE = "https://data.sec.gov/submissions/CIK{cik}.json"

BANKRUPTCY_FORMS = {"8-K", "8-K/A"}
BANKRUPTCY_ITEM_TOKEN = "1.03"


@dataclass
class BankruptcySignal:
    ticker: str
    cik: str
    filing_date: str
    form: str
    item: str
    accession_number: str

    @property
    def evidence_url(self) -> str:
        # SEC submissions JSON is the authoritative source used for this label.
        return SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=self.cik)


class SecBankruptcyLabeler:
    def __init__(self, user_agent: str, sleep_seconds: float = 0.2, timeout_seconds: int = 30):
        self.sleep_seconds = sleep_seconds
        self.timeout_seconds = timeout_seconds
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": user_agent,
                "Accept-Encoding": "gzip, deflate",
            }
        )
        self._ticker_to_cik: dict[str, str] | None = None

    def _sleep(self) -> None:
        if self.sleep_seconds > 0:
            time.sleep(self.sleep_seconds)

    def _fetch_json(self, url: str) -> dict[str, Any]:
        response = self.session.get(url, timeout=self.timeout_seconds)
        response.raise_for_status()
        self._sleep()
        data: dict[str, Any] = response.json()
        return data

    def load_ticker_map(self) -> dict[str, str]:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik

        payload = self._fetch_json(SEC_TICKER_MAP_URL)
        mapping: dict[str, str] = {}

        for record in payload.values():
            ticker = str(record.get("ticker", "")).strip().upper()
            cik_number = record.get("cik_str")
            if ticker and cik_number is not None:
                mapping[ticker] = f"{int(cik_number):010d}"

        self._ticker_to_cik = mapping
        return mapping

    def get_cik_for_ticker(self, ticker: str) -> str | None:
        mapping = self.load_ticker_map()
        return mapping.get(ticker.upper())

    def find_bankruptcy_signal(self, ticker: str) -> BankruptcySignal | None:
        cik = self.get_cik_for_ticker(ticker)
        if cik is None:
            return None

        # Primary SEC endpoint uses 10-digit zero-padded CIK, but try non-padded
        # fallback in case of endpoint variance.
        try:
            payload = self._fetch_json(SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik))
        except requests.HTTPError as exc:
            status = exc.response.status_code if exc.response is not None else None
            if status != 404:
                raise
            non_padded_cik = str(int(cik))
            payload = self._fetch_json(
                SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=non_padded_cik)
            )
        filings = payload.get("filings", {}).get("recent", {})

        forms = filings.get("form", [])
        items = filings.get("items", [])
        filing_dates = filings.get("filingDate", [])
        accessions = filings.get("accessionNumber", [])

        limit = min(len(forms), len(items), len(filing_dates), len(accessions))

        earliest_signal: BankruptcySignal | None = None

        for idx in range(limit):
            form = str(forms[idx] or "").strip().upper()
            item = str(items[idx] or "").strip()
            filing_date = str(filing_dates[idx] or "").strip()
            accession_number = str(accessions[idx] or "").strip()

            if form not in BANKRUPTCY_FORMS:
                continue

            # Item 1.03 = Bankruptcy or Receivership.
            if BANKRUPTCY_ITEM_TOKEN not in item:
                continue

            signal = BankruptcySignal(
                ticker=ticker,
                cik=cik,
                filing_date=filing_date,
                form=form,
                item=item,
                accession_number=accession_number,
            )

            if earliest_signal is None or signal.filing_date < earliest_signal.filing_date:
                earliest_signal = signal

        return earliest_signal


def add_horizon_months(date_series: pd.Series, months: int) -> pd.Series:
    return pd.to_datetime(date_series, errors="coerce") + pd.DateOffset(months=months)


def build_labels(
    input_csv: Path,
    output_csv: Path,
    user_agent: str,
    horizon_months: int = 24,
    sleep_seconds: float = 0.2,
) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    if "ticker" not in df.columns:
        raise ValueError(f"Input CSV must contain a 'ticker' column: {input_csv}")

    df = df.copy()
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()

    unique_tickers = [t for t in df["ticker"].dropna().unique() if t]

    labeler = SecBankruptcyLabeler(user_agent=user_agent, sleep_seconds=sleep_seconds)

    signal_by_ticker: dict[str, BankruptcySignal] = {}

    total = len(unique_tickers)
    for idx, ticker in enumerate(unique_tickers, start=1):
        try:
            signal = labeler.find_bankruptcy_signal(ticker)
            if signal is not None:
                signal_by_ticker[ticker] = signal
            print(f"{idx}/{total} checked: {ticker}")
        except requests.HTTPError as exc:
            print(f"{idx}/{total} SEC HTTP error for {ticker}: {exc}")
        except requests.RequestException as exc:
            print(f"{idx}/{total} SEC request error for {ticker}: {exc}")
        except Exception as exc:
            print(f"{idx}/{total} unexpected error for {ticker}: {exc}")

    def signal_value(ticker: str, field: str) -> str:
        signal = signal_by_ticker.get(ticker)
        if signal is None:
            return ""
        return str(getattr(signal, field))

    df["cik"] = df["ticker"].apply(lambda t: signal_value(t, "cik"))
    df["bankruptcy_filing_date"] = df["ticker"].apply(
        lambda t: signal_value(t, "filing_date")
    )
    df["bankruptcy_form"] = df["ticker"].apply(lambda t: signal_value(t, "form"))
    df["bankruptcy_item"] = df["ticker"].apply(lambda t: signal_value(t, "item"))
    df["accession_number"] = df["ticker"].apply(
        lambda t: signal_value(t, "accession_number")
    )

    df["bankrupt"] = df["bankruptcy_filing_date"].astype(bool).astype(int)

    # If as_of_date exists, label bankruptcy within a forward window.
    if "as_of_date" in df.columns:
        as_of_dt = pd.to_datetime(df["as_of_date"], errors="coerce")
        bankruptcy_dt = pd.to_datetime(df["bankruptcy_filing_date"], errors="coerce")
        horizon_dt = add_horizon_months(df["as_of_date"], horizon_months)

        within_window = (
            bankruptcy_dt.notna()
            & as_of_dt.notna()
            & (bankruptcy_dt > as_of_dt)
            & (bankruptcy_dt <= horizon_dt)
        )
        df["label"] = within_window.astype(int)
    else:
        # Fallback: label equals observed bankrupt flag at ticker level.
        df["label"] = df["bankrupt"]

    df["evidence_source"] = "SEC submissions (8-K Item 1.03)"
    df["evidence_url"] = df["cik"].apply(
        lambda cik: SEC_SUBMISSIONS_URL_TEMPLATE.format(cik=cik) if cik else ""
    )

    df.to_csv(output_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Label historical ticker rows with bankruptcy outcomes using SEC 8-K Item 1.03 signals."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("dataset/tickers.csv"),
        help="Input CSV (required column: ticker; optional: as_of_date)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("dataset/labelled_tickers.csv"),
        help="Output labeled CSV path",
    )
    parser.add_argument(
        "--horizon-months",
        type=int,
        default=24,
        help="Forward label horizon in months when as_of_date exists",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Delay between SEC requests in seconds",
    )
    parser.add_argument(
        "--user-agent",
        type=str,
        default="",
        help=(
            "SEC-compliant User-Agent, e.g. 'Your Name your.email@domain.com'. "
            "If empty, SEC_USER_AGENT env var is used."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    user_agent = args.user_agent.strip() or os.getenv("SEC_USER_AGENT", "").strip()
    if not user_agent:
        raise ValueError(
            "Provide SEC User-Agent via --user-agent or SEC_USER_AGENT env var."
        )

    labelled_df = build_labels(
        input_csv=args.input,
        output_csv=args.output,
        user_agent=user_agent,
        horizon_months=args.horizon_months,
        sleep_seconds=args.sleep,
    )

    bankrupt_rows = int(labelled_df["bankrupt"].sum())
    positive_rows = int(labelled_df["label"].sum())

    print(f"Done. Rows: {len(labelled_df)}")
    print(f"Bankrupt rows (observed): {bankrupt_rows}")
    print(f"Positive labels: {positive_rows}")
    print(f"Saved results to: {args.output}")


if __name__ == "__main__":
    main()
