import pandas as pd
import numpy as np
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import time

# -----------------------------
# STEP 1: Get S&P 1500 Universe
# -----------------------------

def get_sp1500_tickers():
    page_titles = [
        "List_of_S%26P_500_companies",
        "List_of_S%26P_400_companies",
        "List_of_S%26P_600_companies",
    ]

    tickers = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    session = requests.Session()

    def fetch_html(title: str) -> str:
        urls = [
            f"https://en.wikipedia.org/wiki/{title}",
            f"https://en.wikipedia.org/w/index.php?title={title}&action=render",
        ]
        for url in urls:
            resp = session.get(url, headers=headers, timeout=20)
            if resp.status_code == 200 and resp.text:
                return resp.text
        raise RuntimeError(f"Failed to fetch {title}; last status {resp.status_code}")

    def extract_symbols(html: str) -> list[str]:
        tables = pd.read_html(html)
        for table in tables:
            cols = [str(c).strip() for c in table.columns]
            if "Symbol" in cols:
                return table["Symbol"].tolist()
        raise ValueError("No table with a Symbol column found")

    for title in page_titles:
        html = fetch_html(title)
        tickers.extend(extract_symbols(html))

    tickers = list(set(tickers))
    tickers = [ticker.replace(".", "-") for ticker in tickers]

    return tickers


# -----------------------------
# STEP 2: Download Market Data
# -----------------------------

def get_company_info(tickers):
    data = []
    
    for i, ticker in enumerate(tickers):
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            market_cap = info.get("marketCap", None)
            sector = info.get("sector", None)
            
            hist = stock.history(period="6mo")
            avg_volume = hist["Volume"].mean() if not hist.empty else None
            
            if market_cap and sector and avg_volume:
                data.append({
                    "ticker": ticker,
                    "market_cap": market_cap,
                    "sector": sector,
                    "avg_volume": avg_volume
                })
                
            print(f"{i+1}/{len(tickers)} processed")
            time.sleep(0.1)
            
        except:
            continue
    
    return pd.DataFrame(data)


# -----------------------------
# STEP 3: Apply Filters
# -----------------------------

def filter_companies(df):
    # Minimum market cap: $300M
    df = df[df["market_cap"] > 300_000_000]
    
    # Minimum liquidity: 500k shares daily average
    df = df[df["avg_volume"] > 500_000]
    
    return df


# -----------------------------
# STEP 4: Stratified Sampling
# -----------------------------

def stratified_sample(df, n=1000):
    sector_counts = df["sector"].value_counts(normalize=True)
    
    selected = []
    
    for sector, proportion in sector_counts.items():
        sector_df = df[df["sector"] == sector]
        sample_size = int(proportion * n)
        
        sampled = sector_df.sample(
            n=min(sample_size, len(sector_df)),
            random_state=42
        )
        
        selected.append(sampled)
    
    result = pd.concat(selected)
    
    # If under 1000 due to rounding, fill randomly
    if len(result) < n:
        remaining = df[~df["ticker"].isin(result["ticker"])]
        extra = remaining.sample(n=n-len(result), random_state=42)
        result = pd.concat([result, extra])
    
    return result.head(n)


