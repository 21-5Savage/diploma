
import numpy as np
import pandas as pd


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    return (100.0 - 100.0 / (1.0 + gain / (loss + 1e-8))) / 100.0 - 0.5


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_c = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_c).abs(), (low - prev_c).abs()], axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()


FEATURE_COLS = [
    "log_ret_1", "log_ret_3", "log_ret_5", "log_ret_10", "log_ret_20",
    "hl_range", "oc_change",
    "close_vs_ma_5", "close_vs_ma_10", "close_vs_ma_20", "close_vs_ma_50",
    "ma_5_vs_ma_20",
    "rolling_vol_5", "rolling_vol_20",
    "log_vol_chg_1", "vol_vs_ma_5", "vol_vs_ma_20",
    "rsi_14",
    "macd_norm",
    "bb_pos",
    "atr_14_norm",
]


def compute_features_for_ticker(g: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full feature set for a single ticker DataFrame.
    DataFrame must have columns: date, open, high, low, close, volume.
    Returns the same DataFrame with feature columns added.
    """
    g = g.sort_values("date").copy()
    close = g["close"].astype("float64")
    high = g["high"].astype("float64")
    low = g["low"].astype("float64")
    open_ = g["open"].astype("float64")
    volume = g["volume"].astype("float64").clip(lower=1.0)

    log_c = np.log(close)
    g["log_ret_1"] = log_c.diff(1)
    g["log_ret_3"] = log_c.diff(3)
    g["log_ret_5"] = log_c.diff(5)
    g["log_ret_10"] = log_c.diff(10)
    g["log_ret_20"] = log_c.diff(20)

    g["hl_range"] = (high - low) / close
    g["oc_change"] = (close - open_) / open_.clip(lower=1e-6)

    ma5 = close.rolling(5).mean()
    ma10 = close.rolling(10).mean()
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    g["close_vs_ma_5"] = close / ma5.clip(lower=1e-6) - 1.0
    g["close_vs_ma_10"] = close / ma10.clip(lower=1e-6) - 1.0
    g["close_vs_ma_20"] = close / ma20.clip(lower=1e-6) - 1.0
    g["close_vs_ma_50"] = close / ma50.clip(lower=1e-6) - 1.0
    g["ma_5_vs_ma_20"] = ma5 / ma20.clip(lower=1e-6) - 1.0

    g["rolling_vol_5"] = g["log_ret_1"].rolling(5).std()
    g["rolling_vol_20"] = g["log_ret_1"].rolling(20).std()

    log_vol = np.log1p(volume)
    g["log_vol_chg_1"] = log_vol.diff(1)
    g["vol_vs_ma_5"] = volume / volume.rolling(5).mean().clip(lower=1.0) - 1.0
    g["vol_vs_ma_20"] = volume / volume.rolling(20).mean().clip(lower=1.0) - 1.0

    g["rsi_14"] = _compute_rsi(close, 14)

    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    g["macd_norm"] = (ema12 - ema26) / close.clip(lower=1e-6)

    bb_mean = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    g["bb_pos"] = (close - (bb_mean - 2 * bb_std)) / (4 * bb_std).clip(lower=1e-6) - 0.5

    g["atr_14_norm"] = _compute_atr(high, low, close, 14) / close.clip(lower=1e-6)

    g = g.replace([np.inf, -np.inf], np.nan)
    return g


def make_features_df(df: pd.DataFrame) -> pd.DataFrame:
    """Process a multi-ticker DataFrame; returns enriched DataFrame."""
    out = []
    for _, g in df.groupby("ticker", sort=False):
        out.append(compute_features_for_ticker(g))
    if not out:
        return df
    return pd.concat(out, ignore_index=True)
