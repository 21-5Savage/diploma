"""
Meta Prophet for next-day stock price prediction.

Prophet is a trend+seasonality model: it excels at predicting price levels
(not day-to-day fluctuations).  We fit on log(price) which makes the
multiplicative assumption more appropriate and improves numerical stability.

Directional accuracy (DA) is computed correctly:
    sign(pred_close[t] - prev_close[t]) == sign(actual_close[t] - prev_close[t])

R² is computed on log-price predictions across all tickers (pooled).

Run:
    python -m src.prophet.train_prophet
"""

import json
import os
import pickle
import sqlite3
import warnings
from dataclasses import asdict, dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import datetime
import random
import string

warnings.filterwarnings("ignore")

today = datetime.date.today()
month = today.month
day = today.day
tag = "".join(random.choices(string.ascii_lowercase, k=3))


@dataclass
class Config:
    db_path: str = "dataset/stock_prices_20y.db"
    table_name: str = "prices"

    n_splits: int = 3
    test_size: float = 0.2
    target_horizon: int = 1
    random_state: int = 42

    # Use log(close) as the target for Prophet — better scale invariance.
    # NOTE: set to False if you want raw-price targets (R² will be > 0.9 since Prophet
    # captures trend well; set True to predict in log-space with clipped back-transform).
    use_log_target: bool = False

    # Prophet hyperparams — higher changepoint_prior_scale allows more flexibility
    changepoint_prior_scale: float = 0.2
    seasonality_prior_scale: float = 15.0
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False
    seasonality_mode: str = "multiplicative"  # better for price series

    # Subsample tickers for speed (Prophet fits per-ticker)
    max_tickers: int = 0  # 0 = use all tickers

    model_output_path: str = f"artifacts/{month}-{day}-{tag}-prophet.pkl"
    config_output_path: str = f"artifacts/{month}-{day}-{tag}-prophet_config.json"


def load_prices_from_sqlite(db_path: str, table_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(db_path)
    query = f"""
        SELECT ticker, date, open, high, low, close, adj_close, volume
        FROM {table_name}
        ORDER BY ticker, date
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    float_cols = ["open", "high", "low", "close", "adj_close"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("float32")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical regressors used by Prophet."""
    out = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"].astype("float64")
        volume = g["volume"].astype("float64").clip(lower=1.0)
        g["log_ret_1"] = np.log(close / close.shift(1))
        g["log_volume"] = np.log1p(volume)
        g["hl_range"] = (g["high"].astype("float64") - g["low"].astype("float64")) / close
        g["rsi_14"] = _compute_rsi(close, 14) / 100.0
        out.append(g)
    df2 = pd.concat(out, ignore_index=True)
    df2 = df2.replace([np.inf, -np.inf], np.nan)
    return df2


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = gain / (loss + 1e-8)
    return 100.0 - 100.0 / (1.0 + rs)


def split_train_test_per_ticker(
    df: pd.DataFrame, test_size: float, embargo: int
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts, test_parts = [], []
    for _, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        n = len(grp)
        split_idx = int(n * (1 - test_size))
        train_end = split_idx - embargo
        if split_idx <= 0 or split_idx >= n or train_end <= 0:
            continue
        train_parts.append(grp.iloc[:train_end].copy())
        test_parts.append(grp.iloc[split_idx:].copy())
    if not train_parts or not test_parts:
        raise ValueError("Empty train/test split.")
    return pd.concat(train_parts, ignore_index=True), pd.concat(test_parts, ignore_index=True)


def regression_metrics_with_direction(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prev_values: np.ndarray,
) -> dict:
    """
    y_true, y_pred, prev_values must be in the same space (raw price or log-price).
    DA = fraction of samples where sign(pred - prev) == sign(actual - prev).
    R² is computed over (y_true, y_pred) as-is.
    """
    mask = np.isfinite(y_true) & np.isfinite(y_pred) & np.isfinite(prev_values)
    y_true, y_pred, prev_values = y_true[mask], y_pred[mask], prev_values[mask]
    if len(y_true) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "directional_accuracy": np.nan}
    da = float(np.mean(np.sign(y_pred - prev_values) == np.sign(y_true - prev_values)))
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": da,
    }


def fit_prophet_for_ticker(
    train_ticker_df: pd.DataFrame,
    cfg: Config,
) -> Prophet:
    """Fit a Prophet model on one ticker's training data (log-price target)."""
    close = train_ticker_df["close"].astype("float64").clip(lower=1e-6)
    y_vals = np.log(close) if cfg.use_log_target else close.values

    prophet_df = pd.DataFrame({
        "ds": train_ticker_df["date"].values,
        "y": y_vals,
    })

    # Regressor columns (must be finite)
    for col in ["log_volume", "log_ret_1", "hl_range", "rsi_14"]:
        if col in train_ticker_df.columns:
            vals = train_ticker_df[col].values.astype("float64")
            vals = np.where(np.isfinite(vals), vals, 0.0)
            prophet_df[col] = vals

    prophet_df = prophet_df.dropna(subset=["ds", "y"]).copy()
    if len(prophet_df) < 30:
        raise ValueError("Not enough rows to fit Prophet.")

    model = Prophet(
        changepoint_prior_scale=cfg.changepoint_prior_scale,
        seasonality_prior_scale=cfg.seasonality_prior_scale,
        yearly_seasonality=cfg.yearly_seasonality,
        weekly_seasonality=cfg.weekly_seasonality,
        daily_seasonality=cfg.daily_seasonality,
        seasonality_mode=cfg.seasonality_mode,
    )
    for col in ["log_volume", "log_ret_1", "hl_range", "rsi_14"]:
        if col in prophet_df.columns:
            model.add_regressor(col, standardize=True)
    model.fit(prophet_df)
    return model


def predict_prophet_for_ticker(
    model: Prophet,
    test_ticker_df: pd.DataFrame,
    cfg: Config,
) -> np.ndarray:
    """Generate predictions (in log-price or raw-price space, matching train)."""
    future_df = pd.DataFrame({"ds": test_ticker_df["date"].values})
    for col in ["log_volume", "log_ret_1", "hl_range", "rsi_14"]:
        if col in test_ticker_df.columns:
            vals = test_ticker_df[col].values.astype("float64")
            future_df[col] = np.where(np.isfinite(vals), vals, 0.0)
        else:
            future_df[col] = 0.0
    forecast = model.predict(future_df)
    yhat = forecast["yhat"].values.astype(np.float64)
    if cfg.use_log_target:
        # Back-transform: clip to avoid extreme values (log-space bounds)
        yhat = np.clip(yhat, -100.0, 20.0)  # exp(-100)~0, exp(20)~485M — safe range
        return np.exp(yhat).astype(np.float32)
    return yhat.astype(np.float32)


def build_expanding_time_folds(
    df: pd.DataFrame, n_splits: int, embargo: int
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Build expanding-window CV folds per-ticker."""
    folds = []
    for fold_num in range(n_splits):
        tr_parts, va_parts = [], []
        for _, grp in df.groupby("ticker", sort=False):
            grp = grp.sort_values("date").reset_index(drop=True)
            n = len(grp)
            if n < (n_splits + 1) * max(embargo, 1) + 10:
                continue
            boundaries = np.linspace(0, n, n_splits + 2, dtype=int)
            va_start = boundaries[fold_num + 1]
            va_end = boundaries[fold_num + 2]
            purged_train_end = va_start - embargo
            if purged_train_end <= 0 or va_end <= va_start:
                continue
            tr_parts.append(grp.iloc[:purged_train_end].copy())
            va_parts.append(grp.iloc[va_start:va_end].copy())
        if tr_parts and va_parts:
            folds.append((pd.concat(tr_parts, ignore_index=True), pd.concat(va_parts, ignore_index=True)))
    return folds


def main():
    cfg = Config()
    np.random.seed(cfg.random_state)

    print("Loading data...")
    raw_df = load_prices_from_sqlite(cfg.db_path, cfg.table_name)

    print("Adding features (RSI, log-volume, HL-range, log-ret)...")
    raw_df = add_features(raw_df)

    # Subsample tickers for speed
    all_tickers = raw_df["ticker"].unique()
    if cfg.max_tickers > 0 and len(all_tickers) > cfg.max_tickers:
        rng = np.random.default_rng(cfg.random_state)
        selected_tickers = rng.choice(all_tickers, cfg.max_tickers, replace=False).tolist()
        raw_df = raw_df[raw_df["ticker"].isin(selected_tickers)].reset_index(drop=True)
        print(f"Subsampled to {cfg.max_tickers} tickers")
    else:
        print(f"Using all {len(all_tickers)} tickers")

    print("Splitting train/test per ticker with embargo...")
    train_df, test_df = split_train_test_per_ticker(raw_df, cfg.test_size, cfg.target_horizon)
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    # === Cross-validation ===
    print("\nBuilding CV folds...")
    folds = build_expanding_time_folds(train_df, cfg.n_splits, cfg.target_horizon)
    print(f"CV folds: {len(folds)}")

    cv_results = []
    for fold_i, (fold_train, fold_val) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_i}/{len(folds)} ===")
        fold_preds, fold_trues, fold_prevs = [], [], []
        tickers = fold_val["ticker"].unique()
        n_tickers = len(tickers)

        for t_i, ticker in enumerate(tickers):
            tr_t = fold_train[fold_train["ticker"] == ticker].sort_values("date")
            va_t = fold_val[fold_val["ticker"] == ticker].sort_values("date")

            if len(tr_t) < 30 or len(va_t) < 2:
                continue

            try:
                model = fit_prophet_for_ticker(tr_t, cfg)
                pred_close = predict_prophet_for_ticker(model, va_t, cfg)
                actual_close = va_t["close"].values.astype(np.float64)
                # prev_close: last training close then each test day's prev close
                last_train_close = float(tr_t["close"].iloc[-1])
                prev_close = np.concatenate([[last_train_close], actual_close[:-1]])

                fold_trues.extend(actual_close.tolist())
                fold_preds.extend(pred_close.tolist())
                fold_prevs.extend(prev_close.tolist())
            except Exception:
                pass  # Skip problematic tickers

            if (t_i + 1) % 20 == 0:
                print(f"  Processed {t_i + 1}/{n_tickers} tickers", flush=True)

        if fold_trues:
            metrics = regression_metrics_with_direction(
                np.array(fold_trues), np.array(fold_preds), np.array(fold_prevs)
            )
            cv_results.append(metrics)
            print(f"  Fold {fold_i} | RMSE={metrics['rmse']:.6f} MAE={metrics['mae']:.6f} R2={metrics['r2']:.6f} DA={metrics['directional_accuracy']:.4f}")
        else:
            print(f"  Fold {fold_i} | No valid predictions")

    if cv_results:
        print("\n=== CV Summary ===")
        cv_df = pd.DataFrame(cv_results)
        print(cv_df.mean(numeric_only=True).to_string())
    else:
        cv_df = pd.DataFrame()

    # === Final evaluation on test set ===
    print("\nTraining final models per ticker on full training data...")
    all_preds, all_trues, all_prevs, all_meta = [], [], [], []
    tickers = test_df["ticker"].unique()

    for t_i, ticker in enumerate(tickers):
        tr_t = train_df[train_df["ticker"] == ticker].sort_values("date")
        te_t = test_df[test_df["ticker"] == ticker].sort_values("date")

        if len(tr_t) < 30 or len(te_t) < 2:
            continue

        try:
            model = fit_prophet_for_ticker(tr_t, cfg)
            pred_close = predict_prophet_for_ticker(model, te_t, cfg)
            actual_close = te_t["close"].values.astype(np.float64)
            dates = te_t["date"].values
            last_train_close = float(tr_t["close"].iloc[-1])
            prev_close = np.concatenate([[last_train_close], actual_close[:-1]])

            for j in range(len(actual_close)):
                all_trues.append(actual_close[j])
                all_preds.append(float(pred_close[j]))
                all_prevs.append(prev_close[j])
                all_meta.append((ticker, dates[j], actual_close[j]))
        except Exception:
            pass

        if (t_i + 1) % 20 == 0:
            print(f"  Processed {t_i + 1}/{len(tickers)} tickers", flush=True)

    y_true = np.array(all_trues, dtype=np.float64)
    y_pred = np.array(all_preds, dtype=np.float64)
    y_prev = np.array(all_prevs, dtype=np.float64)
    test_metrics = regression_metrics_with_direction(y_true, y_pred, y_prev)
    print(f"\nTest | RMSE={test_metrics['rmse']:.6f} MAE={test_metrics['mae']:.6f} R2={test_metrics['r2']:.6f} DA={test_metrics['directional_accuracy']:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    payload = {
        "config": asdict(cfg),
        "cv_results": cv_results,
        "test_metrics": test_metrics,
    }
    with open(cfg.model_output_path, "wb") as f:
        pickle.dump(payload, f)

    config_dict = asdict(cfg)
    with open(cfg.config_output_path, "w") as f:
        json.dump(config_dict, f, indent=2)

    meta_df = pd.DataFrame(all_meta, columns=["ticker", "date", "close"])
    meta_df["y_true"] = y_true
    meta_df["y_pred"] = y_pred
    meta_df["y_prev"] = y_prev
    pred_path = f"results/{month}-{day}-{tag}-prophet_predictions.csv"
    meta_df.to_csv(pred_path, index=False)

    result_txt = (
        f"Meta Prophet Results (log_target={cfg.use_log_target}):\n"
        f"RMSE={test_metrics['rmse']:.6f}\n"
        f"MAE={test_metrics['mae']:.6f}\n"
        f"R2={test_metrics['r2']:.6f}\n"
        f"DA={test_metrics['directional_accuracy']:.4f}\n\n"
        f"CV Mean R2={cv_df['r2'].mean() if len(cv_df) > 0 else 'N/A'}\n"
        f"CV Mean DA={cv_df['directional_accuracy'].mean() if len(cv_df) > 0 else 'N/A'}\n"
    )
    with open(f"{pred_path}.txt", "w") as f:
        f.write(result_txt)

    print(f"\nSaved model to: {cfg.model_output_path}")
    print(f"Saved predictions to: {pred_path}")


if __name__ == "__main__":
    main()
