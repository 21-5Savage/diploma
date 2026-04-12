"""
Meta Prophet for next-day stock price prediction.

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
    predict_target: str = "price"
    random_state: int = 42

    # Prophet params
    changepoint_prior_scale: float = 0.05
    seasonality_prior_scale: float = 10.0
    yearly_seasonality: bool = True
    weekly_seasonality: bool = True
    daily_seasonality: bool = False

    # Subsample tickers for speed (Prophet fits per-ticker)
    max_tickers: int = 100

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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 2:
        return {"rmse": np.nan, "mae": np.nan, "r2": np.nan, "directional_accuracy": np.nan}
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def fit_prophet_for_ticker(
    train_ticker_df: pd.DataFrame,
    cfg: Config,
) -> Prophet:
    """Fit a Prophet model on one ticker's training data."""
    prophet_df = train_ticker_df[["date", "close"]].rename(columns={"date": "ds", "close": "y"})

    # Add regressors
    prophet_df = prophet_df.copy()
    prophet_df["volume"] = train_ticker_df["volume"].values

    model = Prophet(
        changepoint_prior_scale=cfg.changepoint_prior_scale,
        seasonality_prior_scale=cfg.seasonality_prior_scale,
        yearly_seasonality=cfg.yearly_seasonality,
        weekly_seasonality=cfg.weekly_seasonality,
        daily_seasonality=cfg.daily_seasonality,
    )
    model.add_regressor("volume")
    model.fit(prophet_df)
    return model


def predict_prophet_for_ticker(
    model: Prophet,
    test_ticker_df: pd.DataFrame,
    cfg: Config,
) -> np.ndarray:
    """Generate predictions for horizon=1 (next day close)."""
    future_df = test_ticker_df[["date", "volume"]].rename(columns={"date": "ds"}).copy()
    forecast = model.predict(future_df)
    pred_close = forecast["yhat"].values.astype(np.float32)
    return pred_close


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

    # Subsample tickers for speed
    all_tickers = raw_df["ticker"].unique()
    if len(all_tickers) > cfg.max_tickers:
        np.random.shuffle(all_tickers)
        selected_tickers = all_tickers[: cfg.max_tickers]
        raw_df = raw_df[raw_df["ticker"].isin(selected_tickers)].reset_index(drop=True)
        print(f"Subsampled to {cfg.max_tickers} tickers")

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
        fold_preds = []
        fold_trues = []
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

                actual_close = va_t["close"].values
                if cfg.predict_target == "return":
                    # Compute predicted return: predicted close vs previous close
                    prev_close = va_t["close"].shift(1).values
                    actual_return = actual_close / prev_close - 1.0
                    pred_return = pred_close / prev_close - 1.0
                    # Drop first (NaN from shift)
                    mask = ~np.isnan(actual_return)
                    fold_trues.extend(actual_return[mask].tolist())
                    fold_preds.extend(pred_return[mask].tolist())
                else:
                    fold_trues.extend(actual_close.tolist())
                    fold_preds.extend(pred_close.tolist())
            except Exception as e:
                pass  # Skip problematic tickers

            if (t_i + 1) % 20 == 0:
                print(f"  Processed {t_i + 1}/{n_tickers} tickers")

        if fold_trues:
            metrics = regression_metrics(np.array(fold_trues), np.array(fold_preds))
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
    all_preds = []
    all_trues = []
    all_meta = []
    tickers = test_df["ticker"].unique()

    for t_i, ticker in enumerate(tickers):
        tr_t = train_df[train_df["ticker"] == ticker].sort_values("date")
        te_t = test_df[test_df["ticker"] == ticker].sort_values("date")

        if len(tr_t) < 30 or len(te_t) < 2:
            continue

        try:
            model = fit_prophet_for_ticker(tr_t, cfg)
            pred_close = predict_prophet_for_ticker(model, te_t, cfg)

            actual_close = te_t["close"].values
            dates = te_t["date"].values

            if cfg.predict_target == "return":
                prev_close = te_t["close"].shift(1).values
                actual_return = actual_close / prev_close - 1.0
                pred_return = pred_close / prev_close - 1.0
                mask = ~np.isnan(actual_return)
                for j in range(len(actual_return)):
                    if mask[j]:
                        all_trues.append(actual_return[j])
                        all_preds.append(pred_return[j])
                        all_meta.append((ticker, dates[j], actual_close[j]))
            else:
                for j in range(len(actual_close)):
                    all_trues.append(actual_close[j])
                    all_preds.append(pred_close[j])
                    all_meta.append((ticker, dates[j], actual_close[j]))
        except Exception:
            pass

        if (t_i + 1) % 20 == 0:
            print(f"  Processed {t_i + 1}/{len(tickers)} tickers")

    y_true = np.array(all_trues, dtype=np.float32)
    y_pred = np.array(all_preds, dtype=np.float32)
    test_metrics = regression_metrics(y_true, y_pred)
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
    pred_path = f"results/{month}-{day}-{tag}-prophet_predictions.csv"
    meta_df.to_csv(pred_path, index=False)

    result_txt = f"""Meta Prophet Results:
RMSE={test_metrics['rmse']:.6f}
MAE={test_metrics['mae']:.6f}
R2={test_metrics['r2']:.6f}
DA={test_metrics['directional_accuracy']:.4f}

CV Mean R2={cv_df['r2'].mean() if len(cv_df) > 0 else 'N/A'}
"""
    with open(f"{pred_path}.txt", "w") as f:
        f.write(result_txt)

    print(f"\nSaved model to: {cfg.model_output_path}")
    print(f"Saved predictions to: {pred_path}")


if __name__ == "__main__":
    main()
