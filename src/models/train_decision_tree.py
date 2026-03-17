"""
Decision Tree Regressor: Predict 5-day forward close price
- Target  : 5-day forward % return  (scale-invariant, matches features)
- Predict : reconstruct future price = current_close * (1 + predicted_return)
- Features: EMA20-based ratios (all scale-invariant)
"""

import sqlite3
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

DB_PATH      = "dataset/stock_prices_20y.db"
MODEL_OUT    = "decision_tree_regressor.pkl"
HORIZON      = 5
N_SPLITS     = 5       # Number of walk-forward splits
RANDOM_STATE = 42

# ── Features ──────────────────────────────────────────────────────────────────
def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ema_20"]      = c.ewm(span=20, adjust=False).mean()

    df["price_ema20"] = c / df["ema_20"]                       # where price sits vs trend
    df["ema20_slope"] = df["ema_20"].pct_change(5)             # trend direction
    df["price_mom5"]  = c.pct_change(5)                        # 5-day momentum
    df["price_mom10"] = c.pct_change(10)                       # 10-day momentum
    df["ema20_dist"]  = (c - df["ema_20"]) / df["ema_20"]      # % deviation from EMA20
    df["vol_ratio"]   = df["volume"] / df["volume"].rolling(20).mean()

    # ✅ Target is % return — same scale as features
    df["target"] = c.shift(-HORIZON) / c - 1

    return df.dropna()

FEATURE_COLS = [
    "price_ema20", "ema20_slope",
    "price_mom5",  "price_mom10",
    "ema20_dist",  "vol_ratio",
]

# ── Helpers ───────────────────────────────────────────────────────────────────
def get_tickers(conn):
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
    )]

def load_ticker(conn, ticker):
    return pd.read_sql_query(
        "SELECT date, close, volume FROM prices WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,), parse_dates=["date"],
    ).set_index("date")

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    conn = sqlite3.connect(DB_PATH)
    tickers = get_tickers(conn)
    print(f"Found {len(tickers)} tickers. Starting per-ticker walk-forward training...\n")

    # We will store models per ticker
    output_bundle = {
        "models": {},          # {ticker: model_object}
        "features": FEATURE_COLS
    }

    # Global metrics aggregation
    global_mae  = []
    global_rmse = []
    global_r2   = []

    skipped = 0
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)

    for i, ticker in enumerate(tickers, 1):
        try:
            df = load_ticker(conn, ticker)
            # Need enough data for 5 splits + lags
            if len(df) < 100:
                skipped += 1
                continue

            df = make_features(df)

            X = df[FEATURE_COLS].values
            y = df["target"].values

            # Walk-Forward Validation (TimeSeriesSplit)
            fold_maes, fold_rmses, fold_r2s = [], [], []

            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Train a temporary model for this fold
                model = DecisionTreeRegressor(
                    max_depth=5,            # Reduced depth to avoid overfitting on smaller slices
                    min_samples_leaf=20,    # Conservative leaf size
                    random_state=RANDOM_STATE
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                fold_maes.append(mean_absolute_error(y_test, y_pred))
                fold_rmses.append(np.sqrt(mean_squared_error(y_test, y_pred)))
                fold_r2s.append(r2_score(y_test, y_pred))

            # Aggregate fold metrics for this ticker
            if fold_maes:
                avg_mae  = np.mean(fold_maes)
                avg_rmse = np.mean(fold_rmses)
                avg_r2   = np.mean(fold_r2s)

                global_mae.append(avg_mae)
                global_rmse.append(avg_rmse)
                global_r2.append(avg_r2)

            # ── Final Training ────────────────────────────────────────────────
            # Train one final model on ALL data for this ticker (for inference)
            final_model = DecisionTreeRegressor(
                max_depth=5,
                min_samples_leaf=20,
                random_state=RANDOM_STATE
            )
            final_model.fit(X, y)
            output_bundle["models"][ticker] = final_model

            if i % 20 == 0 or i == len(tickers):
                # Print progress
                curr_mae = np.mean(global_mae) if global_mae else 0.0
                print(f"  Processed {i}/{len(tickers)} tickers … (Mean MAE: {curr_mae*100:.2f}%)")

        except Exception as e:
            print(f"  [WARN] {ticker}: {e}")
            skipped += 1

    conn.close()
    print(f"\nSkipped {skipped} tickers (insufficient data).")

    if global_mae:
        print("\n── Global Walk-Forward CV Results (Average across tickers) ──")
        print(f"  Mean MAE   : {np.mean(global_mae)*100:.4f}%")
        print(f"  Mean RMSE  : {np.mean(global_rmse)*100:.4f}%")
        print(f"  Mean R²    : {np.mean(global_r2):.4f}")
    else:
        print("\nNo models trained successfully.")

    # Save dictionary of models
    with open(MODEL_OUT, "wb") as f:
        pickle.dump(output_bundle, f)
    print(f"\nSaved {len(output_bundle['models'])} per-ticker models to {MODEL_OUT}")
    print("Format: {'models': {ticker: model}, 'features': [...]}")

if __name__ == "__main__":
    train()
