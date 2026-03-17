"""
Test & inference for the Decision Tree Regressor.
Model predicts 5-day % return; price is reconstructed as:
    predicted_price = current_close * (1 + predicted_return)

Usage:
  python test_decision_tree.py --ticker AAPL
  python test_decision_tree.py --ticker AAPL --predict
  python test_decision_tree.py --ticker AAPL --live
  python test_decision_tree.py --ticker AAPL --live --interval 30 --window 120
  python test_decision_tree.py --sample 100
"""

import sqlite3
import argparse
import pickle
import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Use non-interactive backend
matplotlib.use('Agg')

DB_PATH    = "dataset/stock_prices_20y.db"
MODEL_PATH = "decision_tree_regressor.pkl"
HORIZON    = 5

def load_model(ticker=None):
    with open(MODEL_PATH, "rb") as f:
        bundle = pickle.load(f)

    if "models" in bundle: # New format (per-ticker)
        if ticker is None:
            raise ValueError("Model file contains per-ticker models; 'ticker' argument is required.")
        if ticker not in bundle["models"]:
            raise ValueError(f"No model found for ticker '{ticker}'.")
        return bundle["models"][ticker], bundle["features"]
    else: # Old format (global model)
        return bundle["model"], bundle["features"]

def make_features(df: pd.DataFrame) -> pd.DataFrame:
    c = df["close"]
    df["ema_20"]      = c.ewm(span=20, adjust=False).mean()
    df["price_ema20"] = c / df["ema_20"]
    df["ema20_slope"] = df["ema_20"].pct_change(5)
    df["price_mom5"]  = c.pct_change(5)
    df["price_mom10"] = c.pct_change(10)
    df["ema20_dist"]  = (c - df["ema_20"]) / df["ema_20"]
    df["vol_ratio"]   = df["volume"] / df["volume"].rolling(20).mean()
    df["target"]      = c.shift(-HORIZON) / c - 1   # % return
    return df.dropna()

def load_ticker(conn, ticker):
    return pd.read_sql_query(
        "SELECT date, close, volume FROM prices WHERE ticker = ? ORDER BY date",
        conn, params=(ticker,), parse_dates=["date"],
    ).set_index("date")

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate_ticker(ticker: str, plot: bool = True, output_dir: str = "."):
    try:
        model, feat_cols = load_model(ticker)
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        return None

    conn = sqlite3.connect(DB_PATH)
    try:
        df = load_ticker(conn, ticker)
        if len(df) < 60:
            if plot: print(f"Not enough data for {ticker}.")
            conn.close()
            return None
        conn.close()
    except Exception as e:
        if plot: print(f"Error loading data for {ticker}: {e}")
        if conn: conn.close()
        return None

    df = make_features(df)
    missing = [c for c in feat_cols if c not in df.columns]
    if missing:
        if plot: print(f"Missing features: {missing}")
        return None

    X        = df[feat_cols].values
    y_ret    = df["target"].values                    # actual % return
    y_pred_r = model.predict(X)                       # predicted % return

    # Reconstruct prices
    actual_close = df["close"] * (1 + y_ret)
    pred_close   = df["close"] * (1 + y_pred_r)

    mae_r  = mean_absolute_error(y_ret, y_pred_r)
    rmse_r = np.sqrt(mean_squared_error(y_ret, y_pred_r))
    r2     = r2_score(y_ret, y_pred_r)
    mae_price = mean_absolute_error(actual_close, pred_close)

    metrics = {
        "ticker": ticker,
        "mae_return_pct": mae_r * 100,
        "rmse_return_pct": rmse_r * 100,
        "r2": r2,
        "mae_price": mae_price
    }

    if plot:
        print(f"\n── {ticker} ──────────────────────────────────────────────────")
        print(f"  MAE  (return) : {mae_r*100:.3f}%")
        print(f"  RMSE (return) : {rmse_r*100:.3f}%")
        print(f"  R²            : {r2:.4f}")
        print(f"  MAE  (price)  : ${mae_price:.2f}")

        # Plot full history instead of just tail(120)
        # However, to keep it readable, maybe plot last 300 days?
        # User asked for "continuously done and graphed", implying monitoring.
        # Let's plot the last 250 days (approx 1 trading year) for clarity,
        # or maybe the whole thing if it's not too dense.
        # Let's use 300 days.
        plot_window = 300
        tail     = df.tail(plot_window).copy() if len(df) > plot_window else df.copy()

        t_close  = tail["close"]
        t_actual = t_close * (1 + tail["target"])
        t_pred_r = model.predict(tail[feat_cols].values)
        t_pred_p = t_close * (1 + t_pred_r)

        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)

        axes[0].plot(tail.index, t_close,   label="Close today",      color="steelblue", linewidth=1.2)
        axes[0].plot(tail.index, t_actual,  label=f"Actual +{HORIZON}d close", color="orange", linestyle="--")
        axes[0].plot(tail.index, t_pred_p,  label=f"Predicted +{HORIZON}d",    color="green",  linestyle=":")
        axes[0].plot(tail.index, tail["ema_20"], label="EMA 20",       color="purple", linewidth=0.8, alpha=0.7)
        axes[0].set_title(f"{ticker} — Price forecast (last {len(tail)} days)")
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("Price ($)")

        error = t_pred_p - t_actual
        colors = ["green" if e >= 0 else "red" for e in error]
        axes[1].bar(tail.index, error, color=colors, width=1, alpha=0.7)
        axes[1].axhline(0, color="black", linewidth=0.8)
        axes[1].set_title("Price prediction error (predicted − actual)")
        axes[1].set_ylabel("Error ($)")

        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{ticker}_regression_test.png")
        plt.savefig(out_path, dpi=130)
        plt.close(fig) # Close explicitly to free memory
        print(f"  Plot saved → {out_path}")

    return metrics

# ── Predict latest ────────────────────────────────────────────────────────────
def predict_latest(ticker: str):
    try:
        model, feat_cols = load_model(ticker)
    except Exception as e:
        print(f"Error loading model for {ticker}: {e}")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        df = load_ticker(conn, ticker)
        if len(df) < 20:
             print("Not enough data")
             conn.close()
             return
        conn.close()
    except Exception:
        if conn: conn.close()
        return

    df = make_features(df)
    last         = df[feat_cols].iloc[[-1]]
    pred_return  = model.predict(last.values)[0]
    current      = df["close"].iloc[-1]
    pred_price   = current * (1 + pred_return)

    print(f"\n── {ticker} — 5-day price forecast ─────────────────────────")
    print(f"  Date today     : {df.index[-1].date()}")
    print(f"  Current close  : ${current:.2f}")
    print(f"  Predicted +5d  : ${pred_price:.2f}  ({pred_return*100:+.2f}%)")
    print(f"  Direction      : {'▲ UP' if pred_return > 0 else '▼ DOWN'}")

# ── Sample ────────────────────────────────────────────────────────────────────
def evaluate_sample(n: int):
    print(f"\nEvaluating performance on {n if n > 0 else 'ALL'} tickers...\n")
    conn = sqlite3.connect(DB_PATH)

    if n > 0:
        query = "SELECT DISTINCT ticker FROM prices ORDER BY RANDOM() LIMIT ?"
        params = (n,)
    else:
        query = "SELECT DISTINCT ticker FROM prices ORDER BY ticker"
        params = ()

    tickers = [r[0] for r in conn.execute(query, params)]
    conn.close()

    results = []

    # Create output directory for plots
    plot_dir = "test_results_plots"
    os.makedirs(plot_dir, exist_ok=True)
    print(f"Saving plots to '{plot_dir}/' ...")

    for i, t in enumerate(tickers, 1):
        try:
            # Enable plotting for sample evaluation
            m = evaluate_ticker(t, plot=True, output_dir=plot_dir)
            if m:
                results.append(m)
            if i % 10 == 0:
                print(f"  Processed {i}/{len(tickers)}...")
        except Exception as e:
            print(f"Error processing {t}: {e}")
            pass

    if not results:
        print("No results collected.")
        return

    df_res = pd.DataFrame(results)

    print("\n── Aggregate Metrics ────────────────────────────────────────")
    print(df_res.describe())

    # Graphs
    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    sns.histplot(df_res["mae_return_pct"], kde=True, color="skyblue")
    plt.axvline(df_res["mae_return_pct"].mean(), color='red', linestyle='--', label=f'Mean: {df_res["mae_return_pct"].mean():.2f}%')
    plt.title("Distribution of MAE (Return %)")
    plt.xlabel("MAE (%)")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(df_res["r2"].clip(lower=-1, upper=1), kde=True, color="orange") # Clip R2 for readability
    plt.axvline(df_res["r2"].median(), color='green', linestyle='--', label=f'Median: {df_res["r2"].median():.2f}')
    plt.title("Distribution of R² Score (Clipped [-1, 1])")
    plt.xlabel("R² Score")
    plt.legend()

    plt.tight_layout()
    out = "sample_performance_dist.png"
    plt.savefig(out, dpi=130)
    print(f"\nPerformance distribution graph saved → {out}")

# ── Live graph ───────────────────────────────────────────────────────────────
def live_graph(ticker: str, interval: int = 60, window: int = 60):
    """
    Open an interactive window showing predicted vs actual close prices.
    The chart re-reads the database and redraws every *interval* seconds.
    """
    import matplotlib.animation as animation

    # Switch away from the non-interactive Agg backend
    for backend in ("TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg", "WebAgg"):
        try:
            plt.switch_backend(backend)
            break
        except Exception:
            continue
    else:
        print("No interactive matplotlib backend found."
              " Install tkinter (python3-tk) or a Qt binding (PyQt5/PySide6).")
        return

    try:
        model, feat_cols = load_model(ticker)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    fig.suptitle(
        f"{ticker} — Live Price Forecast  (updates every {interval}s)",
        fontsize=12,
    )

    def _fetch():
        conn = sqlite3.connect(DB_PATH)
        try:
            df = load_ticker(conn, ticker)
        finally:
            conn.close()
        return make_features(df).tail(window)

    def _draw(_frame):
        tail = _fetch()
        if tail.empty:
            return

        t_close  = tail["close"]
        t_actual = t_close * (1 + tail["target"])
        t_pred_r = model.predict(tail[feat_cols].values)
        t_pred_p = t_close * (1 + t_pred_r)

        for ax in (ax1, ax2):
            ax.clear()

        ax1.plot(tail.index, t_close,   color="steelblue",  linewidth=1.2, label="Close")
        ax1.plot(tail.index, t_actual,  color="orange",     linestyle="--", label=f"Actual +{HORIZON}d close")
        ax1.plot(tail.index, t_pred_p,  color="limegreen",  linestyle=":",  label=f"Predicted +{HORIZON}d")
        ax1.set_ylabel("Price ($)")
        ax1.legend(fontsize=8)
        ax1.set_title(f"{ticker}  —  last {len(tail)} trading days")

        error  = t_pred_p - t_actual
        colors = ["limegreen" if e >= 0 else "tomato" for e in error]
        ax2.bar(tail.index, error, color=colors, width=1, alpha=0.8)
        ax2.axhline(0, color="black", linewidth=0.8)
        ax2.set_title("Prediction error  (predicted − actual)")
        ax2.set_ylabel("Error ($)")

        fig.autofmt_xdate()
        fig.tight_layout()

    ani = animation.FuncAnimation(
        fig, _draw,
        interval=interval * 1000,   # FuncAnimation expects milliseconds
        cache_frame_data=False,
    )
    _draw(0)    # render immediately on launch, don't wait for the first interval
    plt.show()
    return ani  # keep reference alive to prevent garbage collection


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker",   type=str)
    parser.add_argument("--predict",  action="store_true")
    parser.add_argument("--sample",   type=int)
    parser.add_argument("--live",     action="store_true",
                        help="open interactive chart that refreshes every --interval seconds")
    parser.add_argument("--interval", type=int, default=60, metavar="SEC",
                        help="refresh interval in seconds for --live (default: 60)")
    parser.add_argument("--window",   type=int, default=60, metavar="DAYS",
                        help="number of trading days to display in --live (default: 60)")
    args = parser.parse_args()

    if args.ticker and args.live:
        live_graph(args.ticker, interval=args.interval, window=args.window)
    elif args.ticker and args.predict:
        predict_latest(args.ticker)
    elif args.ticker:
        evaluate_ticker(args.ticker)
    elif args.sample is not None:
        evaluate_sample(args.sample)
    else:
        parser.print_help()
