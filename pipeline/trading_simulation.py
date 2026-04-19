"""
Simulate a simple daily trading strategy based on each model's predictions.

Strategy:
  - Long-only: if model predicts price UP (pred_close > prev_close), buy at
    prev_close and sell at actual_close.  Otherwise stay flat.
  - Long/short: additionally, if model predicts DOWN, short at prev_close and
    cover at actual_close.
  - Buy-and-hold baseline: always long every ticker every day.

Metrics reported:
  - Total return (%), Annualised return, Sharpe ratio, max drawdown,
    win-rate of trades, cumulative P&L on a $10 000 starting capital.

Usage:
    python pipeline/trading_simulation.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).resolve().parent / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

DATE_FROM = "2026-03-16"
DATE_TO = "2026-04-15"

INITIAL_CAPITAL = 10_000.0


# ── helpers ──────────────────────────────────────────────────────────────
def load_rows(model: str) -> pd.DataFrame:
    """Load row-level predictions CSV for a model."""
    path = RESULTS_DIR / f"sampled_{model}_{DATE_FROM}_{DATE_TO}_rows.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["pred_date"])
    # normalise column order
    for col in ("prev_close", "actual_close", "pred_close"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["pred_date", "ticker", "model_name", "prev_close", "actual_close", "pred_close"]].dropna()


def daily_portfolio_return(
    df: pd.DataFrame, strategy: str = "long_only"
) -> pd.DataFrame:
    """Compute daily equal-weight portfolio return for a strategy.

    For each day, every ticker with a signal gets an equal share of capital.
    Returns a DataFrame with columns [pred_date, port_return, n_trades].
    """
    records = []
    for date, g in df.groupby("pred_date", sort=True):
        if strategy == "long_only":
            trades = g[g["pred_close"] > g["prev_close"]].copy()
            if trades.empty:
                records.append({"pred_date": date, "port_return": 0.0, "n_trades": 0})
                continue
            trades["ret"] = (trades["actual_close"] - trades["prev_close"]) / trades["prev_close"]
        elif strategy == "long_short":
            g = g.copy()
            g["signal"] = np.sign(g["pred_close"] - g["prev_close"])
            trades = g[g["signal"] != 0].copy()
            if trades.empty:
                records.append({"pred_date": date, "port_return": 0.0, "n_trades": 0})
                continue
            trades["ret"] = trades["signal"] * (trades["actual_close"] - trades["prev_close"]) / trades["prev_close"]
        elif strategy == "buy_hold":
            trades = g.copy()
            trades["ret"] = (trades["actual_close"] - trades["prev_close"]) / trades["prev_close"]
        else:
            raise ValueError(strategy)

        records.append({
            "pred_date": date,
            "port_return": trades["ret"].mean(),
            "n_trades": len(trades),
        })
    return pd.DataFrame(records)


def cumulative_equity(daily_returns: pd.Series, initial: float = INITIAL_CAPITAL) -> pd.Series:
    return initial * (1 + daily_returns).cumprod()


def max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def sharpe_ratio(daily_returns: pd.Series, annual_factor: float = 252) -> float:
    if daily_returns.std() == 0:
        return 0.0
    return float(daily_returns.mean() / daily_returns.std() * np.sqrt(annual_factor))


def strategy_stats(port_df: pd.DataFrame, label: str, bh_returns: pd.Series | None = None) -> dict:
    rets = port_df["port_return"]
    eq = cumulative_equity(rets)
    total_ret = (eq.iloc[-1] / INITIAL_CAPITAL - 1) * 100
    n_days = len(rets)
    days_above_start = int((eq > INITIAL_CAPITAL).sum())
    if bh_returns is not None and len(bh_returns) == n_days:
        days_beat_market = int((rets.values > bh_returns.values).sum())
    else:
        days_beat_market = None
    return {
        "strategy": label,
        "total_return_pct": round(total_ret, 3),
        "final_equity": round(eq.iloc[-1], 2),
        "sharpe": round(sharpe_ratio(rets), 3),
        "max_drawdown_pct": round(max_drawdown(eq) * 100, 3),
        "mean_daily_trades": round(port_df["n_trades"].mean(), 1),
        "win_rate_pct": round((rets > 0).sum() / max(len(rets), 1) * 100, 1),
        f"days_beat_market/{n_days}": days_beat_market,
        f"days_above_start/{n_days}": days_above_start,
    }


# ── main ─────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["tree", "lstm", "rnn", "prophet", "llm"])
    args = parser.parse_args()

    all_stats: list[dict] = []
    equity_curves: dict[str, pd.DataFrame] = {}

    # -- buy-and-hold baseline (use any model's rows – actuals are the same) --
    first_df = None
    for m in args.models:
        first_df = load_rows(m)
        if not first_df.empty:
            break
    if first_df is None or first_df.empty:
        print("No row-level data found. Exiting.")
        return

    bh_port = daily_portfolio_return(first_df, strategy="buy_hold")
    bh_returns = bh_port["port_return"].reset_index(drop=True)
    bh_stats = strategy_stats(bh_port, "buy_hold (baseline)", bh_returns)
    all_stats.append(bh_stats)
    bh_port["equity"] = cumulative_equity(bh_port["port_return"]).values
    # equity_curves["buy & hold"] = bh_port

    # -- random investment baseline (Monte Carlo avg over N_RANDOM runs) --
    N_RANDOM = 1000
    rng = np.random.default_rng(42)
    dates_sorted = sorted(first_df["pred_date"].unique())
    random_equities = np.zeros((N_RANDOM, len(dates_sorted)))
    for i in range(N_RANDOM):
        daily_rets = []
        for j, date in enumerate(dates_sorted):
            g = first_df[first_df["pred_date"] == date]
            # randomly pick ~50% of tickers to go long
            mask = rng.random(len(g)) < 0.5
            if mask.sum() == 0:
                daily_rets.append(0.0)
            else:
                rets = ((g["actual_close"].values - g["prev_close"].values)
                        / g["prev_close"].values)
                daily_rets.append(rets[mask].mean())
        eq = INITIAL_CAPITAL * np.cumprod(1 + np.array(daily_rets))
        random_equities[i] = eq
    avg_random_equity = random_equities.mean(axis=0)
    avg_random_returns = pd.Series(
        np.diff(avg_random_equity, prepend=INITIAL_CAPITAL) / np.concatenate([[INITIAL_CAPITAL], avg_random_equity[:-1]])
    )
    rand_port = pd.DataFrame({"pred_date": dates_sorted, "port_return": avg_random_returns.values, "n_trades": 50.0})
    rand_stats = strategy_stats(rand_port, "random (baseline)", bh_returns)
    all_stats.append(rand_stats)
    rand_port["equity"] = avg_random_equity
    equity_curves["random"] = rand_port

    # -- per-model strategies --
    for model in args.models:
        df = load_rows(model)
        if df.empty:
            print(f"  [skip] no rows for {model}")
            continue

        for strat in ("long_only", "long_short"):
            label = f"{model} {strat}"
            port = daily_portfolio_return(df, strategy=strat)
            stats = strategy_stats(port, label, bh_returns)
            all_stats.append(stats)
            port["equity"] = cumulative_equity(port["port_return"]).values
            equity_curves[label] = port

    # -- summary table --
    summary = pd.DataFrame(all_stats)
    out_csv = RESULTS_DIR / f"trading_sim_{DATE_FROM}_{DATE_TO}.csv"
    summary.to_csv(out_csv, index=False)

    n_days = len(bh_returns)
    beat_col = f"days_beat_market/{n_days}"
    above_col = f"days_above_start/{n_days}"

    print(f"\n{'='*80}")
    print(f"Trading simulation  {DATE_FROM} → {DATE_TO}   (start ${INITIAL_CAPITAL:,.0f})")
    print(f"{'='*80}")
    print(summary.to_string(index=False))

    ranked_market = (
        summary.dropna(subset=[beat_col])
        .sort_values(beat_col, ascending=False)
        [["strategy", beat_col, above_col, "total_return_pct", "sharpe"]]
        .copy()
        .rename(columns={beat_col: "days_beat_market", above_col: "days_above_start"})
        .assign(rank_beat_market=lambda d: range(1, len(d) + 1))
    )
    ranked_above = (
        summary.sort_values(above_col, ascending=False)
        [["strategy", above_col, beat_col, "total_return_pct", "sharpe"]]
        .copy()
        .rename(columns={above_col: "days_above_start", beat_col: "days_beat_market"})
        .assign(rank_above_start=lambda d: range(1, len(d) + 1))
    )
    # combined ranking CSV: one row per strategy with both ranks
    combined = ranked_market[["strategy", "days_beat_market", "rank_beat_market"]].merge(
        ranked_above[["strategy", "days_above_start", "rank_above_start"]],
        on="strategy", how="outer"
    ).merge(
        summary[["strategy", "total_return_pct", "final_equity", "sharpe"]],
        on="strategy", how="left"
    ).sort_values("rank_beat_market")
    combined["n_days"] = n_days
    rank_csv = RESULTS_DIR / f"trading_sim_rankings_{DATE_FROM}_{DATE_TO}.csv"
    combined.to_csv(rank_csv, index=False)

    print(f"\n── Ranked: days beat market (out of {n_days}) ──")
    for _, row in combined.iterrows():
        print(f"  {row['strategy']:<28s}  {int(row['days_beat_market']):>2}/{n_days}")

    print(f"\n── Ranked: days portfolio above ${INITIAL_CAPITAL:,.0f} (out of {n_days}) ──")
    for _, row in combined.sort_values("rank_above_start").iterrows():
        print(f"  {row['strategy']:<28s}  {int(row['days_above_start']):>2}/{n_days}")

    print(f"\nSaved → {out_csv}")
    print(f"Saved → {rank_csv}")

    # -- equity curve plot --
    fig, ax = plt.subplots(figsize=(14, 6))
    for label, edf in equity_curves.items():
        style = {"linewidth": 1.8}
        if label == "random":
            style.update(ls=":", color="black", linewidth=2.0, alpha=0.7)
        elif label == "buy & hold":
            style.update(ls="--", linewidth=2.0, alpha=0.7)
        ax.plot(edf["pred_date"], edf["equity"], label=label, **style)
    ax.axhline(INITIAL_CAPITAL, color="grey", ls="--", lw=0.8, label="start")
    ax.set_title(f"Equity curves – simulated trading  ({DATE_FROM} → {DATE_TO})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plot_path = PLOTS_DIR / f"trading_sim_equity_{DATE_FROM}_{DATE_TO}.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Plot  → {plot_path}")

    # -- per-model individual plots (long_only + long_short vs buy & hold) --
    model_list = [m for m in args.models if f"{m} long_only" in equity_curves]

    MODEL_COLORS = {
        "tree":    "#e07b39",
        "lstm":    "#3a7ebf",
        "rnn":     "#9b59b6",
        "prophet": "#c0392b",
        "llm":     "#48a868",
    }

    for model in model_list:
        color = MODEL_COLORS.get(model, "steelblue")
        lo_df = equity_curves[f"{model} long_only"]
        ls_df = equity_curves[f"{model} long_short"]
        fig2, ax = plt.subplots(figsize=(10, 5))
        ax.plot(bh_port["pred_date"], bh_port["equity"],
                color="grey", ls="--", lw=1.5, label="buy & hold", alpha=0.8)
        ax.plot(lo_df["pred_date"], lo_df["equity"],
                color=color, ls="-", lw=2.0, label="long only")
        ax.plot(ls_df["pred_date"], ls_df["equity"],
                color=color, ls="-.", lw=1.8, label="long/short", alpha=0.85)
        ax.axhline(INITIAL_CAPITAL, color="grey", ls=":", lw=0.8)
        ax.set_title(f"{model.upper()} – equity curves  ({DATE_FROM} → {DATE_TO})", fontsize=12)
        ax.set_xlabel("Date")
        ax.set_ylabel("Portfolio value ($)")
        ax.tick_params(axis="x", rotation=30)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        fig2.tight_layout()
        per_model_path = PLOTS_DIR / f"trading_sim_{model}_{DATE_FROM}_{DATE_TO}.png"
        fig2.savefig(per_model_path, dpi=150)
        plt.close(fig2)
        print(f"Plot  → {per_model_path}")


if __name__ == "__main__":
    main()
