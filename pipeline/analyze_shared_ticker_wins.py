from __future__ import annotations

import pickle
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  # noqa: F401


DATE_FROM = "2026-03-15"
DATE_TO = "2026-04-15"
SLUG = "2026-03-16_2026-04-15"

DB_PATH = Path("dataset/stock_prices_20y.db")
TICKERS_CSV = Path(f"pipeline/results/sampled_pipeline_tickers_{SLUG}.csv")
LSTM_MODEL_PKL = Path("artifacts/lstm/models/20260415_140411-lstm_torch.pkl")
TREE_MODEL_PKL = Path("artifacts/4-16-mnu-decision_tree.pkl")
RNN_ROWS_CSV = Path(f"pipeline/results/sampled_rnn_{SLUG}_rows.csv")

RESULTS_DIR = Path("pipeline/results")
LSTM_ROWS_CSV = RESULTS_DIR / f"sampled_lstm_{SLUG}_rows.csv"
TREE_ROWS_CSV = RESULTS_DIR / f"sampled_tree_{SLUG}_rows.csv"
PER_TICKER_CSV = RESULTS_DIR / f"shared_sample_per_ticker_{SLUG}.csv"
WIN_COUNTS_CSV = RESULTS_DIR / f"shared_sample_wins_{SLUG}.csv"
DAILY_WINNERS_CSV = RESULTS_DIR / f"shared_sample_daily_winners_{SLUG}.csv"
PLOTS_DIR = RESULTS_DIR / "plots"
DAILY_WINNERS_PLOT = PLOTS_DIR / f"shared_sample_daily_winners_{SLUG}.png"

DEVICE = torch.device("cpu")
SHARED_MODEL_ORDER = ["tree", "lstm", "rnn"]
DAILY_MODEL_ORDER = ["tree", "lstm", "rnn", "llm"]
MODEL_COLORS = {
    "tree": "#e07b39",
    "lstm": "#3a7ebf",
    "rnn": "#9b59b6",
    "llm": "#48a868",
}


def compute_shared_features(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    close = g["close"].astype("float64")
    high = g["high"].astype("float64")
    low = g["low"].astype("float64")
    open_ = g["open"].astype("float64")
    volume = g["volume"].astype("float64").clip(lower=1.0)

    log_close = np.log(close)
    g["log_ret_1"] = log_close.diff(1)
    g["log_ret_3"] = log_close.diff(3)
    g["log_ret_5"] = log_close.diff(5)
    g["log_ret_10"] = log_close.diff(10)
    g["log_ret_20"] = log_close.diff(20)
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
    vol_ma5 = volume.rolling(5).mean()
    vol_ma20 = volume.rolling(20).mean()
    g["vol_vs_ma_5"] = volume / vol_ma5.clip(lower=1.0) - 1.0
    g["vol_vs_ma_20"] = volume / vol_ma20.clip(lower=1.0) - 1.0

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / 14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / 14, min_periods=14).mean()
    g["rsi_14"] = (100.0 - 100.0 / (1.0 + gain / (loss + 1e-8))) / 100.0 - 0.5

    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    g["macd_norm"] = (ema12 - ema26) / close.clip(lower=1e-6)

    bb_mean = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    g["bb_pos"] = (close - (bb_mean - 2 * bb_std)) / (4 * bb_std).clip(lower=1e-6) - 0.5

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    g["atr_14_norm"] = tr.ewm(alpha=1.0 / 14, min_periods=14).mean() / close.clip(lower=1e-6)
    return g.replace([np.inf, -np.inf], np.nan)


class LSTMModel(nn.Module):
    def __init__(self, n_features: int, lstm_units: int, num_layers: int, dense_units: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=lstm_units,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(lstm_units, dense_units)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dense_units, dense_units // 2)
        self.fc3 = nn.Linear(dense_units // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        out = self.dropout(last)
        out = self.act(self.fc1(out))
        out = self.act(self.fc2(out))
        return self.fc3(out).squeeze(-1)


def load_prices(tickers: list[str]) -> pd.DataFrame:
    placeholders = ",".join("?" * len(tickers))
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT ticker, date, open, high, low, close, volume
            FROM prices
            WHERE ticker IN ({placeholders})
              AND date <= ?
            ORDER BY ticker, date
            """,
            conn,
            params=tickers + [DATE_TO],
        )
    df["date"] = pd.to_datetime(df["date"])
    for col in ["open", "high", "low", "close"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype("float64")
    return df


def build_sequence_rows(model_name: str, model: nn.Module, scaler, feature_cols: list[str], seq_len: int, raw: pd.DataFrame) -> pd.DataFrame:
    mean_arr = scaler.mean_.astype(np.float32)
    scale_arr = scaler.scale_.astype(np.float32)
    scale_arr[scale_arr == 0.0] = 1.0

    target_dates = pd.date_range(DATE_FROM, DATE_TO, freq="B")
    rows = []
    for ticker, grp in raw.groupby("ticker", sort=False):
        feat = compute_shared_features(grp).reset_index(drop=True)
        feat_vals = feat[feature_cols].to_numpy(dtype=np.float32)
        closes = feat["close"].to_numpy(dtype=np.float64)
        dates = feat["date"].to_numpy("datetime64[ns]")
        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}

        for pred_date in target_dates:
            if pred_date not in date_to_idx:
                continue
            i = date_to_idx[pred_date]
            if i < seq_len:
                continue
            seq = feat_vals[i - seq_len : i]
            if np.isnan(seq).any():
                continue
            prev_close = closes[i - 1]
            actual_close = closes[i]
            if np.isnan(prev_close) or np.isnan(actual_close) or prev_close <= 0:
                continue

            seq_norm = (seq - mean_arr) / scale_arr
            x = torch.from_numpy(seq_norm).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_return = float(model(x).item())

            pred_close = prev_close * np.exp(np.clip(pred_return, -2.0, 2.0))
            rows.append(
                {
                    "pred_date": pred_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "model_name": model_name,
                    "prev_close": prev_close,
                    "actual_close": actual_close,
                    "pred_close": pred_close,
                    "abs_error": abs(pred_close - actual_close),
                    "directional_correct": float(np.sign(pred_close - prev_close) == np.sign(actual_close - prev_close)),
                }
            )
    return pd.DataFrame(rows)


def build_tree_rows(tree_model, feature_cols: list[str], raw: pd.DataFrame) -> pd.DataFrame:
    target_dates = pd.date_range(DATE_FROM, DATE_TO, freq="B")
    rows = []
    for ticker, grp in raw.groupby("ticker", sort=False):
        feat = compute_shared_features(grp).reset_index(drop=True)
        feat_vals = feat[feature_cols].to_numpy(dtype=np.float32)
        closes = feat["close"].to_numpy(dtype=np.float64)
        dates = feat["date"].to_numpy("datetime64[ns]")
        date_to_idx = {pd.Timestamp(d): i for i, d in enumerate(dates)}

        for pred_date in target_dates:
            if pred_date not in date_to_idx:
                continue
            i = date_to_idx[pred_date]
            if i < 1:
                continue
            x = feat_vals[i - 1]
            if np.isnan(x).any():
                continue
            prev_close = closes[i - 1]
            actual_close = closes[i]
            if np.isnan(prev_close) or np.isnan(actual_close) or prev_close <= 0:
                continue

            pred_return = float(tree_model.predict(x.reshape(1, -1))[0])
            pred_close = prev_close * np.exp(np.clip(pred_return, -2.0, 2.0))
            rows.append(
                {
                    "pred_date": pred_date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "model_name": "tree",
                    "prev_close": prev_close,
                    "actual_close": actual_close,
                    "pred_close": pred_close,
                    "abs_error": abs(pred_close - actual_close),
                    "directional_correct": float(np.sign(pred_close - prev_close) == np.sign(actual_close - prev_close)),
                }
            )
    return pd.DataFrame(rows)


def summarize_per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["ticker", "model_name"], as_index=False)
        .agg(
            mean_abs_error=("abs_error", "mean"),
            rmse=("abs_error", lambda s: float(np.sqrt(np.mean(np.square(s))))),
            directional_acc=("directional_correct", "mean"),
            n_days=("pred_date", "nunique"),
        )
    )
    grouped["rank_mae"] = grouped.groupby("ticker")["mean_abs_error"].rank(method="min")
    grouped["rank_da"] = grouped.groupby("ticker")["directional_acc"].rank(method="min", ascending=False)
    grouped["mae_winner"] = grouped["rank_mae"] == 1
    grouped["da_winner"] = grouped["rank_da"] == 1
    return grouped.sort_values(["ticker", "mean_abs_error", "model_name"]).reset_index(drop=True)


def winner_counts(per_ticker: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model_name, g in per_ticker.groupby("model_name"):
        rows.append(
            {
                "model_name": model_name,
                "mae_wins": int(g["mae_winner"].sum()),
                "da_wins": int(g["da_winner"].sum()),
                "avg_ticker_mae": float(g["mean_abs_error"].mean()),
                "avg_ticker_da": float(g["directional_acc"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["mae_wins", "da_wins", "avg_ticker_mae"], ascending=[False, False, True]).reset_index(drop=True)


def build_daily_winners() -> pd.DataFrame:
    metric_rules = {
        "mae": "min",
        "rmse": "min",
        "directional_acc": "max",
        "mape_pct": "min",
    }
    frames = []
    for model_name in DAILY_MODEL_ORDER:
        path = RESULTS_DIR / f"sampled_{model_name}_{SLUG}_daily.csv"
        df = pd.read_csv(path)
        df["model_name"] = model_name
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)

    rows = []
    for pred_date, g in combined.groupby("pred_date", sort=True):
        row = {"pred_date": pred_date}
        for metric, direction in metric_rules.items():
            best_value = g[metric].min() if direction == "min" else g[metric].max()
            winners = g.loc[np.isclose(g[metric], best_value), "model_name"].tolist()
            row[f"best_{metric}"] = "/".join(sorted(winners))
            row[f"best_{metric}_value"] = float(best_value)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_daily_winners(daily_winners: pd.DataFrame) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    metric_specs = [
        ("best_mae", "Daily Best MAE"),
        ("best_rmse", "Daily Best RMSE"),
        ("best_directional_acc", "Daily Best Dir Acc"),
        ("best_mape_pct", "Daily Best MAPE %"),
    ]
    model_to_y = {name: idx for idx, name in enumerate(DAILY_MODEL_ORDER)}
    dates = pd.to_datetime(daily_winners["pred_date"])

    fig, axes = plt.subplots(2, 2, figsize=(15, 8), constrained_layout=True, sharex=True)
    axes = axes.flatten()

    for ax, (winner_col, title) in zip(axes, metric_specs):
        for model_name in DAILY_MODEL_ORDER:
            xs = []
            ys = []
            for date, winner_str in zip(dates, daily_winners[winner_col]):
                winners = str(winner_str).split("/")
                if model_name in winners:
                    xs.append(date)
                    ys.append(model_to_y[model_name])
            if xs:
                ax.scatter(xs, ys, s=70, color=MODEL_COLORS[model_name], label=model_name.upper())

        ax.set_title(title)
        ax.set_yticks(list(model_to_y.values()))
        ax.set_yticklabels([name.upper() for name in DAILY_MODEL_ORDER])
        ax.grid(True, alpha=0.25, axis="x")
        ax.tick_params(axis="x", rotation=45)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(DAILY_MODEL_ORDER), frameon=False)
    fig.suptitle("", fontsize=15)
    fig.savefig(DAILY_WINNERS_PLOT, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    tickers = pd.read_csv(TICKERS_CSV)["ticker"].tolist()
    raw = load_prices(tickers)
    print(f"Loaded {len(raw):,} price rows for {raw['ticker'].nunique()} sampled tickers")

    with open(LSTM_MODEL_PKL, "rb") as f:
        lstm_payload = pickle.load(f)
    lstm_cfg = lstm_payload["config"]
    lstm_model = LSTMModel(
        n_features=len(lstm_cfg["feature_cols"]),
        lstm_units=lstm_cfg["lstm_units"],
        num_layers=lstm_cfg["num_layers"],
        dense_units=lstm_cfg["dense_units"],
        dropout=lstm_cfg["dropout"],
    ).to(DEVICE)
    lstm_model.load_state_dict(lstm_payload["model_state_dict"])
    lstm_model.eval()
    lstm_rows = build_sequence_rows(
        "lstm",
        lstm_model,
        lstm_payload["scaler"],
        list(lstm_cfg["feature_cols"]),
        int(lstm_cfg["sequence_length"]),
        raw,
    )
    lstm_rows.to_csv(LSTM_ROWS_CSV, index=False)
    print(f"Saved {LSTM_ROWS_CSV}")

    with open(TREE_MODEL_PKL, "rb") as f:
        tree_payload = pickle.load(f)
    tree_rows = build_tree_rows(tree_payload["model"], list(tree_payload["feature_cols"]), raw)
    tree_rows.to_csv(TREE_ROWS_CSV, index=False)
    print(f"Saved {TREE_ROWS_CSV}")

    rnn_rows = pd.read_csv(RNN_ROWS_CSV)
    common_tickers = set(tree_rows["ticker"]) & set(lstm_rows["ticker"]) & set(rnn_rows["ticker"])
    common_dates = set(tree_rows["pred_date"]) & set(lstm_rows["pred_date"]) & set(rnn_rows["pred_date"])

    frames = []
    for df in [tree_rows, lstm_rows, rnn_rows]:
        frames.append(df[df["ticker"].isin(common_tickers) & df["pred_date"].isin(common_dates)].copy())
    combined = pd.concat(frames, ignore_index=True)

    per_ticker = summarize_per_ticker(combined)
    per_ticker.to_csv(PER_TICKER_CSV, index=False)
    print(f"Saved {PER_TICKER_CSV}")

    wins = winner_counts(per_ticker)
    wins.to_csv(WIN_COUNTS_CSV, index=False)
    print(f"Saved {WIN_COUNTS_CSV}")

    daily_winners = build_daily_winners()
    daily_winners.to_csv(DAILY_WINNERS_CSV, index=False)
    print(f"Saved {DAILY_WINNERS_CSV}")
    plot_daily_winners(daily_winners)
    print(f"Saved {DAILY_WINNERS_PLOT}")

    print("\nWinner counts across the shared sample:")
    print(wins.to_string(index=False, float_format="%.4f"))

    print("\nFirst 10 days of daily winners:")
    print(daily_winners.head(10).to_string(index=False))

    print("\nTop 15 tickers by lowest mean absolute error winner:")
    top = per_ticker[per_ticker["mae_winner"]].sort_values(["mean_abs_error", "ticker"]).head(15)
    print(top[["ticker", "model_name", "mean_abs_error", "directional_acc", "n_days"]].to_string(index=False, float_format="%.4f"))

    print("\nNote: the daily-winner plot/CSV now include LLM, but the per-ticker win table still excludes it because the current sampled LLM evaluation used a different ticker subset.")


if __name__ == "__main__":
    main()
