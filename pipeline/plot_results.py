
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


RESULTS_DIR = Path("pipeline/results")
PLOTS_DIR = RESULTS_DIR / "plots"
LLM_RESULTS_DIR = Path("results")

METRIC_SPECS = [
    ("mae", "MAE"),
    ("rmse", "RMSE"),
    ("r2", "R^2"),
    ("directional_acc", "Dir Acc"),
]


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_")


def _plot_metric_panel(ax, df: pd.DataFrame, x_col: str, y_col: str, title: str) -> None:
    if y_col not in df.columns or df[y_col].dropna().empty:
        ax.text(0.5, 0.5, "Skipped", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    plot_df = df[[x_col, y_col]].dropna().copy()
    if plot_df.empty:
        ax.text(0.5, 0.5, "Skipped", ha="center", va="center", fontsize=12)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    ax.plot(plot_df[x_col], plot_df[y_col], marker="o", linewidth=1.8, markersize=4)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.25)


def plot_pipeline_model_graphs(daily_csv: Path, summary_csv: Path | None = None) -> list[Path]:
    df = pd.read_csv(daily_csv)
    if df.empty:
        return []

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    df["pred_date"] = pd.to_datetime(df["pred_date"])
    outputs: list[Path] = []

    summary_df = pd.read_csv(summary_csv) if summary_csv and summary_csv.exists() else pd.DataFrame()

    for model_name, g in df.groupby("model_name"):
        g = g.sort_values("pred_date").reset_index(drop=True)
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
        axes = axes.flatten()

        for ax, (metric_key, metric_label) in zip(axes, METRIC_SPECS):
            _plot_metric_panel(ax, g, "pred_date", metric_key, metric_label)

        title = f"{model_name} Backtest Performance"
        if not summary_df.empty and "model_name" in summary_df.columns:
            row = summary_df[summary_df["model_name"] == model_name]
            if not row.empty:
                s = row.iloc[0]
                title += (
                    f"\nMean MAE={s.get('mean_mae', np.nan):.4f}  "
                    f"Mean RMSE={s.get('mean_rmse', np.nan):.4f}  "
                    f"Mean R^2={s.get('mean_r2', np.nan):.4f}  "
                    f"Mean Dir Acc={s.get('mean_da', np.nan):.4f}"
                )
        fig.suptitle(title, fontsize=14)

        out_path = PLOTS_DIR / f"{_slug(model_name)}_metrics.png"
        fig.savefig(out_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        outputs.append(out_path)

    return outputs


def _find_latest_llm_predictions() -> Path | None:
    candidates = sorted(LLM_RESULTS_DIR.glob("*gemini_predictions.csv"))
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def _directional_accuracy_for_prices(y_true: np.ndarray, y_pred: np.ndarray) -> float | float("nan"):
    if len(y_true) < 2:
        return np.nan
    dir_true = np.sign(np.diff(y_true, prepend=y_true[0]))
    dir_pred = np.sign(y_pred - np.concatenate([[y_true[0]], y_true[:-1]]))
    return float(np.mean(dir_true[1:] == dir_pred[1:])) if len(y_true) > 1 else np.nan


def _directional_accuracy_vs_prev_close(
    prev_close: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray
) -> float | float("nan"):
    mask = np.isfinite(prev_close) & np.isfinite(y_true) & np.isfinite(y_pred)
    prev_close = prev_close[mask]
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) == 0:
        return np.nan

    dir_true = np.sign(y_true - prev_close)
    dir_pred = np.sign(y_pred - prev_close)
    return float(np.mean(dir_true == dir_pred))


def plot_llm_graph(llm_csv: Path | None = None) -> Path | None:
    llm_csv = llm_csv or _find_latest_llm_predictions()
    if llm_csv is None or not llm_csv.exists():
        return None

    df = pd.read_csv(llm_csv)
    required = {"date", "y_true", "y_pred"}
    if not required.issubset(df.columns):
        return None

    df["date"] = pd.to_datetime(df["date"])
    daily_rows = []
    for date, g in df.groupby("date"):
        y_true = g["y_true"].to_numpy(dtype=float)
        y_pred = g["y_pred"].to_numpy(dtype=float)
        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true = y_true[mask]
        y_pred = y_pred[mask]
        if len(y_true) == 0:
            continue

        mae = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        r2 = np.nan
        if "prev_close" in g.columns:
            da = _directional_accuracy_vs_prev_close(
                g["prev_close"].to_numpy(dtype=float),
                g["y_true"].to_numpy(dtype=float),
                g["y_pred"].to_numpy(dtype=float),
            )
        else:
            da = _directional_accuracy_for_prices(y_true, y_pred)
        daily_rows.append({
            "pred_date": date,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "directional_acc": da,
        })

    daily_df = pd.DataFrame(daily_rows).sort_values("pred_date")
    if daily_df.empty:
        return None

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    axes = axes.flatten()
    for ax, (metric_key, metric_label) in zip(axes, METRIC_SPECS):
        _plot_metric_panel(ax, daily_df, "pred_date", metric_key, metric_label)

    fig.suptitle("LLM / Gemini Performance", fontsize=14)
    out_path = PLOTS_DIR / "llm_gemini_metrics.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--daily-csv", default=None, help="Pipeline daily metrics CSV")
    parser.add_argument("--summary-csv", default=None, help="Pipeline summary metrics CSV")
    parser.add_argument("--llm-csv", default=None, help="Optional LLM predictions CSV")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    daily_csv = Path(args.daily_csv) if args.daily_csv else None
    summary_csv = Path(args.summary_csv) if args.summary_csv else None
    llm_csv = Path(args.llm_csv) if args.llm_csv else None

    if daily_csv is None:
        candidates = sorted(RESULTS_DIR.glob("*_daily.csv"))
        daily_csv = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None
    if summary_csv is None:
        candidates = sorted(RESULTS_DIR.glob("*_summary.csv"))
        summary_csv = max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None

    written: list[Path] = []
    if daily_csv and daily_csv.exists():
        written.extend(plot_pipeline_model_graphs(daily_csv, summary_csv))
    llm_plot = plot_llm_graph(llm_csv)
    if llm_plot:
        written.append(llm_plot)

    if not written:
        print("No graphs generated.")
        return

    print("Generated graphs:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
