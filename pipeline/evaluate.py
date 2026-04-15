"""
Evaluate pipeline predictions against actual outcomes.

For each (pred_date, ticker, model), compare the predicted direction (+1/-1)
against the actual next-day return direction.

Stores results in the `evaluations` table.

Usage:
    python -m pipeline.evaluate
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

DB_PATH = os.environ.get("PIPELINE_DB", "pipeline/db/pipeline.db")


def init_eval_table(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS evaluations (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            evaluated_at    TEXT NOT NULL,
            pred_date       TEXT NOT NULL,
            model_name      TEXT NOT NULL,
            n_samples       INTEGER,
            directional_acc REAL,
            rmse            REAL,
            mae             REAL,
            r2              REAL
        )
    """)
    conn.commit()


def get_unevaluated_dates(conn: sqlite3.Connection) -> list[str]:
    evaluated = {r[0] for r in conn.execute(
        "SELECT DISTINCT pred_date FROM evaluations"
    ).fetchall()}
    all_pred = {r[0] for r in conn.execute(
        "SELECT DISTINCT pred_date FROM predictions"
    ).fetchall()}
    return sorted(all_pred - evaluated)


def get_actual_returns(conn: sqlite3.Connection, date: str) -> pd.DataFrame:
    """
    Actual return for pred_date = log(close_on_date / close_on_prev_date).
    We join consecutive rows in the prices table.
    """
    query = """
        SELECT t.ticker,
               t.date  AS actual_date,
               t.close AS close_today,
               p.close AS close_prev,
               LOG(t.close / p.close) AS actual_return
        FROM   prices t
        JOIN   prices p ON t.ticker = p.ticker
                       AND p.date = (
                           SELECT MAX(q.date) FROM prices q
                           WHERE q.ticker = t.ticker AND q.date < t.date
                       )
        WHERE  t.date = ?
    """
    # SQLite doesn't have LOG(); compute in Python instead
    q = """
        SELECT t.ticker, t.date AS actual_date, t.close AS close_today, p.close AS close_prev
        FROM   prices t
        JOIN   prices p ON t.ticker = p.ticker
        WHERE  t.date = ?
          AND  p.date = (
              SELECT MAX(q.date) FROM prices q
              WHERE  q.ticker = t.ticker AND q.date < t.date
          )
    """
    df = pd.read_sql_query(q, conn, params=[date])
    if df.empty:
        return df
    df["actual_return"] = np.log(
        df["close_today"].astype(float) / df["close_prev"].astype(float).clip(lower=1e-6)
    )
    return df[["ticker", "actual_date", "actual_return"]]


def evaluate_date(conn: sqlite3.Connection, pred_date: str) -> None:
    actuals = get_actual_returns(conn, pred_date)
    if actuals.empty:
        print(f"  {pred_date}: no actual data yet, skipping")
        return

    preds_df = pd.read_sql_query(
        "SELECT ticker, model_name, pred_return, pred_direction FROM predictions WHERE pred_date=?",
        conn, params=[pred_date]
    )
    if preds_df.empty:
        return

    merged = preds_df.merge(actuals[["ticker", "actual_return"]], on="ticker", how="inner")
    if merged.empty:
        print(f"  {pred_date}: no matching tickers between predictions and actuals")
        return

    evaluated_at = datetime.utcnow().isoformat()
    for model_name, grp in merged.groupby("model_name"):
        y_true = grp["actual_return"].values.astype(float)
        y_pred = grp["pred_return"].values.astype(float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        y_true, y_pred = y_true[mask], y_pred[mask]
        if len(y_true) < 2:
            continue

        da = float(np.mean(np.sign(y_pred) == np.sign(y_true)))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        mae = float(mean_absolute_error(y_true, y_pred))
        r2 = float(r2_score(y_true, y_pred))

        conn.execute(
            """INSERT INTO evaluations (evaluated_at, pred_date, model_name, n_samples,
               directional_acc, rmse, mae, r2)
               VALUES (?,?,?,?,?,?,?,?)""",
            [evaluated_at, pred_date, model_name, int(mask.sum()), da, rmse, mae, r2],
        )
        print(f"  {pred_date} [{model_name}] n={mask.sum()} DA={da:.4f} R2={r2:.6f}")

    conn.commit()


def print_summary(conn: sqlite3.Connection) -> None:
    df = pd.read_sql_query(
        "SELECT model_name, AVG(directional_acc) AS mean_da, AVG(r2) AS mean_r2, "
        "AVG(rmse) AS mean_rmse, COUNT(*) AS n_days FROM evaluations GROUP BY model_name",
        conn
    )
    if df.empty:
        print("No evaluations yet.")
        return
    print("\n=== Pipeline Evaluation Summary ===")
    print(df.to_string(index=False, float_format="%.4f"))


def main() -> None:
    conn = sqlite3.connect(DB_PATH)
    init_eval_table(conn)

    pending = get_unevaluated_dates(conn)
    print(f"Pending evaluation dates: {len(pending)}")
    for date in pending:
        evaluate_date(conn, date)

    print_summary(conn)
    conn.close()


if __name__ == "__main__":
    main()
