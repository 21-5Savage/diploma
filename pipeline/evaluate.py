"""
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


def get_unevaluated_dates(
    conn: sqlite3.Connection,
    date_from: str | None = None,
    date_to: str | None = None,
) -> list[str]:
    evaluated = {r[0] for r in conn.execute(
        "SELECT DISTINCT pred_date FROM evaluations"
    ).fetchall()}
    all_pred = {r[0] for r in conn.execute(
        "SELECT DISTINCT pred_date FROM predictions"
    ).fetchall()}
    pending = sorted(all_pred - evaluated)
    if date_from is not None:
        pending = [d for d in pending if d >= date_from]
    if date_to is not None:
        pending = [d for d in pending if d <= date_to]
    return pending


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
            "DELETE FROM evaluations WHERE pred_date=? AND model_name=?",
            [pred_date, model_name],
        )
        conn.execute(
            """INSERT INTO evaluations (evaluated_at, pred_date, model_name, n_samples,
               directional_acc, rmse, mae, r2)
               VALUES (?,?,?,?,?,?,?,?)""",
            [evaluated_at, pred_date, model_name, int(mask.sum()), da, rmse, mae, r2],
        )
        print(f"  {pred_date} [{model_name}] n={mask.sum()} DA={da:.4f} R2={r2:.6f}")

    conn.commit()


def print_summary(
    conn: sqlite3.Connection,
    date_from: str | None = None,
    date_to: str | None = None,
) -> None:
    query = (
        "SELECT model_name, AVG(directional_acc) AS mean_da, AVG(r2) AS mean_r2, "
        "AVG(rmse) AS mean_rmse, COUNT(*) AS n_days FROM evaluations"
    )
    conditions = []
    params: list[str] = []
    if date_from is not None:
        conditions.append("pred_date >= ?")
        params.append(date_from)
    if date_to is not None:
        conditions.append("pred_date <= ?")
        params.append(date_to)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " GROUP BY model_name"
    df = pd.read_sql_query(query, conn, params=params)
    if df.empty:
        print("No evaluations yet.")
        return
    print("\n=== Pipeline Evaluation Summary ===")
    print(df.to_string(index=False, float_format="%.4f"))


def main(date_from: str | None = None, date_to: str | None = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    init_eval_table(conn)

    if date_from is not None or date_to is not None:
        query = "SELECT DISTINCT pred_date FROM predictions"
        conditions = []
        params: list[str] = []
        if date_from is not None:
            conditions.append("pred_date >= ?")
            params.append(date_from)
        if date_to is not None:
            conditions.append("pred_date <= ?")
            params.append(date_to)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY pred_date"
        pending = [r[0] for r in conn.execute(query, params).fetchall()]
    else:
        pending = get_unevaluated_dates(conn, date_from=date_from, date_to=date_to)
    print(f"Pending evaluation dates: {len(pending)}")
    for date in pending:
        evaluate_date(conn, date)

    print_summary(conn, date_from=date_from, date_to=date_to)
    conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--date-from", default=None, help="Evaluation start date (YYYY-MM-DD)")
    parser.add_argument("--date-to", default=None, help="Evaluation end date (YYYY-MM-DD)")
    args = parser.parse_args()
    main(date_from=args.date_from, date_to=args.date_to)
