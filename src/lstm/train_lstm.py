"""
Run:
python -m src.lstm.train_lstm
"""

import os

# Force TensorFlow to use CPU only.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import gc
import json
import math
import sqlite3
from dataclasses import asdict, dataclass
from typing import List, Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers

try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


@dataclass
class Config:
    db_path: str = "dataset/stock_prices_20y.db"
    table_name: str = "prices"
    n_splits: int = 5
    test_size: float = 0.2
    target_horizon: int = 5
    predict_target: str = "return"  # "return" or "price"
    random_state: int = 42

    sequence_length: int = 20
    batch_size: int = 256
    epochs: int = 20

    model_output_path: str = "artifacts/lstm_model.keras"
    scaler_output_path: str = "artifacts/lstm_scaler.pkl"
    config_output_path: str = "artifacts/lstm_config.json"

    feature_cols: tuple = ("open", "high", "low", "close", "adj_close", "volume")

    # Internal memory-control setting for scaler fitting only.
    scaler_fit_chunk_rows: int = 50000


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


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


def add_target(df: pd.DataFrame, horizon: int, predict_target: str) -> pd.DataFrame:
    out = []

    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        close = g["close"]

        if predict_target == "return":
            g["target"] = close.shift(-horizon) / close - 1.0
        elif predict_target == "price":
            g["target"] = close.shift(-horizon)
        else:
            raise ValueError("predict_target must be 'return' or 'price'")

        out.append(g)

    return pd.concat(out, ignore_index=True)


def split_train_test_per_ticker(
    df: pd.DataFrame, test_size: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for _, grp in df.groupby("ticker", sort=False):
        grp = grp.sort_values("date").reset_index(drop=True)
        split_idx = int(len(grp) * (1 - test_size))

        if split_idx <= 0 or split_idx >= len(grp):
            continue

        train_parts.append(grp.iloc[:split_idx].copy())
        test_parts.append(grp.iloc[split_idx:].copy())

    if not train_parts or not test_parts:
        raise ValueError("Train/test split produced empty data. Check dataset size and test_size.")

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)
    return train_df, test_df


def build_sequences(
    df: pd.DataFrame,
    feature_cols: List[str],
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_list = []
    y_list = []
    meta_list = []

    for ticker, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)

        if len(g) < sequence_length:
            continue

        feat = g[feature_cols].to_numpy(dtype=np.float32)
        target = g["target"].to_numpy(dtype=np.float32)
        dates = g["date"].to_numpy()
        closes = g["close"].to_numpy(dtype=np.float32)

        for i in range(sequence_length - 1, len(g)):
            if np.isnan(target[i]):
                continue

            x_seq = feat[i - sequence_length + 1 : i + 1]
            y_val = target[i]

            if np.isnan(x_seq).any():
                continue

            X_list.append(x_seq)
            y_list.append(y_val)
            meta_list.append((ticker, dates[i], closes[i]))

    if not X_list:
        raise ValueError("No sequences were created. Check data, sequence_length, and NaN values.")

    X = np.asarray(X_list, dtype=np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    meta = pd.DataFrame(meta_list, columns=["ticker", "date", "close"])
    return X, y, meta


def build_grouped_time_folds(meta: pd.DataFrame, n_splits: int):
    meta = meta.sort_values(["ticker", "date"]).reset_index(drop=True)
    folds = []

    for fold_num in range(n_splits):
        tr_idx_all = []
        va_idx_all = []

        for _, grp in meta.groupby("ticker", sort=False):
            idx = grp.index.to_numpy()
            n = len(idx)

            if n < (n_splits + 1):
                continue

            block = n // (n_splits + 1)
            remainder = n % (n_splits + 1)

            segment_sizes = np.full(n_splits + 1, block, dtype=int)
            segment_sizes[:remainder] += 1

            boundaries = []
            start = 0
            for seg_size in segment_sizes:
                end = start + seg_size
                boundaries.append((start, end))
                start = end

            va_start, va_end = boundaries[fold_num + 1]
            tr_local = idx[:va_start]
            va_local = idx[va_start:va_end]

            if len(tr_local) == 0 or len(va_local) == 0:
                continue

            tr_idx_all.extend(tr_local.tolist())
            va_idx_all.extend(va_local.tolist())

        if tr_idx_all and va_idx_all:
            folds.append(
                (
                    np.asarray(tr_idx_all, dtype=np.int64),
                    np.asarray(va_idx_all, dtype=np.int64),
                )
            )

    if not folds:
        raise ValueError("No CV folds were built. Try reducing n_splits or checking ticker history lengths.")

    return folds


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))

    try:
        r2 = float(r2_score(y_true, y_pred))
    except Exception:
        r2 = float("nan")

    direction_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "directional_accuracy": direction_acc,
    }


def build_lstm_model(sequence_length: int, n_features: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(sequence_length, n_features)),
            layers.LSTM(32),
            layers.Dropout(0.2),
            layers.Dense(16, activation="relu"),
            layers.Dense(1),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="mse",
        metrics=["mae"],
    )
    return model


def fit_scaler_on_indices(
    X_all: np.ndarray,
    indices: np.ndarray,
    chunk_rows: int,
) -> StandardScaler:
    """
    Fit StandardScaler incrementally to avoid allocating X_all[indices]
    as one huge dense copy in memory.
    """
    scaler = StandardScaler()
    n_features = X_all.shape[2]

    for start in range(0, len(indices), chunk_rows):
        batch_idx = indices[start : start + chunk_rows]
        x_chunk = X_all[batch_idx]  # small copy only
        x_chunk_2d = x_chunk.reshape(-1, n_features)
        scaler.partial_fit(x_chunk_2d)
        del x_chunk, x_chunk_2d

    return scaler


def make_sequence_dataset(
    X_all: np.ndarray,
    y_all: np.ndarray,
    indices: np.ndarray,
    scaler: StandardScaler,
    batch_size: int,
    sequence_length: int,
    n_features: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    """
    Build a re-iterable tf.data.Dataset that scales each batch on the fly.
    This avoids holding both raw and scaled full-fold tensors in memory.
    """
    mean_ = scaler.mean_.astype(np.float32)
    scale_ = scaler.scale_.astype(np.float32)
    scale_[scale_ == 0.0] = 1.0

    def gen():
        order = indices.copy()
        if shuffle:
            rng = np.random.default_rng(seed)
            rng.shuffle(order)

        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            xb = X_all[batch_idx].astype(np.float32, copy=True)
            yb = y_all[batch_idx].astype(np.float32, copy=False)

            xb -= mean_
            xb /= scale_

            yield xb, yb

    output_signature = (
        tf.TensorSpec(shape=(None, sequence_length, n_features), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def make_predict_dataset(
    X_all: np.ndarray,
    indices: np.ndarray,
    scaler: StandardScaler,
    batch_size: int,
    sequence_length: int,
    n_features: int,
) -> tf.data.Dataset:
    mean_ = scaler.mean_.astype(np.float32)
    scale_ = scaler.scale_.astype(np.float32)
    scale_[scale_ == 0.0] = 1.0

    def gen():
        for start in range(0, len(indices), batch_size):
            batch_idx = indices[start : start + batch_size]
            xb = X_all[batch_idx].astype(np.float32, copy=True)
            xb -= mean_
            xb /= scale_
            yield xb

    output_signature = tf.TensorSpec(
        shape=(None, sequence_length, n_features), dtype=tf.float32
    )

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    cfg = Config()
    set_seed(cfg.random_state)

    print("TensorFlow version:", tf.__version__)
    print("Visible GPUs:", tf.config.list_physical_devices("GPU"))

    print("Loading data...")
    df = load_prices_from_sqlite(cfg.db_path, cfg.table_name)

    print("Building target...")
    df = add_target(df, horizon=cfg.target_horizon, predict_target=cfg.predict_target)

    print("Splitting train/test per ticker...")
    train_df, test_df = split_train_test_per_ticker(df, test_size=cfg.test_size)
    print(f"Train rows: {len(train_df):,} | Test rows: {len(test_df):,}")

    print("Building sequences...")
    X_train_raw, y_train, meta_train = build_sequences(
        train_df,
        feature_cols=list(cfg.feature_cols),
        sequence_length=cfg.sequence_length,
    )

    print(f"Train sequences: {len(X_train_raw):,}")
    print(f"Tickers        : {meta_train['ticker'].nunique()}")

    print("Building grouped time-series CV folds...")
    folds = build_grouped_time_folds(meta_train, n_splits=cfg.n_splits)
    print(f"Number of CV folds built: {len(folds)}")

    n_features = len(cfg.feature_cols)
    cv_results = []

    for fold_i, (tr_idx, va_idx) in enumerate(folds, start=1):
        print(f"\n=== Fold {fold_i}/{len(folds)} ===")
        print(f"Train seq: {len(tr_idx):,} | Valid seq: {len(va_idx):,}")

        scaler = fit_scaler_on_indices(
            X_all=X_train_raw,
            indices=tr_idx,
            chunk_rows=cfg.scaler_fit_chunk_rows,
        )

        train_ds = make_sequence_dataset(
            X_all=X_train_raw,
            y_all=y_train,
            indices=tr_idx,
            scaler=scaler,
            batch_size=cfg.batch_size,
            sequence_length=cfg.sequence_length,
            n_features=n_features,
            shuffle=False,  # preserve prior behavior
            seed=cfg.random_state + fold_i,
        )

        valid_ds = make_sequence_dataset(
            X_all=X_train_raw,
            y_all=y_train,
            indices=va_idx,
            scaler=scaler,
            batch_size=cfg.batch_size,
            sequence_length=cfg.sequence_length,
            n_features=n_features,
            shuffle=False,
            seed=cfg.random_state + 10_000 + fold_i,
        )

        model = build_lstm_model(cfg.sequence_length, n_features)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=3,
                restore_best_weights=True,
            )
        ]

        model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=cfg.epochs,
            verbose=1,
            callbacks=callbacks,
        )

        pred_ds = make_predict_dataset(
            X_all=X_train_raw,
            indices=va_idx,
            scaler=scaler,
            batch_size=cfg.batch_size,
            sequence_length=cfg.sequence_length,
            n_features=n_features,
        )
        pred_va = model.predict(pred_ds, verbose=0).ravel()
        y_va = y_train[va_idx]

        metrics = regression_metrics(y_va, pred_va)
        cv_results.append(metrics)

        print(
            f"Fold {fold_i} | "
            f"RMSE={metrics['rmse']:.6f}, "
            f"MAE={metrics['mae']:.6f}, "
            f"R2={metrics['r2']:.6f}, "
            f"DirAcc={metrics['directional_accuracy']:.4f}"
        )

        keras.backend.clear_session()
        del train_ds, valid_ds, pred_ds, y_va, pred_va, scaler, model
        gc.collect()

    print("\nCV Summary")
    cv_df = pd.DataFrame(cv_results)
    print(cv_df.mean(numeric_only=True).to_string())

    print("\nTraining final model on all training sequences...")
    all_train_idx = np.arange(len(X_train_raw), dtype=np.int64)

    scaler = fit_scaler_on_indices(
        X_all=X_train_raw,
        indices=all_train_idx,
        chunk_rows=cfg.scaler_fit_chunk_rows,
    )

    final_train_ds = make_sequence_dataset(
        X_all=X_train_raw,
        y_all=y_train,
        indices=all_train_idx,
        scaler=scaler,
        batch_size=cfg.batch_size,
        sequence_length=cfg.sequence_length,
        n_features=n_features,
        shuffle=False,  # preserve prior behavior
        seed=cfg.random_state,
    )

    final_model = build_lstm_model(cfg.sequence_length, n_features)

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="loss",
            patience=2,
            restore_best_weights=True,
        )
    ]

    final_model.fit(
        final_train_ds,
        epochs=cfg.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    os.makedirs("artifacts", exist_ok=True)

    final_model.save(cfg.model_output_path)
    joblib.dump(scaler, cfg.scaler_output_path)

    config_dict = asdict(cfg)
    config_dict["feature_cols"] = list(config_dict["feature_cols"])

    with open(cfg.config_output_path, "w", encoding="utf-8") as f:
        json.dump(config_dict, f, indent=2)

    print("\nSaved artifacts:")
    print(" -", cfg.model_output_path)
    print(" -", cfg.scaler_output_path)
    print(" -", cfg.config_output_path)


if __name__ == "__main__":
    main()
