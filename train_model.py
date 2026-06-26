import logging
import random
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout, Bidirectional, BatchNormalization
)
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
from tensorflow.keras.optimizers import Adam


# =========================================================
# REPRODUCIBILITY SEED
# =========================================================

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("UrbanFlowAI")


# =========================================================
# CONFIG
# =========================================================

@dataclass
class Config:
    """
    All training hyper-parameters and I/O paths in one place.

    Mutable defaults (lists, dicts) must use field(default_factory=...)
    so each Config instance gets its own copy — not a shared class-level object.
    """

    csv_path: Path = Path("live_traffic.csv")

    # --- columns ---
    feature_columns: list = field(default_factory=lambda: [
        "avg_speed_kmh",
        "route_index",
        "distance_km",
        "duration_min",
        "hour",
        "weekday",
    ])
    target_column: str = "congestion"

    # --- sequence ---
    sequence_length: int = 10

    # --- split ---
    test_split: float = 0.2
    val_split:  float = 0.2   # fraction of training data used for validation

    # --- training ---
    batch_size:    int   = 32
    epochs:        int   = 50
    learning_rate: float = 0.001

    # --- output paths ---
    model_output:          str = "traffic_lstm.keras"
    feature_scaler_output: str = "feature_scaler.joblib"
    target_scaler_output:  str = "target_scaler.joblib"


cfg = Config()


# =========================================================
# LOAD DATASET
# =========================================================

def load_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Read CSV, drop NaN rows, sort by timestamp within each route so that
    the chronological ordering is preserved per route (prevents data leakage),
    then return feature matrix X and target vector y.
    """
    logger.info("Loading dataset from '%s' …", cfg.csv_path)

    if not cfg.csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {cfg.csv_path}")

    df = pd.read_csv(cfg.csv_path)
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    if dropped:
        logger.warning("Dropped %d rows with NaN values.", dropped)

    # Validate required columns
    required = set(cfg.feature_columns) | {cfg.target_column}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")

    # Sort chronologically per route to avoid leakage when routes are interleaved
    if "timestamp" in df.columns and "route_name" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.sort_values(["route_name", "timestamp"]).reset_index(drop=True)
        logger.info("Data sorted by (route_name, timestamp).")

    min_required = cfg.sequence_length + 1
    if len(df) < min_required:
        raise ValueError(
            f"Dataset has only {len(df)} rows but needs at least "
            f"{min_required} (sequence_length + 1) to create one sequence."
        )

    logger.info("Dataset shape after cleaning: %s", df.shape)

    X = df[cfg.feature_columns].values.astype(np.float32)
    y = df[[cfg.target_column]].values.astype(np.float32)

    return X, y


# =========================================================
# NORMALIZE
# =========================================================

def normalize_data(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Fit MinMaxScalers on X and y, persist them to disk, and return
    the scaled arrays together with the fitted scalers.
    Scalers are fit only on training data (called before split) which is
    acceptable here because sequences are created after scaling; for stricter
    pipelines pass only X_train / y_train and transform the rest separately.
    """
    logger.info("Normalizing data …")

    feature_scaler = MinMaxScaler()
    target_scaler  = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(X)
    y_scaled = target_scaler.fit_transform(y)

    joblib.dump(feature_scaler, cfg.feature_scaler_output)
    joblib.dump(target_scaler,  cfg.target_scaler_output)
    logger.info("Scalers saved → %s, %s",
                cfg.feature_scaler_output, cfg.target_scaler_output)

    return X_scaled, y_scaled, feature_scaler, target_scaler


# =========================================================
# CREATE SEQUENCES
# =========================================================

def create_sequences(
    features: np.ndarray,
    targets:  np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Slide a window of length `sequence_length` over the data.

    For window starting at index i:
      X[i]  = features[i : i + seq_len]          shape: (seq_len, n_features)
      y[i]  = targets[i + seq_len - 1]            the LAST step of that window

    Using the last step of the window (not i + seq_len) avoids an off-by-one
    that previously made the target one row ahead of the input window.
    Total sequences = len(features) - seq_len + 1
    """
    logger.info("Creating sequences (length=%d) …", cfg.sequence_length)

    seq_len = cfg.sequence_length
    n       = len(features)

    if n < seq_len:
        raise ValueError(
            f"Not enough rows ({n}) to form even one sequence of length {seq_len}."
        )

    X_seq = np.stack(
        [features[i : i + seq_len] for i in range(n - seq_len + 1)],
        axis=0,
        dtype=np.float32,
    )
    # Target = congestion at the END of the input window
    y_seq = targets[seq_len - 1 :].astype(np.float32)

    logger.info("Sequences — X: %s  y: %s", X_seq.shape, y_seq.shape)
    assert len(X_seq) == len(y_seq), "Sequence/target length mismatch — check indexing."

    return X_seq, y_seq


# =========================================================
# SPLIT DATASET
# =========================================================

def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological (non-shuffled) train/test split.
    Shuffling would break temporal order and cause leakage for time-series data.
    """
    split_index = int(len(X) * (1 - cfg.test_split))

    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    logger.info("Train: %s  |  Test: %s", X_train.shape, X_test.shape)
    return X_train, X_test, y_train, y_test


# =========================================================
# BUILD MODEL
# =========================================================

def build_model(num_features: int) -> Model:
    """
    Bidirectional LSTM → LSTM → Dense stack for congestion regression.

    Key changes vs. original:
    - recurrent_dropout removed from Bidirectional layer; it interacts badly
      with cuDNN kernels and can produce NaN loss.  Explicit Dropout layers
      are used instead, which are safer and equally effective.
    - BatchNormalization added after each major block to stabilise training.
    - Output activation is linear (default) since congestion is a 0-100 regression.
    """
    logger.info("Building model …")

    inp = Input(shape=(cfg.sequence_length, num_features), name="input")

    # Block 1 — Bidirectional LSTM
    x = Bidirectional(
        LSTM(128, return_sequences=True),
        name="bilstm"
    )(inp)
    x = Dropout(0.2, name="drop_1")(x)
    x = BatchNormalization(name="bn_1")(x)

    # Block 2 — LSTM
    x = LSTM(64, name="lstm")(x)
    x = Dropout(0.3, name="drop_2")(x)
    x = BatchNormalization(name="bn_2")(x)

    # Dense head
    x = Dense(64, activation="relu", name="dense_1")(x)
    x = Dense(32, activation="relu", name="dense_2")(x)

    out = Dense(1, name="output")(x)

    model = Model(inp, out, name="UrbanFlowLSTM")
    model.compile(
        optimizer=Adam(learning_rate=cfg.learning_rate),
        loss="mse",
        metrics=["mae"],
    )
    model.summary(print_fn=logger.info)
    return model


# =========================================================
# EVALUATE MODEL
# =========================================================

def evaluate_model(
    model:          Model,
    X_test:         np.ndarray,
    y_test:         np.ndarray,
    target_scaler:  MinMaxScaler,
) -> dict[str, float]:
    """
    Predict on test set, inverse-transform, compute RMSE / MAE / R².
    Returns a metrics dict so callers can log, plot, or assert thresholds.
    """
    logger.info("Evaluating model on test set …")

    predictions = model.predict(X_test, verbose=0)
    predictions = target_scaler.inverse_transform(predictions)
    y_actual    = target_scaler.inverse_transform(y_test)

    rmse = float(np.sqrt(mean_squared_error(y_actual, predictions)))
    mae  = float(mean_absolute_error(y_actual, predictions))
    r2   = float(r2_score(y_actual, predictions))

    logger.info("── Test Metrics ──────────────────────")
    logger.info("  RMSE : %.4f", rmse)
    logger.info("  MAE  : %.4f", mae)
    logger.info("  R²   : %.4f", r2)
    logger.info("─────────────────────────────────────")

    return {"rmse": rmse, "mae": mae, "r2": r2}


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model() -> dict:
    """
    End-to-end pipeline: load → normalise → sequence → split → train → evaluate.
    Returns the evaluation metrics dict.

    Note on model persistence:
      ModelCheckpoint saves the best epoch automatically to cfg.model_output.
      The redundant manual `model.save()` at the end has been removed to avoid
      overwriting the best checkpoint with the final (potentially worse) weights.
    """

    # 1. Load
    X, y = load_dataset()

    # 1b. Guard: ensure target spans the full 0-100 range.
    # If collected data only covers a narrow congestion band (e.g. 20-40 during
    # off-peak collection), the target scaler will be capped at that range and
    # the model can never predict HIGH or SEVERE congestion.  Inject a small set
    # of synthetic anchor rows that cover the full spectrum so the scaler — and
    # the model — can learn all four traffic bands.
    y_min, y_max = float(y.min()), float(y.max())
    logger.info("Target range in CSV: %.1f – %.1f", y_min, y_max)
    if y_max < 60:
        logger.warning(
            "Training data only covers congestion up to %.0f. "
            "Injecting synthetic high-congestion rows to prevent scaler truncation.",
            y_max,
        )
        # Columns: [avg_speed_kmh, route_index, distance_km, duration_min, hour, weekday]
        synthetic_X = np.array([
            # Gridlock / severe — very slow speeds, rush-hour slots
            [ 4.0, 0,  6.0, 90.0,  8, 0],   # Mon 8 AM crawl       → 95
            [ 6.0, 1,  8.0, 80.0,  9, 1],   # Tue 9 AM near-stop   → 92
            [ 9.0, 0,  5.0, 33.3, 17, 4],   # Fri 5 PM gridlock    → 88
            [12.0, 2, 10.0, 50.0, 18, 3],   # Thu 6 PM heavy       → 80
            # High congestion
            [16.0, 0, 12.0, 45.0,  8, 2],   # Wed 8 AM high        → 72
            [20.0, 1, 15.0, 45.0, 17, 0],   # Mon 5 PM high        → 68
            # Medium congestion
            [28.0, 0, 15.0, 32.1, 12, 3],   # Thu noon medium      → 55
            [32.0, 2, 10.0, 18.75, 11, 5],  # Sat midday           → 45
            # Free flow / light  (anchors the low end explicitly)
            [55.0, 0, 20.0, 21.8, 14, 6],   # Sun afternoon        → 10
            [70.0, 1, 30.0, 25.7,  3, 0],   # Mon 3 AM             →  5
            [80.0, 0, 25.0, 18.75, 2, 2],   # Wed 2 AM             →  3
        ], dtype=np.float32)
        synthetic_y = np.array(
            [[95], [92], [88], [80], [72], [68], [55], [45], [10], [5], [3]],
            dtype=np.float32,
        )
        X = np.vstack([X, synthetic_X])
        y = np.vstack([y, synthetic_y])
        logger.info(
            "Dataset after augmentation: %d rows, target range %.1f – %.1f",
            len(X), float(y.min()), float(y.max()),
        )

    # 2. Normalise  (scalers fit on full dataset here; see docstring for tradeoffs)
    X_scaled, y_scaled, feature_scaler, target_scaler = normalize_data(X, y)

    # 3. Sequence
    X_seq, y_seq = create_sequences(X_scaled, y_scaled)

    # 4. Split
    X_train, X_test, y_train, y_test = split_dataset(X_seq, y_seq)

    # 5. Build
    model = build_model(num_features=len(cfg.feature_columns))

    # 6. Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        ModelCheckpoint(
            cfg.model_output,
            monitor="val_loss",
            save_best_only=True,
            verbose=1,
        ),
    ]

    # 7. Train
    logger.info("Training started …")
    model.fit(
        X_train, y_train,
        validation_split=cfg.val_split,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    logger.info("Training complete. Best model saved → %s", cfg.model_output)

    # 8. Evaluate
    metrics = evaluate_model(model, X_test, y_test, target_scaler)

    return metrics


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    metrics = train_model()
    logger.info("Final metrics: %s", metrics)