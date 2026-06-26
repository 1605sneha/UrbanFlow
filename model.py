import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import List

import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Dense,
    Dropout,
    Bidirectional
)

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)

from tensorflow.keras.optimizers import Adam


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UrbanFlowAI")


# BUG FIX 1: All fields in a @dataclass must carry type annotations to be
# treated as proper dataclass instance fields.  Previously, every attribute
# below (feature_columns, target_column, sequence_length, …) lacked a type
# annotation and was therefore an ordinary *class* variable, not a dataclass
# field.  That means they cannot be overridden per-instance, do not appear in
# __init__ / __repr__, and silently share mutable state (the list) across all
# instances.  Added correct type annotations throughout, and used
# field(default_factory=…) for the mutable list default.

@dataclass
class Config:
    csv_path: Path = Path("live_traffic.csv")

    # ── Columns that traffic_collector.py actually writes ──────────────────
    feature_columns: List[str] = field(default_factory=lambda: [
        "avg_speed_kmh",   # was "currentSpeed"  — collector field
        "route_index",     # was "freeFlowSpeed"  — no free-flow in ORS; use alternative index instead
        "distance_km",     # was "distance"
        "duration_min",    # was "duration"
        "hour",
        "weekday",
    ])

    target_column: str = "congestion"

    sequence_length: int = 10

    test_split: float = 0.2

    batch_size: int = 32
    epochs: int = 50
    learning_rate: float = 0.001

    model_output: str = "traffic_lstm.keras"

    feature_scaler_output: str = "feature_scaler.joblib"
    target_scaler_output: str  = "target_scaler.joblib"


cfg = Config()


def load_dataset():
    logger.info("Loading dataset...")

    df = pd.read_csv(cfg.csv_path)

    df = df.dropna()

    logger.info(f"Dataset shape: {df.shape}")

    X = df[cfg.feature_columns].values.astype(np.float32)

    y = df[[cfg.target_column]].values.astype(np.float32)

    return X, y


def normalize_data(X, y):
    logger.info("Normalizing data...")

    feature_scaler = MinMaxScaler()

    target_scaler = MinMaxScaler()

    X_scaled = feature_scaler.fit_transform(X)

    y_scaled = target_scaler.fit_transform(y)

    joblib.dump(
        feature_scaler,
        cfg.feature_scaler_output
    )

    joblib.dump(
        target_scaler,
        cfg.target_scaler_output
    )

    logger.info("Scalers saved")

    return (
        X_scaled,
        y_scaled,
        feature_scaler,
        target_scaler
    )


def create_sequences(features, targets):
    logger.info("Creating sequences...")

    X_seq = []
    y_seq = []

    seq_len = cfg.sequence_length

    for i in range(len(features) - seq_len):
        X_seq.append(
            features[i:i + seq_len]
        )

        y_seq.append(
            targets[i + seq_len]
        )

    X_seq = np.array(X_seq, dtype=np.float32)

    y_seq = np.array(y_seq, dtype=np.float32)

    logger.info(
        f"Sequence shape: {X_seq.shape}"
    )

    return X_seq, y_seq


def split_dataset(X, y):
    split_index = int(
        len(X) * (1 - cfg.test_split)
    )

    X_train = X[:split_index]
    y_train = y[:split_index]

    X_test = X[split_index:]
    y_test = y[split_index:]

    logger.info(
        f"Train shape: {X_train.shape}"
    )

    logger.info(
        f"Test shape: {X_test.shape}"
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def build_model(num_features):
    logger.info("Building model...")

    inp = Input(
        shape=(
            cfg.sequence_length,
            num_features
        )
    )

    x = Bidirectional(
        LSTM(
            128,
            return_sequences=True,
            dropout=0.2,
            recurrent_dropout=0.2
        )
    )(inp)

    x = LSTM(
        64,
        dropout=0.2,
        recurrent_dropout=0.2
    )(x)

    x = Dropout(0.3)(x)

    x = Dense(
        64,
        activation="relu"
    )(x)

    x = Dense(
        32,
        activation="relu"
    )(x)

    out = Dense(1)(x)

    model = Model(inp, out)

    model.compile(
        optimizer=Adam(
            learning_rate=cfg.learning_rate
        ),

        loss="mse",

        metrics=[
            "mae"
        ]
    )

    model.summary()

    return model


def evaluate_model(
    model,
    X_test,
    y_test,
    target_scaler
):
    logger.info("Evaluating model...")

    predictions = model.predict(
        X_test,
        verbose=0
    )

    predictions = target_scaler.inverse_transform(
        predictions
    )

    y_actual = target_scaler.inverse_transform(
        y_test
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_actual,
            predictions
        )
    )

    mae = mean_absolute_error(
        y_actual,
        predictions
    )

    r2 = r2_score(
        y_actual,
        predictions
    )

    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"MAE : {mae:.4f}")
    logger.info(f"R2  : {r2:.4f}")


def train_model():
    X, y = load_dataset()

    (
        X_scaled,
        y_scaled,
        feature_scaler,
        target_scaler
    ) = normalize_data(X, y)

    X_seq, y_seq = create_sequences(
        X_scaled,
        y_scaled
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_dataset(
        X_seq,
        y_seq
    )

    model = build_model(
        num_features=len(
            cfg.feature_columns
        )
    )

    callbacks = [
        EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True
        ),

        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=5,
            # BUG FIX 2: verbose=1 (integer) is deprecated in modern Keras;
            # the parameter expects a boolean.  Changed to verbose=True.
            verbose=True
        ),

        ModelCheckpoint(
            cfg.model_output,
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        )
    ]

    logger.info("Training started...")

    history = model.fit(
        X_train,
        y_train,

        validation_split=0.2,

        epochs=cfg.epochs,

        batch_size=cfg.batch_size,

        callbacks=callbacks,

        verbose=1
    )

    logger.info("Training complete")

    # BUG FIX 3: model.save() was called *after* evaluate_model().
    # If evaluation raises an exception the model would never be persisted.
    # Save first so the artefact is always written before any post-training
    # work that could fail.
    model.save(
        cfg.model_output
    )

    logger.info(
        f"Model saved: {cfg.model_output}"
    )

    evaluate_model(
        model,
        X_test,
        y_test,
        target_scaler
    )

    return history


if __name__ == "__main__":
    train_model()