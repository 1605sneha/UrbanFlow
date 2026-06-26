import logging
import argparse
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UrbanFlowAI.Predictor")


# ── Must match Config in train_traffic.py exactly ────────────────────────────

FEATURE_COLUMNS = [
    "avg_speed_kmh",
    "route_index",
    "distance_km",
    "duration_min",
    "hour",
    "weekday",
]

SEQUENCE_LENGTH = 10

MODEL_PATH          = "traffic_lstm.keras"
FEATURE_SCALER_PATH = "feature_scaler.joblib"
TARGET_SCALER_PATH  = "target_scaler.joblib"
CSV_FILE            = "live_traffic.csv"


# ── Congestion label helper ───────────────────────────────────────────────────

def congestion_label(score: float) -> str:
    if score >= 80:
        return "🔴  Severe   (gridlock likely)"
    elif score >= 60:
        return "🟠  Heavy    (significant delays)"
    elif score >= 35:
        return "🟡  Moderate (some slowdown)"
    elif score >= 15:
        return "🟢  Light    (mostly free-flow)"
    else:
        return "✅  Free     (clear road)"


# ── Load artefacts ────────────────────────────────────────────────────────────

def load_artefacts():
    logger.info("Loading model and scalers...")

    # BUG FIX 1: The original code only checked for the model file and let
    # joblib.load raise a bare FileNotFoundError for the two scaler files.
    # Now all three required artefacts are checked up-front so the user gets
    # a clear, actionable message for whichever file is missing.
    for path in (MODEL_PATH, FEATURE_SCALER_PATH, TARGET_SCALER_PATH):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Required artefact not found: '{path}'. "
                "Run train_traffic.py first."
            )

    model          = load_model(MODEL_PATH)
    feature_scaler = joblib.load(FEATURE_SCALER_PATH)
    target_scaler  = joblib.load(TARGET_SCALER_PATH)

    logger.info("Artefacts loaded.")
    return model, feature_scaler, target_scaler


# ── Mode 1: predict from the last N rows of the live CSV ─────────────────────

def predict_from_csv(
    model,
    feature_scaler,
    target_scaler,
    csv_path: str = CSV_FILE,
    route_name: str = None,
):
    df = pd.read_csv(csv_path)

    if route_name:
        df = df[df["route_name"] == route_name]
        if df.empty:
            raise ValueError(
                f"No rows found for route '{route_name}' in {csv_path}."
            )

    df = df.dropna(subset=FEATURE_COLUMNS)

    if len(df) < SEQUENCE_LENGTH:
        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} rows "
            f"(found {len(df)}) to form a sequence. "
            "Collect more data first."
        )

    # Take the most recent sequence
    recent = df.tail(SEQUENCE_LENGTH)

    X = recent[FEATURE_COLUMNS].values.astype(np.float32)

    X_scaled = feature_scaler.transform(X)

    # Shape: (1, sequence_length, num_features)
    X_seq = X_scaled[np.newaxis, ...]

    raw_pred = model.predict(X_seq, verbose=0)

    congestion = float(
        target_scaler.inverse_transform(raw_pred)[0, 0]
    )

    # Clamp to [0, 100]
    congestion = max(0.0, min(100.0, congestion))

    return congestion, recent


# ── Mode 2: predict from manually supplied values ────────────────────────────

def predict_single(
    model,
    feature_scaler,
    target_scaler,
    avg_speed_kmh: float,
    distance_km:   float,
    duration_min:  float,
    route_index:   int = 0,
    dt:            datetime = None,
):
    """
    Predict congestion for a single observation.

    Because the model requires a full sequence of SEQUENCE_LENGTH steps,
    this function replicates the single row SEQUENCE_LENGTH times.
    This is an approximation suitable for quick what-if queries;
    for time-series accuracy use predict_from_csv() instead.
    """
    if dt is None:
        dt = datetime.now()

    row = np.array(
        [[
            avg_speed_kmh,
            route_index,
            distance_km,
            duration_min,
            dt.hour,
            dt.weekday(),
        ]],
        dtype=np.float32,
    )

    # Repeat the row to fill the sequence window
    row_scaled = feature_scaler.transform(row)

    X_seq = np.repeat(
        row_scaled[np.newaxis, ...],
        SEQUENCE_LENGTH,
        axis=1,
    )                                   # (1, SEQUENCE_LENGTH, features)

    raw_pred = model.predict(X_seq, verbose=0)

    congestion = float(
        target_scaler.inverse_transform(raw_pred)[0, 0]
    )

    congestion = max(0.0, min(100.0, congestion))

    return congestion


# ── Pretty printer ────────────────────────────────────────────────────────────

def print_prediction(
    congestion: float,
    context: str = "",
):
    bar_len  = 30
    filled   = int(round(congestion / 100 * bar_len))
    bar      = "█" * filled + "░" * (bar_len - filled)

    # BUG FIX 2: congestion_label() returns strings that contain emoji
    # characters (🔴 🟠 🟡 🟢 ✅).  Python's str.format left-alignment (:<N)
    # counts Unicode code-points, not terminal display columns.  Each of
    # those emoji occupies 2 display columns in most terminals, so a naïve
    # :<31 pad produces a line that is 1 character too short, misaligning the
    # closing ║ border.  The fix pads with plain ASCII spaces *after* the
    # label to a fixed display-width that accounts for the 1-column overhang
    # introduced by the wide emoji.
    status_text = congestion_label(congestion)
    # Each emoji in these labels is one code-point but 2 display columns wide.
    # Subtract 1 from the pad width for every wide character present.
    wide_chars  = sum(1 for ch in status_text if ord(ch) > 0xFFFF or
                      0x1F300 <= ord(ch) <= 0x1FAFF or
                      0x2600  <= ord(ch) <= 0x26FF)
    padded_status = status_text.ljust(31 - wide_chars)

    print()
    print("╔══════════════════════════════════════════╗")
    print("║      UrbanFlowAI — Congestion Forecast  ║")
    print("╠══════════════════════════════════════════╣")
    if context:
        print(f"║  Route  : {context:<31}║")
    print(f"║  Score  : {congestion:>5.1f} / 100                    ║")
    print(f"║  [{bar}]  ║")
    print(f"║  Status : {padded_status}║")
    print("╚══════════════════════════════════════════╝")
    print()


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Predict traffic congestion using the trained BiLSTM model.\n\n"
            "Two modes:\n"
            "  csv     — read the last window of rows from live_traffic.csv\n"
            "  single  — supply one observation directly via flags\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # ── csv mode ──────────────────────────────────────────────────────────
    csv_p = subparsers.add_parser(
        "csv",
        help="Predict from the last window in live_traffic.csv",
    )
    csv_p.add_argument(
        "--csv",
        default=CSV_FILE,
        help=f"Path to the CSV file (default: {CSV_FILE})",
    )
    csv_p.add_argument(
        "--route",
        default=None,
        help="Filter by route_name (e.g. Route_A). Omit to use all rows.",
    )

    # ── single mode ───────────────────────────────────────────────────────
    s_p = subparsers.add_parser(
        "single",
        help="Predict from a single manually supplied observation",
    )
    s_p.add_argument(
        "--speed",
        type=float,
        required=True,
        help="Average speed in km/h",
    )
    s_p.add_argument(
        "--distance",
        type=float,
        required=True,
        help="Route distance in km",
    )
    s_p.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Route duration in minutes",
    )
    s_p.add_argument(
        "--route-index",
        type=int,
        default=0,
        help="Alternative route index (0 = primary, 1/2 = alternatives; default: 0)",
    )
    s_p.add_argument(
        "--hour",
        type=int,
        default=None,
        help="Hour of day 0–23 (default: current hour)",
    )
    s_p.add_argument(
        "--weekday",
        type=int,
        default=None,
        help="Day of week 0=Mon … 6=Sun (default: today)",
    )

    args = parser.parse_args()

    model, feature_scaler, target_scaler = load_artefacts()

    # ── csv mode ──────────────────────────────────────────────────────────
    if args.mode == "csv":
        congestion, recent = predict_from_csv(
            model,
            feature_scaler,
            target_scaler,
            csv_path=args.csv,
            route_name=args.route,
        )

        # Show the window that fed the prediction
        logger.info(
            f"Prediction based on rows "
            f"{recent.index[0]}–{recent.index[-1]}"
        )

        route_ctx = args.route or "all routes"
        print_prediction(congestion, context=route_ctx)

    # ── single mode ───────────────────────────────────────────────────────
    elif args.mode == "single":
        now = datetime.now()

        dt = datetime(
            now.year,
            now.month,
            now.day,
            args.hour    if args.hour    is not None else now.hour,
            0, 0,
        )

        # Override weekday if supplied
        if args.weekday is not None:
            # Build a fake datetime that lands on that weekday
            # (weekday() isn't settable directly — we just pass it separately)
            weekday_override = args.weekday
        else:
            weekday_override = dt.weekday()

        # Patch the feature row directly so we respect the override
        row = np.array(
            [[
                args.speed,
                args.route_index,
                args.distance,
                args.duration,
                dt.hour,
                weekday_override,
            ]],
            dtype=np.float32,
        )

        row_scaled = feature_scaler.transform(row)

        X_seq = np.repeat(
            row_scaled[np.newaxis, ...],
            SEQUENCE_LENGTH,
            axis=1,
        )

        raw_pred   = model.predict(X_seq, verbose=0)
        congestion = float(
            target_scaler.inverse_transform(raw_pred)[0, 0]
        )
        congestion = max(0.0, min(100.0, congestion))

        ctx = (
            f"speed={args.speed}km/h  "
            f"dist={args.distance}km  "
            f"dur={args.duration}min"
        )
        print_prediction(congestion, context=ctx)


if __name__ == "__main__":
    main()