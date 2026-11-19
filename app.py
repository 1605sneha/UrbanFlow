import logging
from datetime import datetime

import joblib
import numpy as np
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS
from tensorflow.keras.models import load_model


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UrbanFlowAI")


# =========================================================
# FLASK SETUP
# =========================================================

app = Flask(__name__)
CORS(app)


# =========================================================
# CONFIG
# =========================================================

ORS_API_KEY = (
    "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6"
    "IjU4YWFkYTYxM2RiNTQyMzI4NTI1YzBiZmU3OWM1MGJmIiwiaCI6"
    "Im11cm11cjY0In0="
)

ORS_DIRECTIONS_URL = (
    "https://api.openrouteservice.org"
    "/v2/directions/driving-car/geojson"
)

ORS_HEALTH_URL = (
    "https://api.openrouteservice.org/ors/health"
)

DEFAULT_FREE_FLOW_SPEED = 50.0

MODEL_PATH          = "traffic_lstm.keras"
FEATURE_SCALER_PATH = "feature_scaler.joblib"
TARGET_SCALER_PATH  = "target_scaler.joblib"

SEQUENCE_LENGTH = 10

# Matches Config.feature_columns in train_model.py exactly:
# ["avg_speed_kmh", "route_index", "distance_km", "duration_min", "hour", "weekday"]
NUM_FEATURES = 6

ORS_HEADERS = {
    "Authorization": ORS_API_KEY,
    "Content-Type":  "application/json",
    "Accept":        "application/json, application/geo+json",
}


# =========================================================
# LOAD MODEL + SCALERS  (once at startup)
# BUG FIX 1: The original code called load_model() and joblib.load() at
# module level with no error handling.  If the model or scaler files are
# absent (e.g. training hasn't been run yet), the entire Flask process
# crashes at import time with an unhelpful traceback.  Wrapping the load
# in a try/except lets the app start and return a meaningful error on every
# request instead of refusing to boot at all.
# =========================================================

try:
    MODEL          = load_model(MODEL_PATH)
    FEATURE_SCALER = joblib.load(FEATURE_SCALER_PATH)
    TARGET_SCALER  = joblib.load(TARGET_SCALER_PATH)
    logger.info("Model and scalers loaded.")
    _MODEL_READY = True
except Exception as _load_err:
    logger.error("Failed to load model/scalers: %s", _load_err)
    MODEL = FEATURE_SCALER = TARGET_SCALER = None
    _MODEL_READY = False


def _require_model():
    """Return a 503 response tuple if the model failed to load."""
    if not _MODEL_READY:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Run train_model.py first.",
        }), 503
    return None


# =========================================================
# TRAFFIC CLASSIFIER
# =========================================================

def classify_congestion(score: float) -> str:
    if score >= 70:
        return "SEVERE"
    elif score >= 50:
        return "HIGH"
    elif score >= 30:
        return "MEDIUM"
    return "LOW"


# =========================================================
# FETCH ROUTES FROM ORS
# =========================================================

def fetch_routes(origin: list, destination: list) -> list:
    body = {
        "coordinates": [origin, destination],
        "alternative_routes": {
            "target_count": 3,
            "weight_factor": 1.6,
        },
    }

    resp = requests.post(
        ORS_DIRECTIONS_URL,
        json=body,
        headers=ORS_HEADERS,
        timeout=15,
    )

    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(f"ORS error {resp.status_code}: {detail}")

    features = resp.json().get("features")
    if not features:
        raise RuntimeError("ORS returned no route features.")

    return features


# =========================================================
# DERIVE SPEED METRICS FROM ORS ROUTE SUMMARY
# =========================================================

def derive_speed_metrics(distance_km: float, duration_min: float) -> dict:
    if duration_min > 0:
        avg_speed = distance_km / (duration_min / 60)
    else:
        avg_speed = DEFAULT_FREE_FLOW_SPEED

    if avg_speed < 10:
        live_congestion = 95.0
    elif avg_speed < 15:
        live_congestion = 85.0
    elif avg_speed < 25:
        live_congestion = 65.0
    elif avg_speed < 40:
        live_congestion = 40.0
    elif avg_speed < 60:
        live_congestion = 20.0
    else:
        live_congestion = 10.0

    return {
        "current_speed":   round(avg_speed, 2),
        "free_flow_speed": DEFAULT_FREE_FLOW_SPEED,
        "live_congestion": live_congestion,
    }


# =========================================================
# BUILD FEATURE VECTOR
# =========================================================
# Signature matches Config.feature_columns in train_model.py:
# ["avg_speed_kmh", "route_index", "distance_km", "duration_min", "hour", "weekday"]

def build_feature_vector(
    avg_speed_kmh: float,
    route_index:   int,
    distance_km:   float,
    duration_min:  float,
    hour:          int,
    weekday:       int,
) -> list:
    return [
        float(avg_speed_kmh),
        float(route_index),
        float(distance_km),
        float(duration_min),
        float(hour),
        float(weekday),
    ]


def build_sequence(features: list) -> np.ndarray:
    """Tile a single feature row to (SEQUENCE_LENGTH, NUM_FEATURES)."""
    row = np.array(features, dtype=np.float32)
    return np.tile(row, (SEQUENCE_LENGTH, 1))


# =========================================================
# PREDICT CONGESTION
# =========================================================

def predict_congestion(sequence: np.ndarray) -> float:
    scaled = FEATURE_SCALER.transform(sequence)
    scaled = scaled.reshape(1, SEQUENCE_LENGTH, NUM_FEATURES)
    raw    = MODEL.predict(scaled, verbose=0)
    return float(TARGET_SCALER.inverse_transform(raw)[0][0])


# =========================================================
# EVALUATE ROUTES
# =========================================================

def evaluate_routes(routes: list, hour: int, weekday: int) -> list:
    evaluated = []

    for idx, route in enumerate(routes):
        summary      = route["properties"]["summary"]
        distance_km  = summary["distance"] / 1000
        duration_min = summary["duration"] / 60

        speed_metrics   = derive_speed_metrics(distance_km, duration_min)
        current_speed   = speed_metrics["current_speed"]
        live_congestion = speed_metrics["live_congestion"]

        features = build_feature_vector(
            avg_speed_kmh=current_speed,
            route_index=idx,
            distance_km=distance_km,
            duration_min=duration_min,
            hour=hour,
            weekday=weekday,
        )

        sequence             = build_sequence(features)
        predicted_congestion = predict_congestion(sequence)
        traffic_level        = classify_congestion(predicted_congestion)

        route_score = (
            predicted_congestion * 0.7 +
            duration_min         * 0.2 +
            distance_km          * 0.1
        )

        evaluated.append({
            "route_index":          idx,
            "distance_km":          round(distance_km,          2),
            "duration_min":         round(duration_min,         2),
            "avg_speed":            round(current_speed,        2),
            "live_speed":           current_speed,
            "free_flow_speed":      DEFAULT_FREE_FLOW_SPEED,
            "live_congestion":      round(live_congestion,      2),
            "predicted_congestion": round(predicted_congestion, 2),
            "traffic_level":        traffic_level,
            "route_score":          round(route_score,          2),
            "geometry":             route["geometry"],
            "rerouted":             False,
        })

    return evaluated


# =========================================================
# SELECT BEST ROUTE
# =========================================================

def select_best_route(routes: list):
    routes.sort(key=lambda r: (r["route_score"], r["duration_min"]))
    best = routes[0]
    if best["route_index"] != 0:
        best["rerouted"] = True
    return best, routes


# =========================================================
# /optimize-route
# =========================================================

@app.route("/optimize-route", methods=["POST"])
def optimize_route():
    err = _require_model()
    if err:
        return err

    try:
        data        = request.get_json(force=True)
        origin      = data["origin"]
        destination = data["destination"]

        now = datetime.now()

        # BUG FIX 2: When "weekday" or "hour" are absent from the request
        # the original code fell back to hardcoded values (weekday default=1
        # → always Monday; hour default=12 → always noon).  Missing temporal
        # fields should default to the *current* wall-clock values so that
        # omitting them is a sensible no-op rather than silently injecting
        # wrong data into the prediction.
        js_weekday = int(data.get("weekday", (now.weekday() + 1) % 7))  # Python→JS: Mon=0→1, Sun=6→0
        hour       = int(data.get("hour",    now.hour))
        weekday    = (js_weekday + 6) % 7  # JS 0=Sun → Python 0=Mon

        logger.info(
            "optimize-route  origin=%s  dest=%s  hour=%d  weekday=%d",
            origin, destination, hour, weekday,
        )

        routes                 = fetch_routes(origin, destination)
        evaluated_routes       = evaluate_routes(routes, hour, weekday)
        best_route, all_routes = select_best_route(evaluated_routes)

        return jsonify({
            "success":    True,
            "rerouted":   best_route["rerouted"],
            "best_route": best_route,
            "all_routes": all_routes,
        })

    except KeyError as exc:
        logger.warning("Missing field: %s", exc)
        return jsonify({"success": False, "error": f"Missing field: {exc}"}), 400

    except RuntimeError as exc:
        logger.error("ORS error: %s", exc)
        return jsonify({"success": False, "error": str(exc)}), 502

    except Exception:
        logger.exception("Unexpected error in /optimize-route")
        return jsonify({"success": False, "error": "Internal server error"}), 500


# =========================================================
# /predict
# =========================================================

@app.route("/predict", methods=["POST"])
def predict():
    err = _require_model()
    if err:
        return err

    try:
        data     = request.get_json(force=True)
        sequence = data.get("sequence", [])

        if not sequence:
            return jsonify({"error": "Empty sequence"}), 400

        now     = datetime.now()
        hour    = now.hour
        weekday = now.weekday()

        avg_speed = float(np.mean(sequence))

        features = build_feature_vector(
            avg_speed_kmh=avg_speed,
            route_index=0,
            distance_km=10.0,   # neutral placeholder
            duration_min=20.0,  # neutral placeholder
            hour=hour,
            weekday=weekday,
        )

        seq        = build_sequence(features)
        congestion = predict_congestion(seq)
        level      = classify_congestion(congestion)

        label_map = {
            "SEVERE": ("Heavy Traffic",    "🔴", "critical", "Avoid this route if possible."),
            "HIGH":   ("High Traffic",     "🟠", "critical", "Expect significant delays."),
            "MEDIUM": ("Moderate Traffic", "🟡", "moderate", "Allow extra travel time."),
            "LOW":    ("Light Traffic",    "🟢", "normal",   "Traffic is smooth — good to go."),
        }
        label, emoji, severity, advice = label_map[level]

        return jsonify({
            "congestion": round(congestion, 2),
            "traffic": {
                "label":    label,
                "emoji":    emoji,
                "severity": severity,
                "advice":   advice,
            },
        })

    except Exception:
        logger.exception("Error in /predict")
        return jsonify({"error": "Internal server error"}), 500


# =========================================================
# /health
# =========================================================

@app.route("/health", methods=["GET"])
def health():
    try:
        r = requests.get(ORS_HEALTH_URL, timeout=4)
        ors_ok = r.ok
    except Exception:
        ors_ok = False

    return jsonify({
        "status":        "running",
        "model_loaded":  _MODEL_READY,
        "ors_connected": ors_ok,
    })


# =========================================================
# SERVER
# =========================================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)