import requests
import pandas as pd
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
 
# ─── CONFIG ──────────────────────────────────────────────────────────────────
API_KEY = "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjU4YWFkYTYxM2RiNTQyMzI4NTI1YzBiZmU3OWM1MGJmIiwiaCI6Im11cm11cjY0In0="
ORS_DIRECTIONS_URL = "https://api.openrouteservice.org/v2/directions/driving-car"
ORS_GEOCODE_URL    = "https://api.openrouteservice.org/geocode/search"
 
HEADERS = {
    "Authorization": API_KEY,
    "Content-Type": "application/json",
}
 
CSV_FILE     = "live_traffic.csv"
POLL_SECONDS = 300          # how often to re-fetch (default: 5 min)
 
# ─── HELPERS ─────────────────────────────────────────────────────────────────
 
def geocode(place_name: str) -> list[float]:
    """Return [lon, lat] for a free-text place name (Kolkata-biased)."""
    params = {
        "api_key": API_KEY,
        "text": place_name,
        "focus.point.lat": 22.5726,
        "focus.point.lon": 88.3639,
        "size": 1,
    }
    resp = requests.get(ORS_GEOCODE_URL, params=params, timeout=15)
    resp.raise_for_status()
    features = resp.json().get("features", [])
    if not features:
        raise ValueError(f"Could not geocode '{place_name}'. Try a more specific address.")
    coords = features[0]["geometry"]["coordinates"]   # [lon, lat]
    label  = features[0]["properties"].get("label", place_name)
    print(f"  → '{place_name}'  resolved to  {label}  ({coords[1]:.5f}, {coords[0]:.5f})")
    return coords
 
 
# BUG 1 FIX: parse_or_geocode was defined as an inner function inside both
# build_routes_interactively() and build_routes_from_args(), and in the latter
# it was re-defined on every loop iteration.  Inner functions that close over
# nothing loop-specific have no reason to be nested: hoisting to module level
# avoids repeated allocation on every iteration and removes the duplication.
 
def parse_or_geocode(text: str) -> tuple[list[float], str]:
    """
    Accept either 'lat,lon' coordinates or a free-text place name.
    Returns ([lon, lat], original_text_label).
    """
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 2:
        try:
            lat, lon = float(parts[0]), float(parts[1])
            return [lon, lat], text  # ORS wants [lon, lat]
        except ValueError:
            pass
    coords = geocode(text)
    return coords, text
 
 
def calculate_congestion(avg_speed_kmh: float) -> int:
    """Heuristic congestion score (0 = free-flow, 100 = gridlock)."""
    if avg_speed_kmh < 10:
        return 95
    elif avg_speed_kmh < 15:
        return 85
    elif avg_speed_kmh < 25:
        return 65
    elif avg_speed_kmh < 40:
        return 40
    elif avg_speed_kmh < 60:
        return 20
    else:
        return 10
 
 
def fetch_route_data(route: dict) -> list[dict]:
    """Call ORS for a route and return a list of row dicts (one per alternative)."""
    body = {
        "coordinates": route["coordinates"],
        "alternative_routes": {
            "target_count": 3,
            "weight_factor": 1.6,
        },
    }
    resp = requests.post(ORS_DIRECTIONS_URL, json=body, headers=HEADERS, timeout=20)
 
    # surface a readable error instead of a raw HTTPError
    if not resp.ok:
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise RuntimeError(
            f"ORS error {resp.status_code} for route '{route['name']}': {detail}"
        )
 
    data = resp.json()
    raw_routes = data.get("routes")
    if not raw_routes:
        raise RuntimeError(f"No routes returned for '{route['name']}'.")
 
    now = datetime.now()
    rows = []
    for idx, r in enumerate(raw_routes):
        summary      = r["summary"]
        distance_km  = summary["distance"] / 1000
        duration_min = summary["duration"] / 60
        avg_speed    = distance_km / (duration_min / 60) if duration_min > 0 else 0
        congestion   = calculate_congestion(avg_speed)
 
        rows.append(
            {
                "timestamp":   now.isoformat(),
                "route_name":  route["name"],
                "origin":      route["origin_label"],
                "destination": route["destination_label"],
                "route_index": idx,
                "distance_km": round(distance_km, 2),
                "duration_min": round(duration_min, 2),
                "avg_speed_kmh": round(avg_speed, 2),
                "congestion":  congestion,
                "hour":        now.hour,
                "weekday":     now.weekday(),  # 0=Monday … 6=Sunday
            }
        )
    return rows
 
 
# BUG 2 FIX: save_rows previously hardcoded the module-level CSV_FILE name
# instead of accepting the output path as a parameter.  main() mutated
# CSV_FILE via `global CSV_FILE`, which is fragile: any call to save_rows
# before main() ran would silently write to the default filename regardless
# of the --output flag.  Passing the path explicitly removes that dependency.
 
def save_rows(rows: list[dict], csv_file: str) -> None:
    df         = pd.DataFrame(rows)
    file_path  = Path(csv_file)
    write_header = not file_path.exists() or file_path.stat().st_size == 0
    df.to_csv(csv_file, mode="a", header=write_header, index=False)
 
 
# ─── INTERACTIVE ROUTE BUILDER ───────────────────────────────────────────────
 
def build_routes_interactively() -> list[dict]:
    print("\n╔══════════════════════════════════════════╗")
    print("║   Live Traffic Collector — Route Setup  ║")
    print("╚══════════════════════════════════════════╝\n")
 
    routes = []
    while True:
        print(f"Route #{len(routes) + 1}")
        origin_text = input("  Enter ORIGIN (place name or 'lat,lon'):  ").strip()
        dest_text   = input("  Enter DESTINATION (place name or 'lat,lon'): ").strip()
        route_name  = input(f"  Route name [Route_{chr(65 + len(routes))}]: ").strip()
        if not route_name:
            route_name = f"Route_{chr(65 + len(routes))}"
 
        try:
            origin_coords, origin_label = parse_or_geocode(origin_text)
            dest_coords,   dest_label   = parse_or_geocode(dest_text)
        except Exception as e:
            print(f"  ✗ {e}\n")
            continue
 
        routes.append(
            {
                "name":              route_name,
                "coordinates":       [origin_coords, dest_coords],
                "origin_label":      origin_label,
                "destination_label": dest_label,
            }
        )
        print(f"  ✓ '{route_name}' added.\n")
 
        more = input("Add another route? [y/N]: ").strip().lower()
        if more != "y":
            break
 
    return routes
 
 
def build_routes_from_args(args) -> list[dict]:
    """Build routes from --route flags like:  --route 'Park Street' 'Howrah'"""
    routes = []
    for idx, (origin_text, dest_text) in enumerate(args.route):
        route_name = f"Route_{chr(65 + idx)}"
        print(f"\nResolving {route_name}: '{origin_text}' → '{dest_text}'")
 
        origin_coords, origin_label = parse_or_geocode(origin_text)
        dest_coords,   dest_label   = parse_or_geocode(dest_text)
 
        routes.append(
            {
                "name":              route_name,
                "coordinates":       [origin_coords, dest_coords],
                "origin_label":      origin_label,
                "destination_label": dest_label,
            }
        )
    return routes
 
 
# ─── MAIN LOOP ────────────────────────────────────────────────────────────────
 
def main():
    parser = argparse.ArgumentParser(
        description="Collect live driving-route data via OpenRouteService."
    )
    parser.add_argument(
        "--route",
        nargs=2,
        metavar=("ORIGIN", "DESTINATION"),
        action="append",
        help="Origin and destination (place name or 'lat,lon'). Repeat for multiple routes.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=POLL_SECONDS,
        help=f"Poll interval in seconds (default: {POLL_SECONDS})",
    )
    parser.add_argument(
        "--output",
        default=CSV_FILE,
        help=f"Output CSV file (default: {CSV_FILE})",
    )
    args = parser.parse_args()
 
    # BUG 2 FIX (continued): instead of mutating the module-level CSV_FILE
    # global, pass the resolved output path directly to save_rows().
    output_file = args.output
 
    # Determine routes
    if args.route:
        routes = build_routes_from_args(args)
    else:
        routes = build_routes_interactively()
 
    if not routes:
        print("No routes configured. Exiting.")
        return
 
    print(f"\n▶  Polling {len(routes)} route(s) every {args.interval}s → {output_file}\n")
 
    while True:
        cycle_start = time.time()
        all_rows = []
        for route in routes:
            try:
                rows = fetch_route_data(route)
                all_rows.extend(rows)
                print(
                    f"[{datetime.now().strftime('%H:%M:%S')}] "
                    f"{route['name']} — {len(rows)} alternative(s) fetched"
                )
            except Exception as e:
                print(f"  ✗ Error on {route['name']}: {e}")
 
        if all_rows:
            save_rows(all_rows, output_file)
            print(f"  ✓ Saved {len(all_rows)} row(s) to '{output_file}'\n")
 
        # Sleep for the remainder of the interval
        elapsed = time.time() - cycle_start
        sleep_for = max(0, args.interval - elapsed)
        time.sleep(sleep_for)
 
 
if __name__ == "__main__":
    main()