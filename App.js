import React, { useEffect, useRef, useState, useCallback } from "react";
import "ol/ol.css";
import Map from "ol/Map";
import View from "ol/View";
import TileLayer from "ol/layer/Tile";
import OSM from "ol/source/OSM";
import { fromLonLat } from "ol/proj";
import Feature from "ol/Feature";
import Point from "ol/geom/Point";
import LineString from "ol/geom/LineString";
import VectorSource from "ol/source/Vector";
import VectorLayer from "ol/layer/Vector";
import { Style, Stroke, Circle as CircleStyle, Fill, Text } from "ol/style";

// ─── Config ──────────────────────────────────────────────────────────────────
const API_KEY =
  "eyJvcmciOiI1YjNjZTM1OTc4NTExMTAwMDFjZjYyNDgiLCJpZCI6IjU4YWFkYTYxM2RiNTQyMzI4NTI1YzBiZmU3OWM1MGJmIiwiaCI6Im11cm11cjY0In0=";
const FLASK_API = "http://127.0.0.1:5000";
const DEFAULT_CENTER = [88.3639, 22.5726]; // Kolkata

// ─── Map Feature Helpers ──────────────────────────────────────────────────────
function makeMarker(lonLat, color, label) {
  const f = new Feature({ geometry: new Point(fromLonLat(lonLat)) });
  f.setStyle(
    new Style({
      image: new CircleStyle({
        radius: 10,
        fill: new Fill({ color }),
        stroke: new Stroke({ color: "#fff", width: 2.5 }),
      }),
      text: new Text({
        text: label,
        offsetY: -20,
        font: "bold 13px 'DM Sans', sans-serif",
        fill: new Fill({ color: "#111" }),
        stroke: new Stroke({ color: "#fff", width: 3 }),
      }),
    })
  );
  return f;
}

function makeGpsMarker(lonLat, accuracyMeters) {
  const center = fromLonLat(lonLat);
  const accuracyFeature = new Feature({ geometry: new Point(center) });
  accuracyFeature.setStyle(
    new Style({
      image: new CircleStyle({
        radius: Math.min(Math.max(accuracyMeters / 5, 14), 60),
        fill: new Fill({ color: "rgba(59,130,246,0.12)" }),
        stroke: new Stroke({ color: "rgba(59,130,246,0.4)", width: 1.5 }),
      }),
    })
  );
  const dotFeature = new Feature({ geometry: new Point(center) });
  dotFeature.setStyle(
    new Style({
      image: new CircleStyle({
        radius: 8,
        fill: new Fill({ color: "#3b82f6" }),
        stroke: new Stroke({ color: "#fff", width: 2.5 }),
      }),
      text: new Text({
        text: "📍 You",
        offsetY: -22,
        font: "bold 12px 'DM Sans', sans-serif",
        fill: new Fill({ color: "#1d4ed8" }),
        stroke: new Stroke({ color: "#fff", width: 3 }),
      }),
    })
  );
  return [accuracyFeature, dotFeature];
}

// ─── Hooks ────────────────────────────────────────────────────────────────────
function useDebounce(fn, delay) {
  const timer = useRef(null);
  return useCallback(
    (...args) => {
      clearTimeout(timer.current);
      timer.current = setTimeout(() => fn(...args), delay);
    },
    [fn, delay]
  );
}

// ─── API Helpers ──────────────────────────────────────────────────────────────
async function geocodeSearch(text) {
  const res = await fetch(
    `https://api.openrouteservice.org/geocode/autocomplete?api_key=${API_KEY}&text=${encodeURIComponent(text)}&size=6`
  );
  if (!res.ok) throw new Error("Geocoding failed");
  const data = await res.json();
  return (data.features || []).map((f) => ({
    label: f.properties.label,
    name: f.properties.name || f.properties.label.split(",")[0],
    secondary: f.properties.label.split(",").slice(1).join(",").trim(),
    coords: f.geometry.coordinates,
  }));
}

async function fetchTrafficPrediction(sequence) {
  const res = await fetch(`${FLASK_API}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ sequence }),
  });
  if (!res.ok) throw new Error(`Traffic API error: ${res.status}`);
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data; // { congestion, traffic: { label, emoji, severity, advice } }
}

// ─── Traffic Severity Config ──────────────────────────────────────────────────
const SEVERITY_STYLES = {
  critical: { bg: "#fef2f2", border: "#fecaca", text: "#dc2626", dot: "#ef4444", badge: "HIGH" },
  moderate: { bg: "#fffbeb", border: "#fde68a", text: "#d97706", dot: "#f59e0b", badge: "MED" },
  normal:   { bg: "#f0fdf4", border: "#bbf7d0", text: "#16a34a", dot: "#22c55e", badge: "LOW" },
};

// ─── Main Component ───────────────────────────────────────────────────────────
export default function App() {
  // Map refs
  const mapRef          = useRef(null);
  const mapObj          = useRef(null);
  const routeSourceRef  = useRef(null);
  const markerSourceRef = useRef(null);
  const gpsSourceRef    = useRef(null);
  const watchIdRef      = useRef(null);

  // Location state
  const [origin,      setOrigin]      = useState(null);
  const [destination, setDestination] = useState(null);
  const [activeField, setActiveField] = useState(null);
  const [originQuery, setOriginQuery] = useState("");
  const [destQuery,   setDestQuery]   = useState("");
  const [suggestions, setSuggestions] = useState([]);
  const [fetching,    setFetching]    = useState(false);

  // Route state
  const [routeStatus, setRouteStatus] = useState(null);
  const [loading,     setLoading]     = useState(false);
  const [error,       setError]       = useState("");

  // GPS state
  const [gpsTracking, setGpsTracking] = useState(false);
  const [gpsPosition, setGpsPosition] = useState(null);
  const [gpsError,    setGpsError]    = useState("");
  const [gpsLoading,  setGpsLoading]  = useState(false);
  const [followGps,   setFollowGps]   = useState(false);

  // Traffic AI state
  const [trafficInfo,    setTrafficInfo]    = useState(null);
  const [trafficLoading, setTrafficLoading] = useState(false);
  const [trafficError,   setTrafficError]   = useState("");
  const [flaskOnline,    setFlaskOnline]    = useState(null); // null=unknown, true, false

  // UI
  const [activeTab,   setActiveTab]   = useState("route"); // "route" | "traffic"
  const [bookingStep, setBookingStep] = useState(null);    // null | "confirm" | "booked"

  const originRef = useRef(null);
  const destRef   = useRef(null);

  // ─── Map Initialisation ───────────────────────────────────────────────────
  useEffect(() => {
    const markerSource = new VectorSource();
    markerSourceRef.current = markerSource;
    const routeSource = new VectorSource();
    routeSourceRef.current = routeSource;
    const gpsSource = new VectorSource();
    gpsSourceRef.current = gpsSource;

    mapObj.current = new Map({
      target: mapRef.current,
      layers: [
        new TileLayer({ source: new OSM() }),
        new VectorLayer({
          source: routeSource,
          style: new Style({
            stroke: new Stroke({ color: "#14b8a6", width: 5, lineDash: [] }),
          }),
        }),
        new VectorLayer({ source: gpsSource }),
        new VectorLayer({ source: markerSource }),
      ],
      view: new View({ center: fromLonLat(DEFAULT_CENTER), zoom: 10 }),
    });
    return () => {
      mapObj.current?.setTarget(null);
      if (watchIdRef.current !== null)
        navigator.geolocation.clearWatch(watchIdRef.current);
    };
  }, []);

  // ─── Marker updates ───────────────────────────────────────────────────────
  useEffect(() => {
    if (!markerSourceRef.current) return;
    markerSourceRef.current.clear();
    if (origin) {
      markerSourceRef.current.addFeature(
        makeMarker(origin.coords, "#14b8a6", "● Pickup")
      );
      if (!destination)
        mapObj.current
          .getView()
          .animate({ center: fromLonLat(origin.coords), zoom: 14, duration: 700 });
    }
    if (destination)
      markerSourceRef.current.addFeature(
        makeMarker(destination.coords, "#f43f5e", "▲ Drop")
      );
  }, [origin, destination]);

  // ─── GPS marker updates ───────────────────────────────────────────────────
  useEffect(() => {
    if (!gpsSourceRef.current) return;
    gpsSourceRef.current.clear();
    if (gpsPosition) {
      makeGpsMarker(gpsPosition.coords, gpsPosition.accuracy).forEach((f) =>
        gpsSourceRef.current.addFeature(f)
      );
      if (followGps) {
        mapObj.current?.getView().animate({
          center: fromLonLat(gpsPosition.coords),
          zoom: Math.max(mapObj.current.getView().getZoom(), 15),
          duration: 500,
        });
      }
    }
  }, [gpsPosition, followGps]);

  // ─── GPS controls ─────────────────────────────────────────────────────────
  const stopGps = useCallback(() => {
    if (watchIdRef.current !== null) {
      navigator.geolocation.clearWatch(watchIdRef.current);
      watchIdRef.current = null;
    }
    setGpsTracking(false);
    setFollowGps(false);
    gpsSourceRef.current?.clear();
    setGpsPosition(null);
  }, []);

  const startGps = useCallback(() => {
    if (!navigator.geolocation) {
      setGpsError("Geolocation not supported by your browser.");
      return;
    }
    setGpsLoading(true);
    setGpsError("");
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        const coords = [pos.coords.longitude, pos.coords.latitude];
        setGpsPosition({ coords, accuracy: pos.coords.accuracy });
        setGpsLoading(false);
        setGpsTracking(true);
        setFollowGps(true);
        mapObj.current
          ?.getView()
          .animate({ center: fromLonLat(coords), zoom: 16, duration: 800 });
      },
      (err) => {
        setGpsLoading(false);
        setGpsError(
          err.code === 1
            ? "Permission denied. Please allow location access."
            : err.code === 2
            ? "Position unavailable. Try again outdoors."
            : "Location request timed out."
        );
      },
      { enableHighAccuracy: true, timeout: 10000 }
    );
    watchIdRef.current = navigator.geolocation.watchPosition(
      (pos) => {
        const coords = [pos.coords.longitude, pos.coords.latitude];
        setGpsPosition({ coords, accuracy: pos.coords.accuracy });
        setGpsLoading(false);
        setGpsTracking(true);
      },
      (err) => console.warn("GPS watch error:", err.message),
      { enableHighAccuracy: true, maximumAge: 5000, timeout: 15000 }
    );
  }, []);

  const gpsPositionRef = useRef(null);
  gpsPositionRef.current = gpsPosition;

  const applyGpsAsOrigin = useCallback((pos) => {
    if (!pos) return;
    setOrigin({ coords: pos.coords, name: "My Location", label: "My Location", secondary: "GPS" });
    setOriginQuery("My Location");
    setRouteStatus(null);
    routeSourceRef.current?.clear();
  }, []);

  const pendingGpsOriginRef = useRef(false);
  useEffect(() => {
    if (pendingGpsOriginRef.current && gpsPosition && !origin) {
      pendingGpsOriginRef.current = false;
      applyGpsAsOrigin(gpsPosition);
    }
  }, [gpsPosition, origin, applyGpsAsOrigin]);

  const handleLocationButtonClick = useCallback(() => {
    const pos = gpsPositionRef.current;
    if (pos) {
      applyGpsAsOrigin(pos);
    } else {
      pendingGpsOriginRef.current = true;
      startGps();
    }
  }, [startGps, applyGpsAsOrigin]);

  // ─── Search / Suggestions ────────────────────────────────────────────────
  const doSearch = useCallback(async (text) => {
    if (text.trim().length < 2) { setSuggestions([]); return; }
    setFetching(true);
    try { setSuggestions(await geocodeSearch(text)); }
    catch (e) { console.error(e); setSuggestions([]); }
    finally { setFetching(false); }
  }, []);

  const debouncedSearch = useDebounce(doSearch, 320);

  const handleOriginChange = (e) => {
    const v = e.target.value;
    setOriginQuery(v); setOrigin(null); setRouteStatus(null); setError("");
    debouncedSearch(v);
  };

  const handleDestChange = (e) => {
    const v = e.target.value;
    setDestQuery(v); setDestination(null); setRouteStatus(null); setError("");
    debouncedSearch(v);
  };

  const pickSuggestion = (s) => {
    if (activeField === "origin") {
      setOrigin(s); setOriginQuery(s.name);
      setSuggestions([]); setActiveField("destination");
      setTimeout(() => destRef.current?.focus(), 60);
    } else {
      setDestination(s); setDestQuery(s.name);
      setSuggestions([]); setActiveField(null);
    }
  };

  // ─── Intelligent AI Route Optimization ───────────────────────────────────
  // Replaces the old ORS-direct fetchRoute. Now calls the Flask
  // /optimize-route endpoint, draws the best route returned by the
  // LSTM model, and populates trafficInfo so the Traffic AI tab is
  // automatically filled after every route search.
  const fetchRoute = useCallback(async () => {

    if (!origin || !destination) return;

    setLoading(true);
    setError("");
    setBookingStep(null);
    setTrafficInfo(null);
    routeSourceRef.current.clear();

    try {

      // ── Call Flask AI optimization API ──────────────────────────────
      const response = await fetch(`${FLASK_API}/optimize-route`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          origin:      origin.coords,       // [lon, lat]
          destination: destination.coords,  // [lon, lat]
          hour:        new Date().getHours(),
          weekday:     new Date().getDay(),
        }),
      });

      if (!response.ok) {
        const txt = await response.text();
        throw new Error(`Optimization failed: ${txt}`);
      }

      const data = await response.json();

      if (!data.best_route) {
        throw new Error(data.error || "No optimized route returned");
      }

      // ── Extract best route ───────────────────────────────────────────
      const bestRoute = data.best_route;

      // ── Draw route on map ────────────────────────────────────────────
      // The Flask backend returns the raw ORS route object under
      // bestRoute.geometry. ORS encodes waypoints in
      // route.geometry.coordinates (GeoJSON LineString).
      const orsGeometry =
        bestRoute.geometry?.geometry?.coordinates ??   // nested GeoJSON
        bestRoute.geometry?.coordinates ??             // flat GeoJSON
        [];

      if (!orsGeometry.length) {
        throw new Error("Route geometry missing in API response.");
      }

      const olCoords = orsGeometry.map((c) => fromLonLat(c));
      const feat = new Feature({ geometry: new LineString(olCoords) });
      routeSourceRef.current.addFeature(feat);

      mapObj.current
        .getView()
        .fit(feat.getGeometry().getExtent(), {
          padding: [60, 60, 60, 60],
          duration: 900,
        });

      setFollowGps(false);

      // ── Route summary ────────────────────────────────────────────────
      const km   = Number(bestRoute.distance_km  || 0).toFixed(1);
      const mins = Math.round(bestRoute.duration_min || 0);
      const hrs  = Math.floor(mins / 60);

      setRouteStatus({
        km,
        time:    hrs > 0 ? `${hrs}h ${mins % 60}m` : `${mins} min`,
        rawMins: mins,
      });

      // ── Populate Traffic AI info ─────────────────────────────────────
      // traffic_level comes from the Flask classify_congestion() helper
      // and matches the SEVERITY_STYLES keys used by the Traffic AI tab.
      const trafficLevel = bestRoute.traffic_level || "LOW";

const congestion =
  bestRoute.congestion_score ??
  bestRoute.predicted_congestion ??
  0;
      setTrafficInfo({
        congestion,
        traffic: {
          label:    trafficLevel,
          emoji:
            trafficLevel === "SEVERE" ? "🔴"
            : trafficLevel === "HIGH" ? "🟠"
            : trafficLevel === "MEDIUM" ? "🟡"
            : "🟢",
          severity:
            trafficLevel === "SEVERE" || trafficLevel === "HIGH"
              ? "critical"
              : trafficLevel === "MEDIUM"
              ? "moderate"
              : "normal",
          advice: data.rerouted
            ? "Heavy traffic detected ahead. The AI selected a better alternate route automatically."
            : "Traffic conditions are optimal for this route.",
        },
      });

      if (data.rerouted) {
        console.info("UrbanFlowAI: AI rerouting activated.");
      }

    } catch (err) {
      console.error(err);
      setError(err.message || "Unable to optimize route");
    } finally {
      setLoading(false);
    }

  }, [origin, destination]);

  // Auto-route when both are set
  useEffect(() => {
    if (origin && destination) fetchRoute();
  }, [origin, destination, fetchRoute]);

  // ─── Traffic AI Prediction (manual tab) ──────────────────────────────────
  const checkFlaskHealth = useCallback(async () => {
    try {
      const res = await fetch(`${FLASK_API}/health`, { signal: AbortSignal.timeout(3000) });
      setFlaskOnline(res.ok);
    } catch {
      setFlaskOnline(false);
    }
  }, []);

  useEffect(() => { checkFlaskHealth(); }, [checkFlaskHealth]);

  // Build a plausible dynamic sequence from route distance/time.
  // In production: replace with real historical readings from your backend.
  const buildDynamicSequence = useCallback((rs) => {
    if (!rs) return [60, 65, 70, 75, 80];
    const base = Math.min(90, Math.max(10, rs.rawMins / 2));
    return Array.from({ length: 5 }, (_, i) =>
      Math.round(base + i * 3 + (Math.random() * 4 - 2))
    );
  }, []);

  const getTrafficPrediction = useCallback(async () => {
    setTrafficLoading(true);
    setTrafficError("");
    setTrafficInfo(null);
    try {
      const sequence = buildDynamicSequence(routeStatus);
      const data = await fetchTrafficPrediction(sequence);
      setTrafficInfo(data);
    } catch (err) {
      console.error(err);
      setTrafficError(err.message || "Could not reach traffic prediction server.");
    } finally {
      setTrafficLoading(false);
    }
  }, [routeStatus, buildDynamicSequence]);

  // ─── UI Actions ───────────────────────────────────────────────────────────
  const clearAll = () => {
    setOrigin(null); setDestination(null);
    setOriginQuery(""); setDestQuery("");
    setSuggestions([]); setRouteStatus(null); setError(""); setActiveField(null);
    setTrafficInfo(null); setTrafficError(""); setBookingStep(null);
    routeSourceRef.current?.clear(); markerSourceRef.current?.clear();
    mapObj.current
      ?.getView()
      .animate({ center: fromLonLat(DEFAULT_CENTER), zoom: 10, duration: 700 });
  };

  const swap = () => {
    setOrigin(destination); setDestination(origin);
    setOriginQuery(destQuery); setDestQuery(originQuery);
    setRouteStatus(null); setTrafficInfo(null); setBookingStep(null);
    routeSourceRef.current?.clear();
  };

  const handleBook   = () => setBookingStep("confirm");
  const confirmBook  = () => setBookingStep("booked");

  const isSearching = activeField !== null && suggestions.length > 0;
  const canSearch   = origin && destination;

  // ─── Render ───────────────────────────────────────────────────────────────
  return (
    <>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;0,9..40,800;1,9..40,400&family=Space+Mono:wght@400;700&display=swap');

        *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'DM Sans', sans-serif; }
        input { font-family: inherit; }
        input::placeholder { color: #94a3b8; font-weight: 400; }
        input:focus { outline: none; }

        :root {
          --teal: #14b8a6;
          --teal-dark: #0d9488;
          --rose: #f43f5e;
          --ink: #0f172a;
          --muted: #64748b;
          --border: #e2e8f0;
          --surface: #f8fafc;
          --sidebar: #ffffff;
        }

        .field-row {
          display: flex; align-items: center; gap: 12px;
          padding: 0 14px; border-radius: 12px;
          border: 1.5px solid var(--border); background: var(--surface);
          transition: all 0.18s ease; cursor: text;
        }
        .field-row:focus-within {
          border-color: var(--teal); background: #fff;
          box-shadow: 0 0 0 3px rgba(20,184,166,0.15);
        }
        .sugg-item { transition: background 0.12s; }
        .sugg-item:hover { background: #f0fdfa !important; }

        .primary-btn {
          width: 100%; padding: 14px;
          background: linear-gradient(135deg, var(--teal) 0%, #0891b2 100%);
          color: #fff; border: none; border-radius: 12px;
          font-size: 14px; font-weight: 700; font-family: 'DM Sans', sans-serif;
          cursor: pointer; letter-spacing: 0.02em;
          box-shadow: 0 4px 16px rgba(20,184,166,0.35);
          transition: all 0.2s ease;
        }
        .primary-btn:hover:not(:disabled) { transform: translateY(-1px); box-shadow: 0 8px 24px rgba(20,184,166,0.45); }
        .primary-btn:active:not(:disabled) { transform: translateY(0); }
        .primary-btn:disabled { background: #e2e8f0; color: #94a3b8; box-shadow: none; cursor: not-allowed; }

        .ghost-btn {
          background: none; border: 1.5px solid var(--border);
          border-radius: 10px; padding: 10px 16px;
          font-size: 13px; font-weight: 600; font-family: 'DM Sans', sans-serif;
          color: var(--muted); cursor: pointer; transition: all 0.18s;
        }
        .ghost-btn:hover { border-color: var(--teal); color: var(--teal); background: rgba(20,184,166,0.05); }

        .clear-x {
          background: #e2e8f0; border: none; border-radius: 50%;
          width: 20px; height: 20px; cursor: pointer;
          display: flex; align-items: center; justify-content: center;
          font-size: 10px; color: #64748b; flex-shrink: 0; transition: background 0.15s;
        }
        .clear-x:hover { background: #cbd5e1; }

        .swap-btn {
          width: 28px; height: 28px; border-radius: 50%;
          background: #fff; border: 1.5px solid var(--border);
          cursor: pointer; display: flex; align-items: center;
          justify-content: center; font-size: 14px; color: #64748b;
          box-shadow: 0 2px 8px rgba(0,0,0,0.08); transition: all 0.18s; flex-shrink: 0;
        }
        .swap-btn:hover { background: #f0fdfa; border-color: var(--teal); color: var(--teal); }

        .loc-btn {
          width: 30px; height: 30px; border-radius: 50%;
          border: 1.5px solid var(--border); background: var(--surface);
          cursor: pointer; display: flex; align-items: center;
          justify-content: center; font-size: 14px; flex-shrink: 0; transition: all 0.18s;
        }
        .loc-btn:hover { border-color: #3b82f6; background: rgba(59,130,246,0.08); }
        .loc-btn.active { border-color: #3b82f6; background: rgba(59,130,246,0.1); }

        .tab-btn {
          flex: 1; padding: 9px; background: none; border: none;
          font-family: 'DM Sans', sans-serif; font-size: 13px; font-weight: 600;
          color: var(--muted); cursor: pointer; border-radius: 8px; transition: all 0.18s;
        }
        .tab-btn.active { background: #fff; color: var(--ink); box-shadow: 0 1px 4px rgba(0,0,0,0.08); }

        .vehicle-card {
          flex: 1; text-align: center; padding: 8px 4px;
          border-radius: 10px; background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.1);
          cursor: pointer; transition: all 0.18s;
        }
        .vehicle-card:hover { background: rgba(255,255,255,0.12); border-color: var(--teal); }
        .vehicle-card.selected { background: rgba(20,184,166,0.2); border-color: var(--teal); }

        .map-gps-fab {
          position: absolute; bottom: 32px; right: 16px; z-index: 20;
          width: 44px; height: 44px; border-radius: 50%;
          background: #fff; border: none;
          box-shadow: 0 4px 16px rgba(0,0,0,0.18);
          cursor: pointer; display: flex; align-items: center;
          justify-content: center; font-size: 20px; transition: all 0.18s;
        }
        .map-gps-fab:hover { transform: scale(1.08); box-shadow: 0 6px 22px rgba(0,0,0,0.22); }
        .map-gps-fab.active { background: #3b82f6; }

        .status-dot {
          width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
          animation: pulse-dot 2s infinite;
        }
        @keyframes pulse-dot { 0%,100% { opacity: 1; } 50% { opacity: 0.5; } }
        @keyframes spin { to { transform: rotate(360deg); } }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes slideUp { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
        @keyframes shimmer { 0% { background-position: -400px 0; } 100% { background-position: 400px 0; } }
        @keyframes scaleIn { from { opacity: 0; transform: scale(0.95); } to { opacity: 1; transform: scale(1); } }

        .spinner {
          width: 16px; height: 16px;
          border: 2px solid #e2e8f0; border-top-color: var(--teal);
          border-radius: 50%; animation: spin 0.7s linear infinite; flex-shrink: 0;
        }
        .spinner.rose { border-top-color: var(--rose); }
        .spinner.white { border-color: rgba(255,255,255,0.3); border-top-color: #fff; }
        .spinner.blue { border-top-color: #3b82f6; }

        .booking-overlay {
          position: absolute; inset: 0; z-index: 50;
          background: rgba(15,23,42,0.7); backdrop-filter: blur(6px);
          display: flex; align-items: center; justify-content: center;
          animation: fadeIn 0.2s ease;
        }
        .booking-card {
          background: #fff; border-radius: 20px; padding: 28px;
          width: 340px; box-shadow: 0 24px 60px rgba(0,0,0,0.3);
          animation: scaleIn 0.2s ease;
        }
      `}</style>

      <div style={{ display: "flex", height: "100vh", width: "100%", overflow: "hidden" }}>

        {/* ─── SIDEBAR ─────────────────────────────────────────────────── */}
        <div style={{
          width: 360, flexShrink: 0, height: "100%", background: "var(--sidebar)",
          display: "flex", flexDirection: "column",
          boxShadow: "4px 0 24px rgba(0,0,0,0.08)", zIndex: 10, overflowY: "auto",
        }}>

          {/* Header */}
          <div style={{
            padding: "24px 22px 18px",
            background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)",
            position: "relative", overflow: "hidden",
          }}>
            <div style={{ position: "absolute", top: -24, right: -24, width: 100, height: 100, borderRadius: "50%", background: "rgba(20,184,166,0.12)" }} />
            <div style={{ position: "absolute", bottom: -16, right: 40, width: 56, height: 56, borderRadius: "50%", background: "rgba(20,184,166,0.08)" }} />
            <div style={{ display: "flex", alignItems: "center", gap: 10, position: "relative" }}>
              <div style={{ width: 38, height: 38, borderRadius: 10, background: "rgba(20,184,166,0.2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 18, border: "1px solid rgba(20,184,166,0.3)" }}>🚀</div>
              <div>
                <div style={{ fontSize: 18, fontWeight: 800, color: "#fff", letterSpacing: "-0.3px", fontFamily: "'Space Mono', monospace" }}>UrbanFlow AI</div>
                <div style={{ fontSize: 11, color: "rgba(255,255,255,0.5)", marginTop: 1 }}>Smart city navigator</div>
              </div>
              {/* Flask online indicator */}
              <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 5, padding: "4px 8px", borderRadius: 20, background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.1)" }}>
                <div className="status-dot" style={{ background: flaskOnline === null ? "#94a3b8" : flaskOnline ? "#22c55e" : "#ef4444" }} />
                <span style={{ fontSize: 10, color: "rgba(255,255,255,0.6)", fontFamily: "'Space Mono', monospace" }}>
                  {flaskOnline === null ? "AI..." : flaskOnline ? "AI ON" : "AI OFF"}
                </span>
              </div>
            </div>
          </div>

          {/* Tabs */}
          <div style={{ padding: "12px 16px 0" }}>
            <div style={{ display: "flex", background: "#f1f5f9", borderRadius: 10, padding: 3, gap: 2 }}>
              {[["route", "🗺 Route"], ["traffic", "🤖 AI Traffic"]].map(([tab, label]) => (
                <button key={tab} className={`tab-btn${activeTab === tab ? " active" : ""}`} onClick={() => setActiveTab(tab)}>{label}</button>
              ))}
            </div>
          </div>

          {/* Form */}
          <div style={{ padding: "16px 16px 0", flex: 1, display: "flex", flexDirection: "column", gap: 0 }}>

            {activeTab === "route" && (
              <>
                {/* GPS error */}
                {gpsError && (
                  <div style={{ marginBottom: 12, padding: "9px 12px", borderRadius: 10, background: "#fef2f2", border: "1px solid #fecaca", fontSize: 12, color: "#dc2626", fontWeight: 500, display: "flex", gap: 6, alignItems: "flex-start", animation: "fadeIn 0.2s ease" }}>
                    <span>⚠️</span>{gpsError}
                  </div>
                )}

                {/* Origin */}
                <div style={{ marginBottom: 8, position: "relative" }}>
                  <div className="field-row">
                    <div style={{ width: 11, height: 11, borderRadius: "50%", background: origin ? "var(--teal)" : "#cbd5e1", border: "2.5px solid " + (origin ? "var(--teal)" : "#cbd5e1"), flexShrink: 0, boxShadow: origin ? "0 0 0 3px rgba(20,184,166,0.2)" : "none", transition: "all 0.2s" }} />
                    <input
                      ref={originRef}
                      type="text"
                      value={originQuery}
                      onChange={handleOriginChange}
                      onFocus={() => { setActiveField("origin"); if (originQuery.length >= 2) debouncedSearch(originQuery); }}
                      onBlur={() => setTimeout(() => { setActiveField(prev => { if (prev === "origin") { setSuggestions([]); return null; } return prev; }); }, 180)}
                      placeholder="Pickup location"
                      style={{ flex: 1, border: "none", background: "transparent", fontSize: 14, fontWeight: origin ? 600 : 400, color: "var(--ink)", padding: "15px 0" }}
                    />
                    {fetching && activeField === "origin" && <div className="spinner" />}
                    {origin
                      ? <button className="clear-x" onClick={() => { setOrigin(null); setOriginQuery(""); setRouteStatus(null); routeSourceRef.current?.clear(); originRef.current?.focus(); setActiveField("origin"); }}>✕</button>
                      : <button className={`loc-btn${gpsTracking && gpsPosition ? " active" : ""}`} onClick={handleLocationButtonClick} title="Use my location">
                          {gpsLoading ? <div className="spinner blue" style={{ width: 12, height: 12 }} /> : "📍"}
                        </button>
                    }
                  </div>
                </div>

                {/* Connector + swap */}
                <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "0 8px", marginBottom: 8 }}>
                  <div style={{ display: "flex", flexDirection: "column", gap: 3, paddingLeft: 2 }}>
                    {[0, 1, 2].map(i => <div key={i} style={{ width: 2, height: 4, borderRadius: 1, background: "#cbd5e1" }} />)}
                  </div>
                  <div style={{ flex: 1 }} />
                  <button className="swap-btn" onClick={swap} title="Swap">⇅</button>
                </div>

                {/* Destination */}
                <div style={{ marginBottom: 14, position: "relative" }}>
                  <div className="field-row">
                    <div style={{ width: 10, height: 10, background: destination ? "var(--rose)" : "transparent", border: "2.5px solid " + (destination ? "var(--rose)" : "#cbd5e1"), transform: "rotate(45deg)", flexShrink: 0, boxShadow: destination ? "0 0 0 3px rgba(244,63,94,0.2)" : "none", transition: "all 0.2s" }} />
                    <input
                      ref={destRef}
                      type="text"
                      value={destQuery}
                      onChange={handleDestChange}
                      onFocus={() => { setActiveField("destination"); if (destQuery.length >= 2) debouncedSearch(destQuery); }}
                      onBlur={() => setTimeout(() => { setActiveField(prev => { if (prev === "destination") { setSuggestions([]); return null; } return prev; }); }, 180)}
                      placeholder="Dropoff location"
                      style={{ flex: 1, border: "none", background: "transparent", fontSize: 14, fontWeight: destination ? 600 : 400, color: "var(--ink)", padding: "15px 0" }}
                    />
                    {fetching && activeField === "destination" && <div className="spinner rose" />}
                    {destination && (
                      <button className="clear-x" onClick={() => { setDestination(null); setDestQuery(""); setRouteStatus(null); routeSourceRef.current?.clear(); destRef.current?.focus(); setActiveField("destination"); }}>✕</button>
                    )}
                  </div>
                </div>

                {/* Suggestions */}
                {isSearching && (
                  <div style={{ background: "#fff", borderRadius: 12, border: "1px solid var(--border)", overflow: "hidden", boxShadow: "0 8px 32px rgba(0,0,0,0.12)", marginBottom: 12, animation: "fadeIn 0.18s ease" }}>
                    {suggestions.map((s, i) => (
                      <div key={i} className="sugg-item" onMouseDown={() => pickSuggestion(s)}
                        style={{ display: "flex", alignItems: "center", gap: 10, padding: "10px 12px", borderBottom: i < suggestions.length - 1 ? "1px solid #f1f5f9" : "none", cursor: "pointer", background: "#fff" }}>
                        <div style={{ width: 32, height: 32, borderRadius: 8, background: activeField === "origin" ? "rgba(20,184,166,0.1)" : "rgba(244,63,94,0.1)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14, flexShrink: 0 }}>
                          {activeField === "origin" ? "🟢" : "🔴"}
                        </div>
                        <div style={{ minWidth: 0 }}>
                          <div style={{ fontSize: 13, fontWeight: 600, color: "var(--ink)", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{s.name}</div>
                          {s.secondary && <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 1, whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{s.secondary}</div>}
                        </div>
                      </div>
                    ))}
                  </div>
                )}

                {/* Route card */}
                {!isSearching && (routeStatus || loading || error) && (
                  <div style={{ marginBottom: 12, borderRadius: 14, overflow: "hidden", animation: "slideUp 0.25s ease", border: "1px solid var(--border)" }}>
                    {loading && (
                      <div style={{ padding: "14px", background: "linear-gradient(90deg,#f1f5f9 25%,#e2e8f0 50%,#f1f5f9 75%)", backgroundSize: "800px 100%", animation: "shimmer 1.5s infinite", display: "flex", alignItems: "center", gap: 10 }}>
                        <div className="spinner" /><span style={{ fontSize: 13, color: "var(--muted)", fontWeight: 500 }}>AI optimizing route…</span>
                      </div>
                    )}
                    {error && !loading && (
                      <div style={{ padding: "12px 14px", background: "#fef2f2", display: "flex", gap: 8, alignItems: "center" }}>
                        <span>❌</span><span style={{ fontSize: 13, color: "#dc2626", fontWeight: 500 }}>{error}</span>
                      </div>
                    )}
                    {routeStatus && !loading && !error && (
                      <div style={{ padding: "16px", background: "linear-gradient(135deg, #0f172a 0%, #1e293b 100%)" }}>
                        <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: 12 }}>
                          <div>
                            <div style={{ fontSize: 28, fontWeight: 800, color: "#fff", letterSpacing: "-0.8px", fontFamily: "'Space Mono', monospace" }}>{routeStatus.time}</div>
                            <div style={{ fontSize: 12, color: "#64748b", marginTop: 3 }}>{routeStatus.km} km • AI Optimized</div>
                          </div>
                          <div style={{ background: "rgba(20,184,166,0.2)", borderRadius: 10, padding: "8px 10px", fontSize: 18, border: "1px solid rgba(20,184,166,0.3)" }}>🛣️</div>
                        </div>
                        {/* Traffic badge from AI prediction */}
                        {trafficInfo && (
                          <div style={{ marginBottom: 10, display: "flex", alignItems: "center", gap: 6, padding: "6px 10px", borderRadius: 8, background: "rgba(255,255,255,0.07)", border: "1px solid rgba(255,255,255,0.1)" }}>
                            <span style={{ fontSize: 14 }}>{trafficInfo.traffic?.emoji}</span>
                            <span style={{ fontSize: 12, color: "rgba(255,255,255,0.8)", fontWeight: 600 }}>{trafficInfo.traffic?.label} traffic</span>
                            <span style={{ marginLeft: "auto", fontSize: 11, color: "rgba(255,255,255,0.4)", fontFamily: "'Space Mono', monospace" }}>{trafficInfo.congestion?.toFixed(0)}%</span>
                          </div>
                        )}
                        {/* Vehicle options */}
                        <div style={{ display: "flex", gap: 6 }}>
                          {[["🚗", "Car"], ["🏍️", "Bike"], ["🚐", "Auto"]].map(([icon, label]) => (
                            <div key={label} className="vehicle-card">
                              <div style={{ fontSize: 16, marginBottom: 2 }}>{icon}</div>
                              <div style={{ fontSize: 10, color: "#94a3b8", fontWeight: 600 }}>{label}</div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}

                {/* CTA */}
                <button className="primary-btn"
                  onClick={routeStatus && !loading ? handleBook : (canSearch && !loading ? fetchRoute : undefined)}
                  disabled={!canSearch || loading}>
                  {loading
                    ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}><span className="spinner white" />AI optimizing…</span>
                    : routeStatus ? "Book Now →" : "Search Route"
                  }
                </button>

                {(origin || destination) && !isSearching && (
                  <button onClick={clearAll} className="ghost-btn" style={{ width: "100%", marginTop: 8 }}>Clear trip</button>
                )}
              </>
            )}

            {/* ─── TRAFFIC AI TAB ─────────────────────────────────────── */}
            {activeTab === "traffic" && (
              <div style={{ animation: "fadeIn 0.2s ease" }}>
                {/* Flask status banner */}
                <div style={{
                  padding: "10px 12px", borderRadius: 10, marginBottom: 14,
                  background: flaskOnline ? "#f0fdf4" : "#fef2f2",
                  border: `1px solid ${flaskOnline ? "#bbf7d0" : "#fecaca"}`,
                  display: "flex", alignItems: "center", gap: 8,
                }}>
                  <div className="status-dot" style={{ background: flaskOnline === null ? "#94a3b8" : flaskOnline ? "#22c55e" : "#ef4444" }} />
                  <span style={{ fontSize: 12, fontWeight: 600, color: flaskOnline ? "#16a34a" : "#dc2626" }}>
                    {flaskOnline === null ? "Checking Flask API…" : flaskOnline ? `Flask API online — ${FLASK_API}` : `Flask API offline — start your server`}
                  </span>
                  <button onClick={checkFlaskHealth} style={{ marginLeft: "auto", background: "none", border: "none", cursor: "pointer", fontSize: 14 }}>🔄</button>
                </div>

                {/* Sequence info */}
                <div style={{ padding: "12px", borderRadius: 10, background: "var(--surface)", border: "1px solid var(--border)", marginBottom: 14 }}>
                  <div style={{ fontSize: 11, fontWeight: 700, color: "var(--muted)", textTransform: "uppercase", letterSpacing: "0.06em", marginBottom: 8 }}>Input Sequence</div>
                  <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                    {buildDynamicSequence(routeStatus).map((v, i) => (
                      <div key={i} style={{ padding: "4px 10px", borderRadius: 6, background: "#fff", border: "1px solid var(--border)", fontSize: 12, fontWeight: 700, color: "var(--ink)", fontFamily: "'Space Mono', monospace" }}>{v}</div>
                    ))}
                  </div>
                  <div style={{ fontSize: 11, color: "#94a3b8", marginTop: 6 }}>
                    {routeStatus ? `Derived from ${routeStatus.km} km / ${routeStatus.time} route` : "Using default sequence — search a route first for dynamic values"}
                  </div>
                </div>

                {/* Predict button */}
                <button className="primary-btn" onClick={getTrafficPrediction} disabled={trafficLoading || !flaskOnline}>
                  {trafficLoading
                    ? <span style={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 10 }}><span className="spinner white" />Predicting…</span>
                    : "🤖 Get AI Prediction"
                  }
                </button>

                {/* Error */}
                {trafficError && !trafficLoading && (
                  <div style={{ marginTop: 12, padding: "10px 12px", borderRadius: 10, background: "#fef2f2", border: "1px solid #fecaca", fontSize: 12, color: "#dc2626", fontWeight: 500, animation: "fadeIn 0.2s ease" }}>
                    ⚠️ {trafficError}
                  </div>
                )}

                {/* Result */}
                {trafficInfo && !trafficLoading && (() => {
                  const isObj    = typeof trafficInfo.traffic === "object" && trafficInfo.traffic !== null;
                  const severity = isObj ? trafficInfo.traffic.severity : (trafficInfo.congestion > 70 ? "critical" : trafficInfo.congestion > 40 ? "moderate" : "normal");
                  const label    = isObj ? trafficInfo.traffic.label  : (trafficInfo.traffic || "Unknown");
                  const emoji    = isObj ? trafficInfo.traffic.emoji  : "";
                  const advice   = isObj ? trafficInfo.traffic.advice : "";
                  const styles   = SEVERITY_STYLES[severity] || SEVERITY_STYLES.normal;

                  return (
                    <div style={{ marginTop: 14, borderRadius: 14, overflow: "hidden", animation: "slideUp 0.25s ease", border: `1.5px solid ${styles.border}` }}>
                      <div style={{ padding: "16px", background: styles.bg }}>
                        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10 }}>
                          <div style={{ fontSize: 28 }}>{emoji}</div>
                          <div>
                            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                              <span style={{ fontSize: 18, fontWeight: 800, color: styles.text, fontFamily: "'Space Mono', monospace" }}>{label}</span>
                              <span style={{ fontSize: 10, fontWeight: 700, padding: "2px 6px", borderRadius: 4, background: styles.dot, color: "#fff" }}>{styles.badge}</span>
                            </div>
                            <div style={{ fontSize: 12, color: "var(--muted)", marginTop: 2 }}>Congestion index: <strong style={{ color: "var(--ink)" }}>{trafficInfo.congestion?.toFixed(1)}</strong></div>
                          </div>
                        </div>
                        {advice && <div style={{ fontSize: 12, color: "var(--muted)", lineHeight: 1.5, padding: "8px 10px", borderRadius: 8, background: "rgba(255,255,255,0.6)", border: `1px solid ${styles.border}` }}>{advice}</div>}
                      </div>
                      {/* Congestion bar */}
                      <div style={{ padding: "10px 14px", background: "#fff" }}>
                        <div style={{ fontSize: 10, color: "var(--muted)", fontWeight: 600, marginBottom: 4, textTransform: "uppercase", letterSpacing: "0.06em" }}>Congestion Level</div>
                        <div style={{ height: 8, borderRadius: 4, background: "#e2e8f0", overflow: "hidden" }}>
                          <div style={{ height: "100%", width: `${Math.min(100, trafficInfo.congestion || 0)}%`, background: `linear-gradient(90deg, #22c55e, #f59e0b, #ef4444)`, borderRadius: 4, transition: "width 0.8s ease" }} />
                        </div>
                        <div style={{ display: "flex", justifyContent: "space-between", marginTop: 3 }}>
                          <span style={{ fontSize: 9, color: "#94a3b8" }}>0</span>
                          <span style={{ fontSize: 9, color: "#94a3b8" }}>50</span>
                          <span style={{ fontSize: 9, color: "#94a3b8" }}>100</span>
                        </div>
                      </div>
                    </div>
                  );
                })()}

                {!trafficInfo && !trafficLoading && !trafficError && (
                  <div style={{ marginTop: 20, textAlign: "center", color: "#94a3b8", fontSize: 13 }}>
                    <div style={{ fontSize: 32, marginBottom: 8 }}>🤖</div>
                    Press the button above to get an AI traffic prediction from your Flask model.
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Footer */}
          <div style={{ padding: "14px 16px", borderTop: "1px solid #f1f5f9", display: "flex", alignItems: "center", gap: 8, flexShrink: 0 }}>
            <div style={{ width: 30, height: 30, borderRadius: 8, background: "linear-gradient(135deg, var(--teal), #0891b2)", display: "flex", alignItems: "center", justifyContent: "center", fontSize: 14 }}>🗺️</div>
            <div>
              <div style={{ fontSize: 11, fontWeight: 700, color: "var(--ink)", fontFamily: "'Space Mono', monospace" }}>UrbanFlow AI v2</div>
              <div style={{ fontSize: 10, color: "#94a3b8" }}>OpenRouteService + Flask LSTM</div>
            </div>
          </div>
        </div>

        {/* ─── MAP ─────────────────────────────────────────────────────── */}
        <div style={{ flex: 1, height: "100%", position: "relative" }}>
          <div ref={mapRef} style={{ width: "100%", height: "100%" }} />

          {/* GPS FAB */}
          <button
            className={`map-gps-fab${gpsTracking ? " active" : ""}`}
            onClick={() => {
              if (!gpsTracking) { startGps(); return; }
              if (gpsPosition) {
                setFollowGps(true);
                mapObj.current?.getView().animate({ center: fromLonLat(gpsPosition.coords), zoom: 16, duration: 600 });
              }
            }}
            title={gpsTracking ? "Re-center on my location" : "Enable GPS"}
          >
            {gpsTracking ? "🔵" : "📡"}
          </button>
        </div>

        {/* ─── BOOKING OVERLAY ──────────────────────────────────────────── */}
        {bookingStep === "confirm" && (
          <div className="booking-overlay" onClick={() => setBookingStep(null)}>
            <div className="booking-card" onClick={e => e.stopPropagation()}>
              <div style={{ fontSize: 24, marginBottom: 12 }}>🚗</div>
              <div style={{ fontSize: 18, fontWeight: 800, color: "var(--ink)", marginBottom: 4 }}>Confirm Booking</div>
              <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 18 }}>
                {origin?.name} → {destination?.name}
                <br />
                <strong style={{ color: "var(--ink)" }}>{routeStatus?.km} km · {routeStatus?.time}</strong>
              </div>
              {trafficInfo && (() => {
                const isObj = typeof trafficInfo.traffic === "object";
                const label = isObj ? trafficInfo.traffic.label : trafficInfo.traffic;
                const emoji = isObj ? trafficInfo.traffic.emoji : "";
                return (
                  <div style={{ marginBottom: 16, padding: "8px 12px", borderRadius: 8, background: "var(--surface)", border: "1px solid var(--border)", fontSize: 13, color: "var(--muted)" }}>
                    AI Traffic: {emoji} <strong>{label}</strong>
                  </div>
                );
              })()}
              <div style={{ display: "flex", gap: 10 }}>
                <button className="ghost-btn" style={{ flex: 1 }} onClick={() => setBookingStep(null)}>Cancel</button>
                <button className="primary-btn" style={{ flex: 2 }} onClick={confirmBook}>Confirm →</button>
              </div>
            </div>
          </div>
        )}

        {bookingStep === "booked" && (
          <div className="booking-overlay" onClick={() => setBookingStep(null)}>
            <div className="booking-card" style={{ textAlign: "center" }} onClick={e => e.stopPropagation()}>
              <div style={{ fontSize: 48, marginBottom: 12 }}>🎉</div>
              <div style={{ fontSize: 20, fontWeight: 800, color: "var(--ink)", marginBottom: 6 }}>Ride Booked!</div>
              <div style={{ fontSize: 13, color: "var(--muted)", marginBottom: 20 }}>Your driver is on the way. Enjoy the ride!</div>
              <button className="primary-btn" onClick={clearAll}>Start New Trip</button>
            </div>
          </div>
        )}
      </div>
    </>
  );
} 