# server/main.py

"""
Main FastAPI application file.

REST ingestion from Arduino (via scripts/ingest_arduino.py):
 - POST /api/sensor-data   <- receives corrected sensor payloads
 - POST /api/status        <- receives online/offline status
 - POST /api/calibration   <- receives calibration metadata
 - GET  /api/history       <- returns recent sensor readings (in-memory)

Each sensor payload triggers (when change is significant):
 - An automatic crop prediction using loaded ML models
 - Broadcast over WebSocket (/ws/esp32) with:
     type: "sensor"     -> latest sensor reading
     type: "prediction" -> latest prediction
     type: "status"     -> device online/offline
     type: "calibration"-> calibration info
     type: "history"    -> recent sensor readings on connect
"""

import asyncio
import json
import time
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import (
    FastAPI,
    HTTPException,
    WebSocket,
    WebSocketDisconnect,
    Request,
)
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import io
import os

import pandas as pd  # <-- NEW: for fertilizer CSV

# --- Matplotlib (for offline maps) ---
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point

# --- State Lookup (GeoPandas) ---
import geopandas as gpd

# --- Local Imports ---
try:
    from server.models_loader import model_artifacts
except ImportError:
    model_artifacts = None  # Set to None to handle startup failure

from server.utils import get_disease_alerts
from server.db_json import save_prediction, read_prediction_history
from utils.area_utils import calculate_area_acres

# ---------------- In-memory sensor / status state ----------------
SENSOR_HISTORY: List[Dict[str, Any]] = []  # last N sensor readings
LAST_STATUS: Dict[str, Any] = {"online": False, "updated_at": None}
LAST_CALIBRATION: Optional[Dict[str, Any]] = None

# Auto-prediction significant-change memory (for /api/sensor-data)
LAST_LIVE_FEATURES: Optional[Dict[str, float]] = None

SIGNIFICANT_THRESHOLDS: Dict[str, float] = {
    "N": 5.0,
    "P": 5.0,
    "K": 5.0,
    "pH": 0.2,
    "temperature": 1.0,
    "humidity": 5.0,
    "rainfall": 50.0,
}


def has_significant_change(
    previous: Optional[Dict[str, float]],
    current: Dict[str, float],
    thresholds: Dict[str, float],
) -> bool:
    """Return True if ANY feature changes more than its threshold."""
    if previous is None:
        return True
    for key, threshold in thresholds.items():
        old = previous.get(key)
        new = current.get(key)
        if old is None or new is None:
            continue
        try:
            if abs(float(new) - float(old)) >= float(threshold):
                return True
        except Exception:
            continue
    return False


# ---------------- Load State Boundary File ----------------
STATE_FILE_PATH = "data/india_states.geojson"
STATE_BOUNDARIES = None
try:
    if os.path.exists(STATE_FILE_PATH):
        STATE_BOUNDARIES = gpd.read_file(STATE_FILE_PATH)
        print("[INFO] Successfully loaded India state boundaries for lookup.")
    else:
        print(f"[WARN] State boundary file not found at: {STATE_FILE_PATH}")
        print("[WARN] State lookup will be disabled.")
except Exception as e:
    print(f"[ERROR] Could not load state boundary file: {e}")


def get_state_from_coords(lat: float, lon: float) -> str:
    """Find the state name from lat/lon using the loaded GeoJSON."""
    if STATE_BOUNDARIES is None:
        return "Unknown (No GeoJSON)"

    try:
        point = Point(lon, lat)
        containing_states = STATE_BOUNDARIES[STATE_BOUNDARIES.contains(point)]

        if not containing_states.empty:
            state_name = "Unknown State"
            if "NAME_1" in containing_states.columns:
                state_name = containing_states.iloc[0]["NAME_1"]
            elif "st_nm" in containing_states.columns:
                state_name = containing_states.iloc[0]["st_nm"]
            elif "state" in containing_states.columns:
                state_name = containing_states.iloc[0]["state"]
            return state_name
        else:
            return "Outside Boundaries"
    except Exception as e:
        print(f"[ERROR] State lookup failed: {e}")
        return "Lookup Error"


# ---------------- Fertilizer mapping from dataset ----------------
FERTILIZER_MAP: Dict[str, str] = {}

try:
    # Adjust path if your CSV is elsewhere
    FERT_CSV_PATH = os.path.join("data", "arginode_corrected_fertilizers.csv")
    if os.path.exists(FERT_CSV_PATH):
        fert_df = pd.read_csv(FERT_CSV_PATH)

        # Try to detect crop column
        crop_col = None
        for c in ["crop", "Crop", "label", "Label", "CropName", "Crop_Name"]:
            if c in fert_df.columns:
                crop_col = c
                break

        # Try to detect fertilizer column
        fert_col = None
        for c in ["fertilizer", "Fertilizer", "fertilizer_name", "Fertilizer_Name"]:
            if c in fert_df.columns:
                fert_col = c
                break

        if crop_col and fert_col:
            for crop_value, group in fert_df.groupby(crop_col):
                vals = (
                    group[fert_col]
                    .dropna()
                    .astype(str)
                    .str.strip()
                )
                if not vals.empty:
                    most_common = vals.value_counts().idxmax()
                    key = str(crop_value).strip().lower()
                    FERTILIZER_MAP[key] = most_common

            print(f"[INFO] Loaded {len(FERTILIZER_MAP)} fertilizer rules from CSV.")
        else:
            print("[WARN] Could not detect crop/fertilizer columns in fertilizer CSV.")
    else:
        print(f"[WARN] Fertilizer CSV not found at: {FERT_CSV_PATH}")
except Exception as e:
    print(f"[ERROR] Failed to load fertilizer CSV: {e}")


# ---------------- Data Models (Pydantic) ----------------
class PolygonPoint(BaseModel):
    lat: float
    lng: float


class PredictionInput(BaseModel):
    N: float
    P: float
    K: float
    soil_ph: float = Field(..., alias="pH")
    avg_temp_c: float = Field(..., alias="temperature")
    soil_moisture_pct: float = Field(..., alias="humidity")
    annual_rainfall_mm: float = Field(..., alias="rainfall")
    latitude: float
    longitude: float
    device_id: Optional[str] = None
    timestamp: Optional[str] = None
    place_name: Optional[str] = None
    polygon: Optional[List[PolygonPoint]] = None

    class Config:
        populate_by_name = True


class SavedPredictionData(PredictionInput):
    state: Optional[str] = None


class PredictionResponse(BaseModel):
    predicted_crop: str
    confidence: float
    fertilizer_recommendation: str
    disease_alerts: List[str]
    area_acres: Optional[float] = None
    warnings: List[str]
    received_data: SavedPredictionData


# ---------------- Initialize FastAPI App ----------------
app = FastAPI()

# Allow frontend dev server (localhost:3000) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # you can restrict to ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- WebSocket state ----------------
ws_clients: set[WebSocket] = set()


async def broadcast_ws(message: dict):
    """Send a JSON message to all connected websocket clients."""
    dead: List[WebSocket] = []
    for ws in ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            dead.append(ws)
    for ws in dead:
        try:
            ws_clients.remove(ws)
        except KeyError:
            pass


# ---------------- Core prediction helper ----------------
def _make_prediction(input_data: PredictionInput) -> Dict[str, Any]:
    warnings: List[str] = []
    try:
        feature_values = [
            input_data.N,
            input_data.P,
            input_data.K,
            input_data.soil_ph,
            input_data.annual_rainfall_mm,
            input_data.avg_temp_c,
            input_data.soil_moisture_pct,
        ]
        features_np = np.array(feature_values, dtype=float).reshape(1, -1)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid feature data: {e}")

    scaler = model_artifacts["scaler"]
    features_scaled = scaler.transform(features_np)

    model = model_artifacts["crop_model"]
    le = model_artifacts["label_encoder"]

    pred_numeric = model.predict(features_scaled)
    pred_proba = model.predict_proba(features_scaled)

    crop_name = le.inverse_transform(pred_numeric)[0]
    confidence = float(np.max(pred_proba))

    # ---------- Fertilizer recommendation ----------
    # 1) Prefer mapping from your dataset CSV
    fert_from_dataset = None
    try:
        fert_from_dataset = FERTILIZER_MAP.get(str(crop_name).strip().lower())
    except Exception:
        fert_from_dataset = None

    fertilizer_rec: Optional[str] = None

    if fert_from_dataset:
        fertilizer_rec = fert_from_dataset
    else:
        # 2) Fall back to any existing fertilizer_recommender in model_artifacts
        recommender = None
        try:
            if isinstance(model_artifacts, dict):
                recommender = model_artifacts.get("fertilizer_recommender")
        except Exception:
            recommender = None

        if recommender is not None:
            try:
                fertilizer_rec = recommender.get_recommendation(
                    crop_name, features=features_np.flatten()
                )
            except Exception as e:
                warnings.append(f"Fertilizer recommender failed: {e}")
                fertilizer_rec = None

    # 3) Final fallback if nothing worked
    if not fertilizer_rec:
        fertilizer_rec = "No specific fertilizer found in dataset"

    disease_alerts = get_disease_alerts(crop_name)

    area_acres = None
    if input_data.polygon:
        try:
            polygon_data = [p.dict() for p in input_data.polygon]
            area_acres = calculate_area_acres(polygon_data)
        except Exception as e:
            warnings.append(f"Could not calculate area: {e}")

    return {
        "predicted_crop": crop_name,
        "confidence": confidence,
        "fertilizer_recommendation": fertilizer_rec,
        "disease_alerts": disease_alerts,
        "area_acres": area_acres,
        "warnings": warnings,
    }


# ---------------- REST endpoints for ingestion script ----------------
@app.post("/api/sensor-data")
async def receive_sensor_data(request: Request):
    """
    Accepts corrected sensor payload from ingestion script.
    Also triggers prediction (if change is significant) and broadcasts
    sensor + prediction via WebSocket.
    """
    global LAST_LIVE_FEATURES

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    # Base sensor entry (what LiveFeed + history will see)
    entry: Dict[str, Any] = {
        "ts": payload.get("ts", time.time()),
        "raw": payload.get("raw"),
        "corrected": payload.get("corrected"),
        "calibrated": payload.get("calibrated", False),
        "received_at": time.time(),
        # extra fields from Arduino / ingestion script
        "ph": payload.get("ph", payload.get("pH")),
        "temperature": payload.get("temperature", payload.get("temp")),
        "humidity": payload.get("humidity"),
        "moisture": payload.get("moisture"),
        "lat": payload.get("lat", payload.get("latitude")),
        "lon": payload.get("lon", payload.get("longitude")),
        "place_name": payload.get("place_name"),
    }

    # Keep in-memory history
    SENSOR_HISTORY.append(entry)
    if len(SENSOR_HISTORY) > 1000:
        SENSOR_HISTORY.pop(0)

    # Broadcast sensor reading to all LiveFeed clients
    await broadcast_ws({"type": "sensor", "data": entry})

    # ---------------- Auto prediction ----------------
    if model_artifacts is not None:
        try:
            corrected = entry.get("corrected") or {}
            n_val = corrected.get("n", corrected.get("N", 0.0))
            p_val = corrected.get("p", corrected.get("P", 0.0))
            k_val = corrected.get("k", corrected.get("K", 0.0))

            ph_val = entry.get("ph", 7.0)
            temp_val = entry.get("temperature", 25.0)
            hum_val = entry.get("humidity", 50.0)
            rain_val = payload.get("rainfall", 0.0)  # rainfall may not be present
            lat_val = entry.get("lat", 0.0)
            lon_val = entry.get("lon", 0.0)

            def _f(v, default=0.0):
                try:
                    return float(v)
                except Exception:
                    return float(default)

            n_val = _f(n_val, 0.0)
            p_val = _f(p_val, 0.0)
            k_val = _f(k_val, 0.0)
            ph_val = _f(ph_val, 7.0)
            temp_val = _f(temp_val, 25.0)
            hum_val = _f(hum_val, 50.0)
            rain_val = _f(rain_val, 0.0)
            lat_val = _f(lat_val, 0.0)
            lon_val = _f(lon_val, 0.0)

            current_features = {
                "N": n_val,
                "P": p_val,
                "K": k_val,
                "pH": ph_val,
                "temperature": temp_val,
                "humidity": hum_val,
                "rainfall": rain_val,
            }

            # only predict if values moved enough
            if not has_significant_change(
                LAST_LIVE_FEATURES, current_features, SIGNIFICANT_THRESHOLDS
            ):
                return {"ok": True, "skipped_prediction": True}

            LAST_LIVE_FEATURES = current_features

            pi = PredictionInput(
                N=n_val,
                P=p_val,
                K=k_val,
                pH=ph_val,
                temperature=temp_val,
                humidity=hum_val,
                rainfall=rain_val,
                latitude=lat_val,
                longitude=lon_val,
                place_name=payload.get("place_name", "Arduino Live"),
                timestamp=str(entry["ts"]),
            )

            results = _make_prediction(pi)
            state_name = get_state_from_coords(pi.latitude, pi.longitude)

            saved_data_dict = pi.dict(by_alias=False)
            saved_data_dict["state"] = state_name
            saved_data_obj = SavedPredictionData(**saved_data_dict)

            response_data = {**results, "received_data": saved_data_obj}

            # Save to history JSON
            try:
                save_payload = response_data.copy()
                save_payload["received_data"] = save_payload["received_data"].dict()
                save_payload["source"] = "ESP32"
                save_prediction(save_payload)
            except Exception as e:
                print(f"[ERROR] Failed to save live prediction to JSON DB: {e}")
                response_data["warnings"].append(
                    "Failed to save live prediction to history."
                )

            # Broadcast to WebSocket clients
            ws_payload = response_data.copy()
            ws_payload["received_data"] = ws_payload["received_data"].dict()
            await broadcast_ws({"type": "prediction", "data": ws_payload})

        except Exception as e:
            print(f"[ERROR] Live prediction from /api/sensor-data failed: {e}")

    return {"ok": True}


@app.post("/api/status")
async def set_status(request: Request):
    """Accepts {"online": true/false} from ingestion script."""
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    online = bool(payload.get("online", False))
    LAST_STATUS["online"] = online
    LAST_STATUS["updated_at"] = time.time()

    await broadcast_ws({"type": "status", "data": LAST_STATUS})
    return {"ok": True}


@app.post("/api/calibration")
async def set_calibration(request: Request):
    """Accepts calibration metadata."""
    global LAST_CALIBRATION
    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    LAST_CALIBRATION = payload
    await broadcast_ws({"type": "calibration", "data": LAST_CALIBRATION})
    return {"ok": True}


@app.get("/api/history")
async def get_sensor_history(limit: int = 100):
    """Return recent sensor readings (in-memory)."""
    return SENSOR_HISTORY[-limit:]


# ---------------- Startup / Shutdown ----------------
@app.on_event("startup")
async def startup_event():
    if model_artifacts is None:
        print("\n[CRITICAL ERROR] MODELS ARE NOT LOADED.")
    else:
        print("\n[INFO] Server started successfully. Models are loaded.\n")


@app.on_event("shutdown")
async def shutdown_event():
    print("[INFO] Server shutting down.")


# ---------------- WebSocket endpoint ----------------
@app.websocket("/ws/esp32")
async def ws_esp32(websocket: WebSocket):
    """
    Single WebSocket endpoint used by the React LiveFeed.
    On connect:
      - sends last status
      - sends last calibration
      - sends recent sensor history (up to 15)
      - sends last saved prediction (from JSON history)
    Then keeps the connection alive.
    """
    await websocket.accept()
    ws_clients.add(websocket)

    try:
        # status
        if LAST_STATUS["updated_at"] is not None:
            await websocket.send_json({"type": "status", "data": LAST_STATUS})

        # calibration
        if LAST_CALIBRATION is not None:
            await websocket.send_json({"type": "calibration", "data": LAST_CALIBRATION})

        # history (last 15 readings)
        history_slice = SENSOR_HISTORY[-15:]
        await websocket.send_json({"type": "history", "data": history_slice})

        # latest prediction from JSON DB
        history = read_prediction_history()
        if history:
            latest = history[-1]
            await websocket.send_json({"type": "prediction", "data": latest})

        # keep connection alive
        while True:
            await asyncio.sleep(30)

    except WebSocketDisconnect:
        pass
    finally:
        if websocket in ws_clients:
            ws_clients.remove(websocket)


# ---------------- Manual prediction API ----------------
@app.get("/")
def get_root():
    if model_artifacts is None:
        return {"status": "error", "message": "Models are not loaded."}
    return {"status": "ok", "message": "Offline ML Server is running."}


@app.post("/predict_crop", response_model=PredictionResponse)
def post_predict_crop(input_data: PredictionInput):
    if model_artifacts is None:
        raise HTTPException(status_code=503, detail="Models are not loaded.")

    try:
        results = _make_prediction(input_data)
        state_name = get_state_from_coords(
            input_data.latitude, input_data.longitude
        )

        saved_data_dict = input_data.dict(by_alias=False)
        saved_data_dict["state"] = state_name
        saved_data_obj = SavedPredictionData(**saved_data_dict)

        response_data = {**results, "received_data": saved_data_obj}

        try:
            save_payload = response_data.copy()
            save_payload["received_data"] = save_payload["received_data"].dict()
            save_payload["source"] = "manual"
            save_prediction(save_payload)
            print("[INFO] Prediction successfully saved to db/predictions.json")
        except Exception as e:
            print(f"[ERROR] Failed to save prediction to JSON DB: {e}")
            response_data["warnings"].append(
                "Failed to save prediction to history."
            )

        return response_data
    except Exception as e:
        print(f"[ERROR] Unhandled error in /predict_crop: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/history")
def get_history():
    return read_prediction_history()


# ---------------- Offline map endpoint ----------------
@app.get("/history_map/{prediction_id}.png")
def get_history_map_image(prediction_id: str):
    history = read_prediction_history()
    record = next((item for item in history if item.get("id") == prediction_id), None)

    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    data = record.get("received_data", {})
    lat = data.get("latitude")
    lon = data.get("longitude")
    state_name = data.get("state", None)
    polygon_data = data.get("polygon")

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_facecolor("#aadaff")
    ax.set_facecolor("#aadaff")

    if STATE_BOUNDARIES is not None:
        STATE_BOUNDARIES.plot(
            ax=ax, facecolor="#eeeeee", edgecolor="#999999", linewidth=0.5
        )
    else:
        ax.text(
            0.5,
            0.5,
            "india_states.geojson file missing",
            horizontalalignment="center",
            verticalalignment="center",
            transform=ax.transAxes,
            fontsize=12,
            color="red",
        )

    if state_name and STATE_BOUNDARIES is not None:
        state_col = None
        if "NAME_1" in STATE_BOUNDARIES.columns:
            state_col = "NAME_1"
        elif "st_nm" in STATE_BOUNDARIES.columns:
            state_col = "st_nm"
        elif "state" in STATE_BOUNDARIES.columns:
            state_col = "state"

        if state_col:
            highlight_state = STATE_BOUNDARIES[STATE_BOUNDARIES[state_col] == state_name]
            if not highlight_state.empty:
                highlight_state.plot(
                    ax=ax,
                    facecolor="orange",
                    edgecolor="black",
                    linewidth=1.0,
                )

    if polygon_data and len(polygon_data) >= 3:
        poly_coords = [(p["lng"], p["lat"]) for p in polygon_data]
        poly_shape = Polygon(poly_coords)
        gpd.GeoSeries([poly_shape]).plot(
            ax=ax,
            facecolor="green",
            edgecolor="darkgreen",
            alpha=0.7,
            label="Field Boundary",
        )

    if lat and lon:
        ax.plot(
            lon,
            lat,
            "o",
            color="red",
            markersize=8,
            markeredgecolor="white",
            label="Prediction Point",
        )

    title = data.get("place_name", "Prediction Location")
    if state_name:
        title = f"{title}\n(State: {state_name})"
    ax.set_title(title, fontsize=14)

    ax.set_aspect("equal")
    ax.axis("off")

    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format="png", bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)
    img_buffer.seek(0)
    return StreamingResponse(img_buffer, media_type="image/png")