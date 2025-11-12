# server/main.py
"""
Main FastAPI application file.
Serial -> WebSocket bridge now also runs automatic predictions for every incoming sensor packet,
then broadcasts both the raw sensor (type: "sensor") and the prediction result (type: "prediction")
to connected websocket clients.

This variant adds a `source` field to saved prediction records:
 - "ESP32" for predictions triggered by the serial live feed
 - "manual" for predictions triggered by the form POST
"""

import asyncio
import json
import threading
import time
from typing import List, Dict, Any, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# Serial support
import serial  # pip install pyserial

# stdlib
import io
import os

# --- Matplotlib (for offline maps) ---
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point
# -------------------------------------

# --- State Lookup (GeoPandas) ---
import geopandas as gpd
# --------------------------------

# --- Local Imports ---
try:
    from server.models_loader import model_artifacts
except ImportError:
    model_artifacts = None  # Set to None to handle startup failure

from server.utils import get_disease_alerts
from server.db_json import save_prediction, read_prediction_history
from utils.area_utils import calculate_area_acres

# --- Serial config (adjust for your machine) ---
SERIAL_PORT = os.environ.get("ESP32_SERIAL_PORT", "/dev/tty.SLAB_USBtoUART")
SERIAL_BAUD = int(os.environ.get("ESP32_SERIAL_BAUD", "115200"))
SERIAL_READ_TIMEOUT = 1.0  # seconds

# --- WebSocket / Serial bridge globals ---
connected_websockets = set()  # set of WebSocket
serial_thread = None
serial_thread_stop = threading.Event()
asyncio_loop = None  # will be set at startup

# --- Load State Boundary File ---
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
    """Finds the state name from lat/lon using the loaded GeoJSON."""
    if STATE_BOUNDARIES is None:
        return "Unknown (No GeoJSON)"
    
    try:
        point = Point(lon, lat) # GeoPandas/Shapely use (lon, lat)
        containing_states = STATE_BOUNDARIES[STATE_BOUNDARIES.contains(point)]
        
        if not containing_states.empty:
            state_name = "Unknown State"
            if 'NAME_1' in containing_states.columns:
                state_name = containing_states.iloc[0]['NAME_1']
            elif 'st_nm' in containing_states.columns:
                state_name = containing_states.iloc[0]['st_nm']
            elif 'state' in containing_states.columns:
                state_name = containing_states.iloc[0]['state']
            
            return state_name
        else:
            return "Outside Boundaries"
    except Exception as e:
        print(f"[ERROR] State lookup failed: {e}")
        return "Lookup Error"


# --- Data Models (Pydantic) ---
class PolygonPoint(BaseModel):
    lat: float
    lng: float

class PredictionInput(BaseModel):
    N: float
    P: float
    K: float
    soil_ph: float = Field(..., alias='pH')
    avg_temp_c: float = Field(..., alias='temperature')
    soil_moisture_pct: float = Field(..., alias='humidity')
    annual_rainfall_mm: float = Field(..., alias='rainfall')
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

# --- Initialize FastAPI App ---
app = FastAPI()

# -------------------------
# Helper broadcast function
# -------------------------
async def broadcast_text(text: str):
    """Send a text message to all connected websocket clients (safe gather)."""
    if not connected_websockets:
        return
    ws_snapshot = list(connected_websockets)
    coros = []
    for ws in ws_snapshot:
        try:
            coros.append(ws.send_text(text))
        except Exception:
            pass
    if coros:
        try:
            await asyncio.gather(*coros, return_exceptions=True)
        except Exception:
            pass

# -------------------------
# Serial reader thread
# -------------------------
def safe_float(v):
    try:
        return float(v)
    except Exception:
        return None

def build_prediction_payload(parsed: dict) -> Dict[str, Any]:
    """
    Given parsed sensor data (dict), run the ML prediction pipeline and build a JSON-serializable result.
    This function also attaches a 'source' field = "ESP32" to the saved payload.
    """
    # Map incoming keys to PredictionInput names (be tolerant)
    mapped = {
        "N": parsed.get("N") if parsed.get("N") is not None else parsed.get("n"),
        "P": parsed.get("P") if parsed.get("P") is not None else parsed.get("p"),
        "K": parsed.get("K") if parsed.get("K") is not None else parsed.get("k"),
        "pH": parsed.get("pH") if parsed.get("pH") is not None else parsed.get("ph"),
        "temperature": parsed.get("temp") if parsed.get("temp") is not None else parsed.get("temperature"),
        "humidity": parsed.get("humidity"),
        "rainfall": parsed.get("rainfall"),
        "latitude": parsed.get("lat") if parsed.get("lat") is not None else parsed.get("latitude"),
        "longitude": parsed.get("lon") if parsed.get("lon") is not None else parsed.get("longitude"),
    }

    # Convert types (safe)
    for k in ["N","P","K","pH","temperature","humidity","rainfall","latitude","longitude"]:
        if mapped.get(k) is not None:
            try:
                mapped[k] = float(mapped[k])
            except Exception:
                mapped[k] = mapped[k]

    # Fill required fields with defaults if missing (so _make_prediction can fail gracefully)
    for required in ["N","P","K","pH","temperature","humidity","rainfall","latitude","longitude"]:
        if mapped.get(required) is None:
            # If a value missing, raise later in _make_prediction
            mapped[required] = 0.0

    try:
        pi = PredictionInput(
            N = mapped["N"],
            P = mapped["P"],
            K = mapped["K"],
            pH = mapped["pH"],
            temperature = mapped["temperature"],
            humidity = mapped["humidity"],
            rainfall = mapped["rainfall"],
            latitude = mapped["latitude"],
            longitude = mapped["longitude"],
            place_name = parsed.get("place_name", "ESP32 Live"),
            timestamp = parsed.get("ts")
        )
    except Exception as e:
        raise

    # Run prediction
    results = _make_prediction(pi)

    # Find state
    state_name = get_state_from_coords(pi.latitude, pi.longitude)

    # Build received_data for response (serializable)
    rec = pi.dict(by_alias=False)
    rec["state"] = state_name

    # Compose final response payload
    response_payload = {
        "predicted_crop": results.get("predicted_crop"),
        "confidence": results.get("confidence"),
        "fertilizer_recommendation": results.get("fertilizer_recommendation"),
        "disease_alerts": results.get("disease_alerts"),
        "area_acres": results.get("area_acres"),
        "warnings": results.get("warnings", []),
        "received_data": rec
    }

    # Try to save to history (best effort) and tag source = "ESP32"
    try:
        to_save = response_payload.copy()
        to_save["received_data"] = to_save["received_data"]
        # add source tag for history origin
        to_save["source"] = "ESP32"
        save_prediction(to_save)
    except Exception:
        # Don't fail the pipeline if saving fails
        pass

    return response_payload

def serial_reader_thread(port: str, baud: int):
    """Thread that reads serial lines and forwards them to websocket clients and also produces predictions."""
    try:
        ser = serial.Serial(port=port, baudrate=baud, timeout=SERIAL_READ_TIMEOUT)
    except Exception as e:
        print(f"[SERIAL] Could not open serial port {port} at {baud} baud: {e}")
        return

    while not serial_thread_stop.is_set():
        try:
            raw = ser.readline()
            if not raw:
                continue
            try:
                line = raw.decode(errors='ignore').strip()
            except Exception:
                line = raw.decode('utf-8', 'ignore').strip()

            if not line:
                continue

            # Attempt parse JSON; if fails, try simple k=v pairs fallback
            parsed = None
            try:
                parsed = json.loads(line)
            except Exception:
                # parse loose "k=v k2=v2" style
                try:
                    obj = {}
                    for pair in line.split():
                        if "=" in pair:
                            k,v = pair.split("=",1)
                            num = None
                            try:
                                num = float(v)
                            except:
                                num = v
                            obj[k.strip()] = num
                    parsed = obj if obj else {"raw": line}
                except Exception:
                    parsed = {"raw": line}

            # Broadcast sensor message
            sensor_message = {"type":"sensor", "data": parsed}
            try:
                if asyncio_loop and not asyncio_loop.is_closed():
                    asyncio.run_coroutine_threadsafe(broadcast_text(json.dumps(sensor_message)), asyncio_loop)
            except Exception:
                pass

            # Try make prediction only when parsed has at least N,P,K or numeric keys
            try:
                # If parsed contains numeric keys, proceed
                if any(k in parsed for k in ("N","P","K","n","p","k")):
                    prediction_payload = build_prediction_payload(parsed)
                    prediction_message = {"type":"prediction", "data": prediction_payload}
                    if asyncio_loop and not asyncio_loop.is_closed():
                        asyncio.run_coroutine_threadsafe(broadcast_text(json.dumps(prediction_message)), asyncio_loop)
            except Exception:
                # If prediction fails, ignore and continue
                pass

        except Exception as e:
            print(f"[SERIAL] Read error: {e}")
            time.sleep(0.5)

    try:
        ser.close()
    except Exception:
        pass
    print("[SERIAL] Serial reader thread stopped")

# -------------------------
# Startup / Shutdown
# -------------------------
@app.on_event("startup")
async def startup_event():
    global asyncio_loop, serial_thread, serial_thread_stop
    try:
        asyncio_loop = asyncio.get_event_loop()
    except Exception:
        asyncio_loop = None

    serial_thread_stop.clear()
    # Start the serial thread even if device not present — it will try to open
    serial_thread = threading.Thread(target=serial_reader_thread, args=(SERIAL_PORT, SERIAL_BAUD), daemon=True)
    serial_thread.start()
    print(f"[SERIAL] Serial thread started (attempting {SERIAL_PORT}).")

    if model_artifacts is None:
        print("\n[CRITICAL ERROR] MODELS ARE NOT LOADED.")
    else:
        print("\n[INFO] Server started successfully. Models are loaded.\n")

@app.on_event("shutdown")
async def shutdown_event():
    global serial_thread_stop, serial_thread
    serial_thread_stop.set()
    if serial_thread and serial_thread.is_alive():
        serial_thread.join(timeout=2.0)
    print("[SERIAL] Shutdown complete")

# -------------------------
# WebSocket endpoint (clients connect to receive serial data & predictions)
# -------------------------
@app.websocket("/ws/esp32")
async def ws_esp32(websocket: WebSocket):
    """
    WebSocket endpoint consumed by the frontend.
    Clients connecting here will receive messages of two types:
      - {"type":"sensor", "data": {...}}  (raw sensor packet)
      - {"type":"prediction", "data": {...}} (prediction response)
    """
    await websocket.accept()
    connected_websockets.add(websocket)
    try:
        while True:
            try:
                _ = await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                await asyncio.sleep(0.1)
    finally:
        try:
            connected_websockets.discard(websocket)
        except Exception:
            pass

# -------------------------
# Prediction API (unchanged except tagging saved record source = "manual")
# -------------------------
def _make_prediction(input_data: PredictionInput) -> Dict[str, Any]:
    warnings = []
    try:
        feature_values = [
            input_data.N, input_data.P, input_data.K, input_data.soil_ph,
            input_data.annual_rainfall_mm, input_data.avg_temp_c, input_data.soil_moisture_pct
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
    recommender = model_artifacts["fertilizer_recommender"]
    fertilizer_rec = recommender.get_recommendation(crop_name, features=features_np.flatten())
    disease_alerts = get_disease_alerts(crop_name)
    area_acres = None
    if input_data.polygon:
        try:
            polygon_data = [p.dict() for p in input_data.polygon]
            area_acres = calculate_area_acres(polygon_data)
        except Exception as e:
            warnings.append(f"Could not calculate area: {e}")
    return {"predicted_crop": crop_name, "confidence": confidence, "fertilizer_recommendation": fertilizer_rec,
            "disease_alerts": disease_alerts, "area_acres": area_acres, "warnings": warnings}

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
        state_name = get_state_from_coords(input_data.latitude, input_data.longitude)
        
        saved_data_dict = input_data.dict(by_alias=False)
        saved_data_dict['state'] = state_name
        saved_data_obj = SavedPredictionData(**saved_data_dict)
        
        response_data = {**results, "received_data": saved_data_obj}
        
        try:
            save_payload = response_data.copy()
            save_payload["received_data"] = save_payload["received_data"].dict()
            # tag manual/form source
            save_payload["source"] = "manual"
            save_prediction(save_payload)
            print("[INFO] Prediction successfully saved to db/predictions.json")
        except Exception as e:
            print(f"[ERROR] Failed to save prediction to JSON DB: {e}")
            response_data["warnings"].append("Failed to save prediction to history.")

        return response_data
        
    except Exception as e:
        print(f"[ERROR] Unhandled error in /predict_crop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history():
    return read_prediction_history()

# --- THIS IS THE "BEAUTIFIED" OFFLINE MAP ENDPOINT (unchanged) ---
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
    fig.patch.set_facecolor('#aadaff')
    ax.set_facecolor('#aadaff')
    
    if STATE_BOUNDARIES is not None:
        STATE_BOUNDARIES.plot(ax=ax, facecolor='#eeeeee', edgecolor='#999999', linewidth=0.5)
    else:
        ax.text(0.5, 0.5, 'india_states.geojson file missing',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=12, color='red')

    if state_name and STATE_BOUNDARIES is not None:
        state_col = None
        if 'NAME_1' in STATE_BOUNDARIES.columns: state_col = 'NAME_1'
        elif 'st_nm' in STATE_BOUNDARIES.columns: state_col = 'st_nm'
        elif 'state' in STATE_BOUNDARIES.columns: state_col = 'state'
        
        if state_col:
            highlight_state = STATE_BOUNDARIES[STATE_BOUNDARIES[state_col] == state_name]
            if not highlight_state.empty:
                highlight_state.plot(ax=ax, facecolor='orange', edgecolor='black', linewidth=1.0)

    if polygon_data and len(polygon_data) >= 3:
        poly_coords = [(p['lng'], p['lat']) for p in polygon_data]
        poly_shape = Polygon(poly_coords)
        gpd.GeoSeries([poly_shape]).plot(
            ax=ax,
            facecolor='green',
            edgecolor='darkgreen',
            alpha=0.7,
            label='Field Boundary'
        )
    
    if lat and lon:
        ax.plot(lon, lat, 'o', color='red', markersize=8, markeredgecolor='white', label='Prediction Point')
        
    title = data.get("place_name", "Prediction Location")
    if state_name:
        title = f"{title}\n(State: {state_name})"
    ax.set_title(title, fontsize=14)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)
    img_buffer.seek(0)
    return StreamingResponse(img_buffer, media_type="image/png")