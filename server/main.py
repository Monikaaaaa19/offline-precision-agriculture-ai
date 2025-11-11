# server/main.py
"""
Main FastAPI application file.
NOW INCLUDES "BEAUTIFIED" 100% OFFLINE MAP (HIGHLIGHTS THE STATE IN ORANGE).
"""

import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse # Used for sending images
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import io # Used for sending images
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
    model_artifacts = None # Set to None to handle startup failure

from server.utils import get_disease_alerts
from server.db_json import save_prediction, read_prediction_history
from utils.area_utils import calculate_area_acres

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
# (These are unchanged from the previous correct step)

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

# --- Helper Function ---
def _make_prediction(input_data: PredictionInput) -> Dict[str, Any]:
    # (Unchanged)
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


# --- API Endpoints ---

@app.on_event("startup")
def startup_event():
    # (Unchanged)
    if model_artifacts is None:
        print("\n[CRITICAL ERROR] MODELS ARE NOT LOADED.")
    else:
        print("\n[INFO] Server started successfully. Models are loaded.\n")

@app.get("/")
def get_root():
    # (Unchanged)
    if model_artifacts is None:
        return {"status": "error", "message": "Models are not loaded."}
    return {"status": "ok", "message": "Offline ML Server is running."}

@app.post("/predict_crop", response_model=PredictionResponse)
def post_predict_crop(input_data: PredictionInput):
    # (Unchanged from previous step)
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
    # (Unchanged)
    return read_prediction_history()


# --- THIS IS THE "BEAUTIFIED" OFFLINE MAP ENDPOINT ---

@app.get("/history_map/{prediction_id}.png")
def get_history_map_image(prediction_id: str):
    """
    Generates a "beautified" 100% offline PNG image of the prediction's location.
    It plots all of India as a base map and highlights the specific state in ORANGE.
    """
    history = read_prediction_history()
    record = next((item for item in history if item.get("id") == prediction_id), None)
    
    if record is None:
        raise HTTPException(status_code=404, detail="Prediction not found")

    data = record.get("received_data", {})
    lat = data.get("latitude")
    lon = data.get("longitude")
    state_name = data.get("state", None) # The state name we found
    polygon_data = data.get("polygon")
    
    fig, ax = plt.subplots(figsize=(8, 8)) # Use a square figure
    
    # Set "ocean" background
    fig.patch.set_facecolor('#aadaff') # Light blue
    ax.set_facecolor('#aadaff')
    
    # 1. Draw ALL of India as a base map (if file exists)
    if STATE_BOUNDARIES is not None:
        STATE_BOUNDARIES.plot(
            ax=ax,
            facecolor='#eeeeee', # Light gray "land" color
            edgecolor='#999999', # Darker gray borders
            linewidth=0.5
        )
        # This sets the bounds to show ALL of India
    else:
        # Fallback if GeoJSON is missing
        ax.text(0.5, 0.5, 'india_states.geojson file missing',
                horizontalalignment='center',
                verticalalignment='center',
                transform=ax.transAxes,
                fontsize=12, color='red')

    # 2. Find and draw the HIGHLIGHTED state (if we have a state name)
    if state_name and STATE_BOUNDARIES is not None:
        # Find the correct column name for the state
        state_col = None
        if 'NAME_1' in STATE_BOUNDARIES.columns: state_col = 'NAME_1'
        elif 'st_nm' in STATE_BOUNDARIES.columns: state_col = 'st_nm'
        elif 'state' in STATE_BOUNDARIES.columns: state_col = 'state'
        
        if state_col:
            # Find the specific state polygon
            highlight_state = STATE_BOUNDARIES[STATE_BOUNDARIES[state_col] == state_name]
            
            if not highlight_state.empty:
                # Plot the highlighted state on top
                highlight_state.plot(
                    ax=ax,
                    facecolor='orange', # <-- CHANGED TO ORANGE
                    edgecolor='black',
                    linewidth=1.0
                )
                # --- ZOOMING IS REMOVED ---

    # 3. Draw the user's field polygon (if it exists)
    if polygon_data and len(polygon_data) >= 3:
        poly_coords = [(p['lng'], p['lat']) for p in polygon_data]
        poly_shape = Polygon(poly_coords)
        gpd.GeoSeries([poly_shape]).plot(
            ax=ax,
            facecolor='green', # A different color to stand out
            edgecolor='darkgreen',
            alpha=0.7,
            label='Field Boundary'
        )
    
    # 4. Draw the single prediction point (Red Dot)
    if lat and lon:
        ax.plot(
            lon, lat, 'o', # 'o' is a circle marker
            color='red',
            markersize=8, # Made it a bit smaller
            markeredgecolor='white',
            label='Prediction Point'
        )
        
    # 5. Style the plot
    title = data.get("place_name", "Prediction Location")
    if state_name:
        title = f"{title}\n(State: {state_name})"
    ax.set_title(title, fontsize=14)
    
    ax.set_aspect('equal') # CRITICAL: Makes it look like a map
    ax.axis('off') # CRITICAL: Removes the ugly graph axes
    
    # 6. Save the plot to an in-memory file
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)  # Close the figure to free up memory
    img_buffer.seek(0)
    
    # 7. Return the image
    return StreamingResponse(img_buffer, media_type="image/png")