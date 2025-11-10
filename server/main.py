# server/main.py
"""
Main FastAPI application file.

This server runs 100% offline, loading local models to provide
crop, fertilizer, and disease alert predictions.
"""

import numpy as np
from fastapi import FastAPI, HTTPException, File, UploadFile
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# --- Local Imports ---
# These imports load our custom, offline modules

# 1. Load models from models_loader.py
# This `model_artifacts` is the dictionary we loaded on startup.
try:
    from server.models_loader import model_artifacts
except ImportError:
    print("\n[CRITICAL] server/models_loader.py not found.")
    print("[CRITICAL] Server cannot start without model loader.\n")
    model_artifacts = None # Set to None to handle startup failure

# 2. Import helper functions
from server.utils import get_disease_alerts
from server.db_json import save_prediction, read_prediction_history
from utils.area_utils import calculate_area_acres

# --- Data Models (using Pydantic) ---
# These classes define the expected JSON structure for our API

class PolygonPoint(BaseModel):
    """A single lat/lng point for a polygon."""
    lat: float
    lng: float

class PredictionInput(BaseModel):
    """
    This is the JSON we expect from the ESP32 or any client.
    Based on your example: {N,P,K,pH,temperature,humidity,rainfall,...}
    """
    # Core sensor features for the model
    # We must match the training order:
    # ['N', 'P', 'K', 'soil_ph', 'annual_rainfall_mm', 'avg_temp_c', 'soil_moisture_pct']

    N: float
    P: float
    K: float

    # *** FIX 1: Aliasing logic is corrected here ***
    # Our internal variable name is on the left (e.g., 'soil_ph')
    # The JSON key we expect from the client is on the right (e.g., alias='pH')
    soil_ph: float = Field(..., alias='pH')
    avg_temp_c: float = Field(..., alias='temperature')
    soil_moisture_pct: float = Field(..., alias='humidity')
    annual_rainfall_mm: float = Field(..., alias='rainfall')

    # Optional metadata
    device_id: Optional[str] = None
    timestamp: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    place_name: Optional[str] = None
    polygon: Optional[List[PolygonPoint]] = None

    class Config:
        # *** FIX 3: Updated for Pydantic v2 ***
        populate_by_name = True


class PredictionResponse(BaseModel):
    """This is the JSON response the server will send back."""
    predicted_crop: str
    confidence: float
    fertilizer_recommendation: str
    disease_alerts: List[str]
    area_acres: Optional[float] = None
    warnings: List[str]
    received_data: PredictionInput


# --- Initialize FastAPI App ---
app = FastAPI(
    title="Offline ML Prediction Server",
    description="Locally-hosted API for crop and disease prediction.",
    version="1.0.0"
)


# --- Helper Function ---
def _make_prediction(input_data: PredictionInput) -> Dict[str, Any]:
    """
    Internal function to run the prediction logic.
    Separated from the API endpoint for clarity.
    """
    warnings = []

    # 1. Extract features in the exact order the model was trained on
    # Order: ['N', 'P', 'K', 'soil_ph', 'annual_rainfall_mm', 'avg_temp_c', 'soil_moisture_pct']
    try:
        # *** FIX 2: Feature list updated to match Pydantic model ***
        feature_values = [
            input_data.N,
            input_data.P,
            input_data.K,
            input_data.soil_ph,
            input_data.annual_rainfall_mm,
            input_data.avg_temp_c,
            input_data.soil_moisture_pct
        ]
        features_np = np.array(feature_values, dtype=float).reshape(1, -1)
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid feature data: {e}")

    # 2. Scale features
    scaler = model_artifacts["scaler"]
    features_scaled = scaler.transform(features_np)

    # 3. Predict crop
    model = model_artifacts["crop_model"]
    le = model_artifacts["label_encoder"]

    pred_numeric = model.predict(features_scaled)
    pred_proba = model.predict_proba(features_scaled)

    crop_name = le.inverse_transform(pred_numeric)[0]
    confidence = float(np.max(pred_proba))

    # 4. Get Fertilizer Recommendation
    recommender = model_artifacts["fertilizer_recommender"]
    # We pass the *unscaled* numpy features, as the fertilizer
    # model's artifacts (if it exists) has its *own* scaler.
    fertilizer_rec = recommender.get_recommendation(
        crop_name,
        features=features_np.flatten() # Pass 1D array
    )

    # 5. Get Disease Alerts (Rule-Based)
    disease_alerts = get_disease_alerts(crop_name)

    # 6. Calculate Area (if polygon provided)
    area_acres = None
    if input_data.polygon:
        try:
            # Convert Pydantic models to simple dicts for the util function
            polygon_data = [p.dict() for p in input_data.polygon]
            area_acres = calculate_area_acres(polygon_data)
        except Exception as e:
            print(f"[ERROR] Area calculation failed: {e}")
            warnings.append(f"Could not calculate area: {e}")

    return {
        "predicted_crop": crop_name,
        "confidence": confidence,
        "fertilizer_recommendation": fertilizer_rec,
        "disease_alerts": disease_alerts,
        "area_acres": area_acres,
        "warnings": warnings,
    }


# --- API Endpoints ---

@app.on_event("startup")
def startup_event():
    """Code to run when the server starts."""
    if model_artifacts is None:
        print("\n[CRITICAL ERROR] MODELS ARE NOT LOADED.")
        print("[CRITICAL ERROR] Server is starting, but all /predict endpoints will fail.")
        print("[CRITICAL ERROR] Run 'python train_crop_model.py' and restart server.\n")
    else:
        print("\n[INFO] Server started successfully. Models are loaded.\n")

@app.get("/")
def get_root():
    """Root endpoint to check server status."""
    if model_artifacts is None:
        return {
            "status": "error",
            "message": "Models are not loaded. Run 'python train_crop_model.py'."
        }
    return {
        "status": "ok",
        "message": "Offline ML Server is running. Models are loaded.",
        "artifacts": list(model_artifacts.keys())
    }

@app.post("/predict_crop", response_model=PredictionResponse)
def post_predict_crop(input_data: PredictionInput):
    """
    Main endpoint for crop, fertilizer, and disease prediction.
    Accepts sensor JSON and returns a full recommendation.
    """
    if model_artifacts is None:
        raise HTTPException(
            status_code=503, # 503 Service Unavailable
            detail="Models are not loaded. Cannot make predictions."
        )

    try:
        # 1. Get prediction results
        results = _make_prediction(input_data)

        # 2. Combine results with input data for the response
        response_data = {
            **results,
            "received_data": input_data
        }

        # 3. Save to our JSON database (in the background)
        # We save the full response data for history
        try:
            # We need to convert the Pydantic model to a dict to save it
            save_data = response_data.copy()
            save_data["received_data"] = save_data["received_data"].dict()
            save_prediction(save_data)
        except Exception as e:
            print(f"[ERROR] Failed to save prediction to JSON DB: {e}")
            # We don't fail the request, just add a warning
            response_data["warnings"].append("Failed to save prediction to history.")

        return response_data

    except HTTPException as e:
        # Re-raise HTTPExceptions (like validation errors)
        raise e
    except Exception as e:
        # Catch any other unexpected errors during prediction
        print(f"[ERROR] Unhandled error in /predict_crop: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_disease")
def post_predict_disease(
    image: UploadFile = File(...),
    latitude: Optional[float] = None,
    longitude: Optional[float] = None
):
    """
    Endpoint for optional image-based disease classification.
    """

    # We'll check for the disease model (which we haven't trained yet)
    if model_artifacts and "disease_model" not in model_artifacts:
        return {
            "message": "Endpoint is active, but no disease *image* model is loaded.",
            "info": "The server is currently providing rule-based disease alerts only.",
            "image_filename": image.filename,
            "coordinates": {"lat": latitude, "lon": longitude}
        }
    elif model_artifacts is None:
         raise HTTPException(
            status_code=503,
            detail="Models are not loaded. Cannot process disease."
        )

    # --- Placeholder for future disease model logic ---
    # 1. (Future) Save image to a temporary path
    # 2. (Future) Load and preprocess image
    # 3. (Future) `disease_model.predict(image)`
    # 4. (Future) Decode predictions

    return {
        "warning": "Disease image model not yet implemented.",
        "disease_class": "Placeholder Disease",
        "confidence": 0.95,
        "suggested_treatment": "This is a placeholder treatment."
    }

@app.get("/history")
def get_history():
    """Retrieves the JSON file of all past predictions."""
    return read_prediction_history()