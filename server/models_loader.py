# server/models_loader.py
"""
Loads all ML models and artifacts into memory at server startup.

This centralizes model loading and provides a single, shared
instance of the models for the FastAPI app.
"""

import os
import joblib
from typing import Dict, Any, Optional

# Import our custom utility classes and paths
from server.utils import (
    FertilizerRecommender,
    CROP_MODEL_PATH,
    SCALER_PATH,
    LABEL_ENCODER_PATH
)

def load_all_models() -> Optional[Dict[str, Any]]:
    """
    Tries to load all required models from disk.
    Returns a dictionary of artifacts if successful, else None.
    """
    print("[INFO] --- Loading All ML Models ---")
    artifacts = {}
    
    try:
        # 1. Load Crop Prediction Model
        print(f"[INFO] Loading crop model from: {CROP_MODEL_PATH}")
        artifacts["crop_model"] = joblib.load(CROP_MODEL_PATH)
        print("[INFO] Crop model loaded.")
        
        # 2. Load Feature Scaler
        print(f"[INFO] Loading scaler from: {SCALER_PATH}")
        artifacts["scaler"] = joblib.load(SCALER_PATH)
        print("[INFO] Scaler loaded.")
        
        # 3. Load Crop Label Encoder
        print(f"[INFO] Loading label encoder from: {LABEL_ENCODER_PATH}")
        artifacts["label_encoder"] = joblib.load(LABEL_ENCODER_PATH)
        print("[INFO] Label encoder loaded.")
        
        # 4. Initialize Fertilizer Recommender
        # This will load its own artifacts (map or model)
        print("[INFO] Initializing fertilizer recommender...")
        artifacts["fertilizer_recommender"] = FertilizerRecommender()
        
        print("[SUCCESS] --- All models and artifacts loaded successfully! ---")
        return artifacts

    except FileNotFoundError as e:
        print(f"\n[CRITICAL ERROR] Model file not found: {e.filename}")
        print("[CRITICAL ERROR] Please run 'python train_crop_model.py' to generate models.")
        return None
    except Exception as e:
        print(f"\n[CRITICAL ERROR] A general error occurred during model loading: {e}")
        return None

# --- Load models on startup ---
# This code runs ONCE when this file is imported by the server.
# The `model_artifacts` dictionary will be imported by main.py.
model_artifacts = load_all_models()