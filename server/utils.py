# server/utils.py
"""
Utility functions for the server.
Includes:
- Rule-based disease alerts
- Fertilizer recommendation logic (NOW PRIORITIZES ML MODEL)
"""

import os
import json
import joblib
import numpy as np
from typing import Dict, Any, List, Optional

# --- Model & Artifact Paths ---
MODELS_DIR = "models"
FERTILIZER_MAP_PATH = os.path.join(MODELS_DIR, "fertilizer_map.json")
FERTILIZER_MODEL_PATH = os.path.join(MODELS_DIR, "fertilizer_model.joblib")
CROP_MODEL_PATH = os.path.join(MODELS_DIR, "crop_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")


# --- Disease Alert Mappings (Rule-Based) ---
DISEASE_ALERTS = {
    "banana": ["Panama Disease", "Sigatoka Leaf Spot"],
    "coconut": ["Bud Rot", "Leaf Blight"],
    "coffee": ["Coffee Leaf Rust", "Berry Disease"],
    "cotton": ["Bollworm", "Leaf Curl Virus"],
    "jute": ["Stem Rot", "Anthracnose"],
    "maize": ["Stalk Rot", "Maize Rust"],
    "mulberry": ["Powdery Mildew", "Leaf Spot"],
    "potato": ["Late Blight", "Common Scab"],
    "pulses": ["Powdery Mildew", "Rust", "Aphids"],
    "rice": ["Brown Spot", "Sheath Blight", "Rice Blast"],
    "sugarcane": ["Red Rot", "Sugarcane Smut"],
    "tea": ["Blister Blight", "Red Spider Mite"],
    "wheat": ["Wheat Rust", "Powdery Mildew"],
    "default": ["No specific alerts available."]
}

def get_disease_alerts(crop_name: str) -> list:
    # (This function is unchanged)
    lookup_key = crop_name.lower()
    return DISEASE_ALERTS.get(lookup_key, DISEASE_ALERTS["default"])


# --- Fertilizer Recommendation Logic ---

class FertilizerRecommender:
    """
    A helper class to load fertilizer artifacts once and provide recommendations.
    
    *** NEW PRIORITY ***
    1.  Trained ML model (models/fertilizer_model.joblib) (if it exists)
    2.  JSON lookup map (models/fertilizer_map.json) (as a fallback)
    3.  Default text fallback.
    """
    def __init__(self):
        # We load both, as we did before
        self.fertilizer_map = self._load_fertilizer_map()
        self.fertilizer_model_artifacts = self._load_fertilizer_model()
        
        if self.fertilizer_model_artifacts:
            print("[INFO] Fertilizer Recommender: Using trained ML model (Priority 1).")
        elif self.fertilizer_map:
            print("[INFO] Fertilizer Recommender: Using JSON lookup map (Priority 2).")
        else:
            print("[WARN] Fertilizer Recommender: No map or model found. Using text defaults.")

    def _load_fertilizer_map(self) -> Optional[Dict[str, str]]:
        # (This function is unchanged)
        try:
            with open(FERTILIZER_MAP_PATH, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[INFO] Fertilizer map not found at: {FERTILIZER_MAP_PATH}")
            return None
        except Exception as e:
            print(f"[ERROR] Error loading fertilizer map: {e}")
            return None

    def _load_fertilizer_model(self) -> Optional[Dict[str, Any]]:
        # (This function is unchanged)
        # This loads the .joblib file containing the model, scaler, and encoder
        try:
            artifacts = joblib.load(FERTILIZER_MODEL_PATH)
            if 'model' in artifacts and 'scaler' in artifacts and 'label_encoder' in artifacts:
                return artifacts
            else:
                print(f"[ERROR] Invalid fertilizer model artifacts in: {FERTILIZER_MODEL_PATH}")
                return None
        except FileNotFoundError:
            print(f"[INFO] Optional fertilizer model not found at: {FERTILIZER_MODEL_PATH}")
            return None
        except Exception as e:
            print(f"[ERROR] Error loading fertilizer model: {e}")
            return None

    def get_recommendation(self, crop_name: str, features: Optional[np.ndarray] = None) -> str:
        """
        Gets a fertilizer recommendation for a predicted crop.
        """
        lookup_key = crop_name.lower()
        
        # --- PRIORITY 1: ML Model Prediction ---
        # We MUST have the 'features' (N,P,K etc.) to use the model
        if self.fertilizer_model_artifacts and features is not None:
            try:
                model = self.fertilizer_model_artifacts['model']
                scaler = self.fertilizer_model_artifacts['scaler']
                le = self.fertilizer_model_artifacts['label_encoder']
                
                # Reshape features to 2D array (1 sample, 7 features)
                features_2d = features.reshape(1, -1)
                features_scaled = scaler.transform(features_2d)
                
                # Predict numeric label
                pred_numeric = model.predict(features_scaled)
                
                # Decode to text
                return le.inverse_transform(pred_numeric)[0]
                
            except Exception as e:
                print(f"[ERROR] Fertilizer model prediction failed: {e}")
                # Fall through to map lookup

        # --- PRIORITY 2: JSON Map Lookup (Fallback) ---
        if self.fertilizer_map:
            recommendation = self.fertilizer_map.get(lookup_key)
            if recommendation:
                return recommendation

        # --- PRIORITY 3: Default Fallback ---
        return f"Standard fertilizer application recommended for {crop_name}."