# server/utils.py
"""
Utility functions for the server.
Includes:
- Rule-based disease alerts
- Fertilizer recommendation logic
- (Future) Input validation helpers
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
# As requested, this is the exact if-then mapping for disease alerts
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
    """
    Finds disease alerts for a given crop name using the
    rule-based DISEASE_ALERTS dictionary.
    """
    # Normalize the crop name to lowercase to match the keys
    lookup_key = crop_name.lower()
    
    # Find the alerts, or return the 'default' list if not found
    return DISEASE_ALERTS.get(lookup_key, DISEASE_ALERTS["default"])


# --- Fertilizer Recommendation Logic ---

class FertilizerRecommender:
    """
    A helper class to load fertilizer artifacts once and provide recommendations.
    
    Priority:
    1.  JSON lookup map (models/fertilizer_map.json)
    2.  Trained ML model (models/fertilizer_model.joblib) (if it exists)
    3.  Default text fallback.
    """
    def __init__(self):
        self.fertilizer_map = self._load_fertilizer_map()
        self.fertilizer_model_artifacts = self._load_fertilizer_model()
        
        if self.fertilizer_map:
            print("[INFO] Fertilizer Recommender: Using JSON lookup map.")
        elif self.fertilizer_model_artifacts:
            print("[INFO] Fertilizer Recommender: Using trained ML model.")
        else:
            print("[WARN] Fertilizer Recommender: No map or model found. Using text defaults.")

    def _load_fertilizer_map(self) -> Optional[Dict[str, str]]:
        """Loads the JSON map created by train_crop_model.py."""
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
        """Loads the optional fertilizer model artifacts."""
        try:
            # The joblib file contains a dict: {'model', 'scaler', 'label_encoder'}
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
        
        # Priority 1: JSON Map Lookup
        if self.fertilizer_map:
            recommendation = self.fertilizer_map.get(lookup_key)
            if recommendation:
                return recommendation

        # Priority 2: ML Model Prediction
        # (This path is used if map lookup fails or map doesn't exist,
        #  AND the model was trained and loaded, AND features were provided)
        if self.fertilizer_model_artifacts and features is not None:
            try:
                model = self.fertilizer_model_artifacts['model']
                scaler = self.fertilizer_model_artifacts['scaler']
                le = self.fertilizer_model_artifacts['label_encoder']
                
                # Scale features (must be in 1D numpy array)
                features_2d = features.reshape(1, -1)
                features_scaled = scaler.transform(features_2d)
                
                # Predict numeric label
                pred_numeric = model.predict(features_scaled)
                
                # Decode to text
                return le.inverse_transform(pred_numeric)[0]
                
            except Exception as e:
                print(f"[ERROR] Fertilizer model prediction failed: {e}")
                # Fall through to default

        # Priority 3: Default Fallback
        return f"Standard fertilizer application recommended for {crop_name}."