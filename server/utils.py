"""
Utility functions for the server.
Includes:
- Rule-based disease alerts
- Fertilizer recommendation logic (JSON priority over ML)
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
    "default": ["No specific alerts available."],
}


def get_disease_alerts(crop_name: str) -> List[str]:
  """
  Returns a list of disease alerts for a crop.
  Always returns a list, never None.
  """
  return DISEASE_ALERTS.get(crop_name.lower(), DISEASE_ALERTS["default"])


# --- Fertilizer Recommendation Logic ---
class FertilizerRecommender:
    """
    Handles fertilizer recommendation using priority:

    Priority Order:
        1️⃣ JSON mapping (fertilizer_map.json)
        2️⃣ Trained ML model (fertilizer_model.joblib)
        3️⃣ Generic fallback text
    """

    def __init__(self):
        self.fertilizer_map = self._load_fertilizer_map()
        self.fertilizer_model_artifacts = self._load_fertilizer_model()

        if self.fertilizer_map:
            print("[INFO] Fertilizer Recommender: Using JSON lookup map (Priority 1).")
        elif self.fertilizer_model_artifacts:
            print("[INFO] Fertilizer Recommender: Using trained ML model (Priority 2).")
        else:
            print("[WARN] Fertilizer Recommender: Using fallback rules (Priority 3).")

    # ----------------- LOADERS -----------------
    def _load_fertilizer_map(self) -> Optional[Dict[str, str]]:
        try:
            with open(FERTILIZER_MAP_PATH, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"[INFO] Fertilizer map not found: {FERTILIZER_MAP_PATH}")
            return None
        except Exception as e:
            print(f"[ERROR] Error reading fertilizer map: {e}")
            return None

    def _load_fertilizer_model(self) -> Optional[Dict[str, Any]]:
        try:
            artifacts = joblib.load(FERTILIZER_MODEL_PATH)
            if all(k in artifacts for k in ("model", "scaler", "label_encoder")):
                return artifacts
            print("[ERROR] Fertilizer model missing components.")
            return None
        except FileNotFoundError:
            print(f"[INFO] Fertilizer ML model not found: {FERTILIZER_MODEL_PATH}")
            return None
        except Exception as e:
            print(f"[ERROR] Loading fertilizer ML model failed: {e}")
            return None

    # ----------------- PREDICTOR -----------------
    def get_recommendation(self, crop_name: str, features: Optional[np.ndarray] = None) -> str:
        """
        Recommendation priority:

        1) fertilizer_map.json (dataset-based per crop)
        2) ML model (if available + features)
        3) Generic fallback
        """
        lookup = (crop_name or "").lower()

        # 1️⃣ JSON Lookup (dataset) – MOST IMPORTANT
        if self.fertilizer_map:
            rec = self.fertilizer_map.get(lookup)
            if rec:
                return rec

        # 2️⃣ ML Model Prediction (optional)
        if self.fertilizer_model_artifacts and features is not None:
            try:
                model = self.fertilizer_model_artifacts["model"]
                scaler = self.fertilizer_model_artifacts["scaler"]
                le = self.fertilizer_model_artifacts["label_encoder"]

                features_2d = features.reshape(1, -1)
                scaled = scaler.transform(features_2d)
                pred_numeric = model.predict(scaled)
                return le.inverse_transform(pred_numeric)[0]
            except Exception as e:
                print(f"[ERROR] Fertilizer ML model failed → {e}")

        # 3️⃣ Default fallback
        return f"Normal fertilizers needed for {crop_name}."