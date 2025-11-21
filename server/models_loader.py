# server/models_loader.py

"""
Central place to load:
  - Crop prediction model
  - Scaler
  - Label encoder
  - Fertilizer recommender (NEW: CSV similarity-based)

Exposes a single dict:
  model_artifacts = {
      "crop_model": ...,
      "scaler": ...,
      "label_encoder": ...,
      "fertilizer_recommender": <CSVFertilizerRecommender>,
  }
"""

import os
import joblib

from .fertilizer_recommender_csv import CSVFertilizerRecommender


def _resolve_path(*parts: str) -> str:
    """Helper to resolve paths relative to project root."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_dir, *parts)


def load_models():
    print("[INFO] --- Loading All ML Models ---")

    artifacts = {}

    # -------- Crop model --------
    crop_model_path = _resolve_path("models", "crop_model.joblib")
    print(f"[INFO] Loading crop model from: {os.path.relpath(crop_model_path)}")
    if not os.path.exists(crop_model_path):
        raise FileNotFoundError(f"Crop model not found at {crop_model_path}")
    artifacts["crop_model"] = joblib.load(crop_model_path)
    print("[INFO] Crop model loaded.")

    # -------- Scaler --------
    scaler_path = _resolve_path("models", "scaler.joblib")
    print(f"[INFO] Loading scaler from: {os.path.relpath(scaler_path)}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found at {scaler_path}")
    artifacts["scaler"] = joblib.load(scaler_path)
    print("[INFO] Scaler loaded.")

    # -------- Label encoder --------
    le_path = _resolve_path("models", "label_encoder.joblib")
    print(f"[INFO] Loading label encoder from: {os.path.relpath(le_path)}")
    if not os.path.exists(le_path):
        raise FileNotFoundError(f"Label encoder not found at {le_path}")
    artifacts["label_encoder"] = joblib.load(le_path)
    print("[INFO] Label encoder loaded.")

    # -------- Fertilizer recommender (CSV-based) --------
    print("[INFO] Initializing fertilizer recommender (CSV similarity)...")

    # Adjust this path if your CSV lives somewhere else:
    fert_csv_path = _resolve_path("data", "arginode_corrected_fertilizers.csv")

    fert_recommender = CSVFertilizerRecommender(fert_csv_path)
    artifacts["fertilizer_recommender"] = fert_recommender

    print("[INFO] Fertilizer Recommender: Using CSV nearest-neighbour over sensor features.")
    print("[SUCCESS] --- All models and artifacts loaded successfully! ---")

    return artifacts


# This is what main.py imports
try:
    model_artifacts = load_models()
except Exception as e:
    print(f"[CRITICAL] Failed to load models: {e}")
    model_artifacts = None