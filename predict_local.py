# predict_local.py
"""
Offline-only CLI to test saved models.

Loads the trained crop model, scaler, and encoders to make a
single prediction from command-line input.

Usage:
(Ensure 'venv' is active)

python predict_local.py --input-row "34,12,45,6.5,2.4,28.6,73.5"

(The order is N,P,K,pH,rainfall,temperature,humidity)
Note: The order in train_crop_model.py was:
['N', 'P', 'K', 'soil_ph', 'annual_rainfall_mm', 'avg_temp_c', 'soil_moisture_pct']
We must match this order.
Let's check the ESP32 example:
N, P, K, pH, temperature, humidity, rainfall
This is a DIFFERENT order.

Let's stick to the order from the training script:
FEATURES = ['N', 'P', 'K', 'soil_ph', 'annual_rainfall_mm', 'avg_temp_c', 'soil_moisture_pct']
So the example should be:
N, P, K, pH, rainfall, temperature, humidity
"34,12,45,6.5,2.4,28.6,73.5"
"""
import joblib
import json
import numpy as np
import argparse
import os
import sys

# Import the rule-based alerts from the server file
try:
    from server.utils import get_disease_alerts
except ImportError:
    print("[ERROR] Could not import from server.utils.")
    print("[ERROR] Make sure 'server/utils.py' exists.")
    sys.exit(1)


# --- Configuration ---
MODELS_DIR = "models"
CROP_MODEL_PATH = os.path.join(MODELS_DIR, "crop_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
FERTILIZER_MAP_PATH = os.path.join(MODELS_DIR, "fertilizer_map.json")

# This is the expected order of features for the model
EXPECTED_FEATURES = [
    'N', 'P', 'K', 'soil_ph', 'annual_rainfall_mm', 'avg_temp_c', 'soil_moisture_pct'
]

def load_artifacts():
    """Loads all model artifacts from disk."""
    print("[INFO] Loading model artifacts...")
    try:
        model = joblib.load(CROP_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        
        with open(FERTILIZER_MAP_PATH, 'r') as f:
            fertilizer_map = json.load(f)
            
        print("[INFO] All artifacts loaded successfully.")
        return model, scaler, label_encoder, fertilizer_map
        
    except FileNotFoundError as e:
        print(f"[ERROR] File not found: {e.filename}")
        print("[ERROR] Please run 'python train_crop_model.py' first.")
        return None
    except Exception as e:
        print(f"[ERROR] An error occurred while loading artifacts: {e}")
        return None

def predict(input_data, model, scaler, label_encoder):
    """
    Makes a single prediction.
    input_data: A 1D numpy array of raw feature values.
    """
    # Reshape data to 2D array (1 sample, 7 features)
    input_2d = input_data.reshape(1, -1)
    
    # Scale the features
    input_scaled = scaler.transform(input_2d)
    
    # Make prediction (returns a numeric label)
    pred_numeric = model.predict(input_scaled)
    
    # Get prediction probabilities
    pred_proba = model.predict_proba(input_scaled)
    
    # Get the confidence of the top prediction
    confidence = np.max(pred_proba)
    
    # Decode numeric label back to string (crop name)
    crop_name = label_encoder.inverse_transform(pred_numeric)[0]
    
    return crop_name, confidence

def get_fertilizer_rec(crop_name, fertilizer_map):
    """Gets fertilizer recommendation from the loaded map."""
    # Normalize key for lookup
    lookup_key = crop_name.lower()
    
    # Use .get() to safely find the recommendation
    recommendation = fertilizer_map.get(lookup_key)
    
    if recommendation:
        return recommendation
    else:
        # Fallback if crop (somehow) isn't in the map
        return "No specific fertilizer recommendation found in map."

def main():
    parser = argparse.ArgumentParser(description="Offline Crop Prediction CLI")
    parser.add_argument(
        "--input-row",
        type=str,
        required=True,
        help=f"A comma-separated string of input features in this exact order: {','.join(EXPECTED_FEATURES)}"
    )
    args = parser.parse_args()
    
    # --- 1. Load Artifacts ---
    artifacts = load_artifacts()
    if artifacts is None:
        sys.exit(1)
        
    model, scaler, label_encoder, fertilizer_map = artifacts
    
    # --- 2. Parse Input ---
    try:
        # Split string into a list of strings
        input_list = args.input_row.split(',')
        
        # Convert to a numpy array of floats
        input_data = np.array([float(val) for val in input_list])
        
        # Validate length
        if len(input_data) != len(EXPECTED_FEATURES):
            print(f"[ERROR] Input data has {len(input_data)} values, but model expects {len(EXPECTED_FEATURES)}.")
            print(f"[INFO] Expected order: {EXPECTED_FEATURES}")
            sys.exit(1)
            
    except ValueError:
        print(f"[ERROR] Invalid input row. Could not convert all values to numbers.")
        print(f"[INFO] Please provide a comma-separated string of numbers only.")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] Error parsing input: {e}")
        sys.exit(1)

    # --- 3. Make Predictions ---
    print(f"\n[INFO] Making prediction for input: {input_data}")
    
    try:
        crop_name, confidence = predict(input_data, model, scaler, label_encoder)
        
        # --- 4. Get Recommendations ---
        fertilizer_rec = get_fertilizer_rec(crop_name, fertilizer_map)
        disease_alerts = get_disease_alerts(crop_name) # From server/utils.py
        
        # --- 5. Display Results ---
        print("\n--- 🚀 Prediction Results ---")
        print(f"  Predicted Crop:  {crop_name}")
        print(f"  Confidence:      {confidence:.2%}")
        print("\n  Recommendations:")
        print(f"   Fertilizer:     {fertilizer_rec}")
        print(f"   Disease Alerts: {', '.join(disease_alerts)}")
        print("------------------------------\n")
        
    except Exception as e:
        print(f"[ERROR] An error occurred during prediction: {e}")
        # This can happen if the model expects a different number of features
        if "X has" in str(e) and "features" in str(e):
             print("[HINT] This often means the input data length doesn't match the model's training data.")
        sys.exit(1)


if __name__ == "__main__":
    main()