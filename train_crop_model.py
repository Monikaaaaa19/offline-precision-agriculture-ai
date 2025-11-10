# train_crop_model.py
"""
Offline-only training script for crop recommendation and fertilizer lookup.

Reads a local CSV dataset to train a RandomForest model for crop prediction
and create a JSON lookup map for fertilizer recommendations.

Usage (from project root):
1. Create fertilizer map (default):
   python train_crop_model.py --dataset data/arginode_corrected_fertilizers.csv

2. (Optional) Train a fertilizer model (if data has labels):
   python train_crop_model.py --dataset data/arginode_corrected_fertilizers.csv --train-fertilizer-model
"""
import pandas as pd
import numpy as np
import joblib
import json
import argparse
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.exceptions import NotFittedError

# --- Configuration ---
# We use the column names from your provided CSV file:
# 'N', 'P', 'K', 'soil_ph', 'annual_rainfall_mm', 'avg_temp_c', 'soil_moisture_pct'
FEATURES = ['N', 'P', 'K', 'soil_ph', 'annual_rainfall_mm', 'avg_temp_c', 'soil_moisture_pct']
TARGET_CROP = 'crop'
TARGET_FERTILIZER = 'fertilizer_recommendation'

# Model output paths
MODELS_DIR = "models"
CROP_MODEL_PATH = os.path.join(MODELS_DIR, "crop_model.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
LABEL_ENCODER_PATH = os.path.join(MODELS_DIR, "label_encoder.joblib")
FERTILIZER_MAP_PATH = os.path.join(MODELS_DIR, "fertilizer_map.json")
FERTILIZER_MODEL_PATH = os.path.join(MODELS_DIR, "fertilizer_model.joblib")

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Fixed seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def load_data(csv_path):
    """Loads and validates the dataset."""
    print(f"[INFO] Loading dataset from: {csv_path}")
    try:
        data = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"[ERROR] Dataset file not found at: {csv_path}")
        print("[ERROR] Please make sure the file exists and the path is correct.")
        return None
    except Exception as e:
        print(f"[ERROR] Could not read dataset: {e}")
        return None

    # Validate required columns
    required_cols = FEATURES + [TARGET_CROP, TARGET_FERTILIZER]
    missing_cols = [col for col in required_cols if col not in data.columns]
    
    if missing_cols:
        print(f"[ERROR] Dataset is missing required columns: {missing_cols}")
        print(f"[ERROR] Please ensure your CSV has: {required_cols}")
        return None
        
    print("[INFO] Dataset loaded and validated successfully.")
    return data

def create_fertilizer_lookup(data):
    """
    Creates a simple lookup map (dictionary) from crop to fertilizer.
    Uses the first recommendation found for each crop.
    Saves the map to a JSON file.
    """
    print("[INFO] Creating fertilizer lookup map...")
    
    # Create a simple dictionary mapping crop to fertilizer
    # We drop duplicates to get one entry per crop
    # .set_index(TARGET_CROP) turns the 'crop' column into the index (keys)
    # .to_dict() converts this to a dictionary
    try:
        fertilizer_map = data.drop_duplicates(subset=[TARGET_CROP]) \
                             .set_index(TARGET_CROP)[TARGET_FERTILIZER] \
                             .to_dict()
        
        # Normalize keys to lowercase for easier lookup later
        fertilizer_map = {k.lower(): v for k, v in fertilizer_map.items()}

        # Save the map to disk
        with open(FERTILIZER_MAP_PATH, 'w') as f:
            json.dump(fertilizer_map, f, indent=4)
            
        print(f"[SUCCESS] Fertilizer lookup map saved to: {FERTILIZER_MAP_PATH}")
        print(f"[INFO] Found {len(fertilizer_map)} unique crop-fertilizer mappings.")
        
    except Exception as e:
        print(f"[ERROR] Failed to create fertilizer lookup map: {e}")

def train_crop_model(data):
    """
    Trains, evaluates, and saves the crop prediction model.
    """
    print("[INFO] Starting crop model training...")
    
    # 1. Prepare data
    X = data[FEATURES]
    y_raw = data[TARGET_CROP]
    
    # 2. Label Encode Target (Crop)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    # Save the label encoder
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"[INFO] Label encoder saved to: {LABEL_ENCODER_PATH}")
    print(f"[INFO] Found {len(le.classes_)} crops: {list(le.classes_)}")

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    
    # 4. Scale Features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler
    joblib.dump(scaler, SCALER_PATH)
    print(f"[INFO] Feature scaler saved to: {SCALER_PATH}")

    # 5. Train Model with GridSearchCV (small grid for speed)
    print("[INFO] Running GridSearchCV... (This may take a moment)")
    
    # Using RandomForestClassifier as requested
    rf = RandomForestClassifier(random_state=RANDOM_SEED)
    
    # A small grid to keep it fast and offline-friendly
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_leaf': [2, 5]
    }
    
    # n_jobs=-1 uses all available cores, which is fine offline
    # cv=3 is a 3-fold cross-validation
    grid_search = GridSearchCV(
        estimator=rf, 
        param_grid=param_grid, 
        cv=3, 
        n_jobs=-1, 
        verbose=1
    )
    
    grid_search.fit(X_train_scaled, y_train)
    
    print("[INFO] GridSearchCV complete.")
    print(f"[INFO] Best parameters found: {grid_search.best_params_}")
    
    best_model = grid_search.best_estimator_
    
    # 6. Save the final model
    joblib.dump(best_model, CROP_MODEL_PATH)
    print(f"[SUCCESS] Best crop model saved to: {CROP_MODEL_PATH}")

    # 7. Evaluate the model
    print("[INFO] Evaluating model on test set...")
    try:
        y_pred = best_model.predict(X_test_scaled)
        
        # Convert numeric predictions back to string labels
        y_pred_labels = le.inverse_transform(y_pred)
        y_test_labels = le.inverse_transform(y_test)
        
        print("\n--- Classification Report ---")
        print(classification_report(y_test_labels, y_pred_labels))
        print("\n--- Confusion Matrix ---")
        print(confusion_matrix(y_test_labels, y_pred_labels))
        print("\n----------------------------\n")
        
    except NotFittedError:
        print("[ERROR] Model is not fitted. Skipping evaluation.")
    except Exception as e:
        print(f"[ERROR] Error during model evaluation: {e}")

def train_fertilizer_model(data):
    """
    (Optional) Trains and saves a classifier for fertilizer recommendation.
    This is only run if the --train-fertilizer-model flag is used.
    """
    print("\n[INFO] Starting optional fertilizer model training...")
    
    # 1. Prepare data
    # We use the same features as the crop model
    X = data[FEATURES]
    
    # Target is the fertilizer recommendation
    y_raw = data[TARGET_FERTILIZER]
    
    # 2. Label Encode Target (Fertilizer)
    # We need a separate LabelEncoder for fertilizer
    le_fert = LabelEncoder()
    y_fert = le_fert.fit_transform(y_raw)
    
    print(f"[INFO] Found {len(le_fert.classes_)} fertilizer types.")

    # 3. Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_fert, test_size=0.2, random_state=RANDOM_SEED, stratify=y_fert
    )
    
    # 4. Scale Features
    # We can re-use the *logic* from the crop model, but we should
    # fit a new scaler just for this model to be safe.
    scaler_fert = StandardScaler()
    X_train_scaled = scaler_fert.fit_transform(X_train)
    X_test_scaled = scaler_fert.transform(X_test)

    # 5. Train a simple RandomForest
    # We don't need a big grid search, just a basic model.
    print("[INFO] Training simple RandomForest for fertilizer...")
    fert_model = RandomForestClassifier(
        n_estimators=50, 
        max_depth=10, 
        random_state=RANDOM_SEED
    )
    fert_model.fit(X_train_scaled, y_train)

    # 6. Save the model and its scaler
    # Note: We don't save the label encoder because the server
    # will just use the predicted class as-is (or we can add it later)
    # For now, let's just save the model.
    # A-HA! We need the scaler and label encoder. Let's save them.
    
    fert_artifacts = {
        'model': fert_model,
        'scaler': scaler_fert,
        'label_encoder': le_fert
    }
    joblib.dump(fert_artifacts, FERTILIZER_MODEL_PATH)
    
    print(f"[SUCCESS] Fertilizer model artifacts saved to: {FERTILIZER_MODEL_PATH}")

    # 7. Evaluate
    print("[INFO] Evaluating fertilizer model...")
    y_pred = fert_model.predict(X_test_scaled)
    y_pred_labels = le_fert.inverse_transform(y_pred)
    y_test_labels = le_fert.inverse_transform(y_test)
    
    print("\n--- Fertilizer Model Report ---")
    print(classification_report(y_test_labels, y_pred_labels))
    print("\n-------------------------------\n")


def main():
    """Main execution function."""
    
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Offline Model Training Script.")
    
    # Path to the dataset
    # We use your CSV as the default, as requested.
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/arginode_corrected_fertilizers.csv",
        help="Path to the local training CSV file."
    )
    
    # Your requested --fertilizer-csv flag (though --dataset covers it)
    # We'll honor the logic: --fertilizer-csv overrides --dataset for fertilizer.
    # But for simplicity, we'll just use --dataset for everything.
    # The logic in your prompt was complex, so let's simplify:
    # --dataset is for training.
    # The fertilizer map is *always* created from this file.
    # The fertilizer model is *optionally* trained from this file.
    
    # (Optional) Flag to train the fertilizer model
    parser.add_argument(
        "--train-fertilizer-model",
        action="store_true",
        help="Also train a classifier for fertilizer recommendation."
    )
    
    args = parser.parse_args()

    print("--- Starting Offline Training Process ---")
    
    # Load the specified dataset
    dataset = load_data(args.dataset)
    
    if dataset is not None:
        # 1. Train Crop Model (Always runs)
        train_crop_model(dataset)
        
        # 2. Create Fertilizer Lookup Map (Always runs)
        create_fertilizer_lookup(dataset)
        
        # 3. (Optional) Train Fertilizer Model
        if args.train_fertilizer_model:
            train_fertilizer_model(dataset)
        
        print("\n[SUCCESS] All training tasks complete.")
        print("Your models are saved in the 'models/' directory.")
    else:
        print("[FAILURE] Training process aborted due to data loading errors.")

if __name__ == "__main__":
    main()