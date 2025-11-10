# Offline-First ML Prediction Project

This project is a complete, offline-first system for running ML models on a laptop to provide agricultural recommendations. It is designed to receive sensor data (e.g., from an ESP32) and run models locally without any internet connection.

## Features

- **100% Offline**: No runtime internet access, API calls, or remote resources.
- **Crop Prediction**: A `RandomForestClassifier` trained on N, P, K, pH, temperature, humidity, and rainfall.
- **Fertilizer Recommendation**: A rule-based lookup generated from a local CSV.
- **Disease Alerts**: A simple, rule-based if-then mapping based on the predicted crop.
- **Web Server**: A local FastAPI server to host the models as an API.
- **React Frontend**: A user-friendly web interface with a form, 3D results, and a history page with maps.
- **Data Ingest**: A Python script to listen for data from an ESP32 over a serial port.

---

## How to Run This Project (Offline)

Follow these steps to set up and run the entire application from scratch.

### Step 1: Set Up the Python Backend

First, set up the Python environment and train your models.

1.  **Open a terminal** in the project's root folder (`offline_ml_project`).
2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    ```
3.  **Activate the environment**:
    ```bash
    source venv/bin/activate
    ```
4.  **Install Python libraries** (needs internet one time):
    ```bash
    pip install -r requirements.txt
    ```
5.  **Train the ML models**:
    This script reads `data/arginode_corrected_fertilizers.csv`, trains the crop model, and generates the fertilizer lookup map.
    ```bash
    python train_crop_model.py
    ```
    Your models are now saved in the `models/` directory.

### Step 2: Run the Backend Server

The backend server hosts your models at `http://127.0.0.1:8000`.

1.  **In your first terminal** (with `venv` active), run:
    ```bash
    uvicorn server.main:app --reload
    ```
2.  Leave this terminal running. You can check if it's working by opening `http://127.0.0.1:8000` in your browser.

### Step 3: Run the Frontend Application

The frontend provides the user interface at `http://localhost:3000`.

1.  **Open a _second_ terminal**.
2.  **Navigate to the `frontend` folder**:
    ```bash
    cd frontend
    ```
3.  **Create the `.env` fix file** (this fixes the `npm start` error):
    ```bash
    echo "DANGEROUSLY_DISABLE_HOST_CHECK=true" > .env
    ```
4.  **Install Node.js libraries** (needs internet one time):
    ```bash
    npm install
    ```
5.  **Start the React app**:
    ```bash
    npm start
    ```
6.  This will automatically open `http://localhost:3000` in your browser.

---

## How to Use the System

### Option 1: Use the Web Interface (Recommended)

1.  Go to `http://localhost:3000` in your browser.
2.  On the **Start** tab, fill in the sensor values (N, P, K, etc.).
3.  Click "Predict Crop". The 3D result will appear.
4.  Click the **History** tab to see your saved prediction, complete with a map.

### Option 2: Use the ESP32 Ingest Script

This script simulates a real ESP32 device sending data to your server.

1.  **Open a _third_ terminal** (or use your second terminal after stopping the `npm start` process).
2.  **Activate the environment**: `source venv/bin/activate`
3.  **Run the simulator**:

    ```bash
    python scripts/ingest_from_esp32.py --simulate
    ```

    This will send one pre-configured prediction (with location) to your server. You can see it appear on the **History** page.

4.  **To use a real ESP32**:
    Plug in your ESP32 and find its port name (e.g., `/dev/tty.usbserial-XXXX`). Then, run:
    ```bash
    python scripts/ingest_from_esp32.py --serial-port /dev/tty.usbserial-XXXX
    ```

# tools/visualize_prediction.py

"""
Generates a static PNG "preview" of a prediction.
Useful for quick, offline checks.
"""
import matplotlib
import matplotlib.pyplot as plt
import argparse

# Use a non-interactive backend for saving files

matplotlib.use('Agg')

def create_visualization(crop, confidence, fertilizer, diseases, output_file):
"""Creates and saves a simple matplotlib visualization."""

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor('#f4f7f6')

    # Title
    ax.text(0.5, 0.9, "Prediction Result", ha='center', va='center',
            fontsize=20, fontweight='bold', color='#333')

    # Crop
    ax.text(0.1, 0.7, "Predicted Crop:", ha='left', va='center',
            fontsize=14, color='#555')
    ax.text(0.5, 0.7, crop, ha='left', va='center',
            fontsize=16, fontweight='bold', color='#007aff')

    # Confidence
    ax.text(0.1, 0.6, "Confidence:", ha='left', va='center',
            fontsize=14, color='#555')
    ax.text(0.5, 0.6, f"{confidence:.1%}", ha='left', va='center',
            fontsize=16, color='#333')

    # Fertilizer
    ax.text(0.1, 0.45, "Fertilizer:", ha='left', va='center',
            fontsize=14, color='#555')
    ax.text(0.5, 0.45, fertilizer, ha='left', va='center',
            fontsize=12, color='#333', wrap=True)

    # Diseases
    ax.text(0.1, 0.3, "Disease Alerts:", ha='left', va='center',
            fontsize=14, color='#555')
    ax.text(0.5, 0.3, "\n".join(diseases), ha='left', va='top',
            fontsize=12, color='#b33e3a')

    ax.axis('off') # Hide axes
    plt.tight_layout()

    try:
        plt.savefig(output_file, dpi=100, bbox_inches='tight')
        print(f"[SUCCESS] Visualization saved to: {output_file}")
    except Exception as e:
        print(f"[ERROR] Failed to save visualization: {e}")

if **name** == "**main**":
parser = argparse.ArgumentParser(description="Create a prediction visualization.")
parser.add_argument("--crop", type=str, required=True, help="Predicted crop name")
parser.add_argument("--confidence", type=float, required=True, help="Prediction confidence (e.g., 0.85)")
parser.add_argument("--fertilizer", type=str, required=True, help="Fertilizer recommendation")
parser.add_argument("--diseases", type=str, nargs='+', required=True, help="List of disease alerts")
parser.add_argument("--output", type=str, default="prediction_preview.png", help="Output PNG file")

    args = parser.parse_args()

    create_visualization(args.crop, args.confidence, args.fertilizer, args.diseases, args.output)

# docker-compose.offline.yml

# (Optional) For running the server and a local DB in Docker.

# This setup is 100% offline _after_ the initial image build.

version: '3.8'

services:

# The FastAPI server

server:
build:
context: .
dockerfile: Dockerfile # You would need to create this Dockerfile
ports: - "8000:8000"
volumes: - ./models:/app/models # Mount local models - ./db:/app/db # Mount local JSON db # Set to "host" to allow connecting to services on the host machine # Note: This is not ideal for production but useful for offline local dev.
network_mode: "host"
environment: - OFFLINE_MODE=True

# --- Optional Local Databases ---

# Uncomment the one you want to use.

# Local MongoDB (if you wanted to use it instead of JSON)

# mongo:

# image: mongo:5.0

# ports:

# - "27017:27017"

# volumes:

# - mongo_data:/data/db

# restart: unless-stopped

# Local PostgreSQL (if you wanted to use it)

# postgres:

# image: postgres:14

# ports:

# - "5432:5432"

# environment:

# - POSTGRES_USER=myuser

# - POSTGRES_PASSWORD=mypassword

# - POSTGRES_DB=mydb

# volumes:

# - postgres_data:/var/lib/postgresql/data

# restart: unless-stopped

volumes:
mongo_data:
postgres_data:

# docker-compose.offline.yml

# (Optional) For running the server and a local DB in Docker.

# This setup is 100% offline _after_ the initial image build.

version: '3.8'

services:

# The FastAPI server

server:
build:
context: .
dockerfile: Dockerfile # You would need to create this Dockerfile
ports: - "8000:8000"
volumes: - ./models:/app/models # Mount local models - ./db:/app/db # Mount local JSON db # Set to "host" to allow connecting to services on the host machine # Note: This is not ideal for production but useful for offline local dev.
network_mode: "host"
environment: - OFFLINE_MODE=True

# --- Optional Local Databases ---

# Uncomment the one you want to use.

# Local MongoDB (if you wanted to use it instead of JSON)

# mongo:

# image: mongo:5.0

# ports:

# - "27017:27017"

# volumes:

# - mongo_data:/data/db

# restart: unless-stopped

# Local PostgreSQL (if you wanted to use it)

# postgres:

# image: postgres:14

# ports:

# - "5432:5432"

# environment:

# - POSTGRES_USER=myuser

# - POSTGRES_PASSWORD=mypassword

# - POSTGRES_DB=mydb

# volumes:

# - postgres_data:/var/lib/postgresql/data

# restart: unless-stopped

volumes:
mongo_data:
postgres_data:

# generate_synthetic_disease_images.py

"""
(Optional) Generates a synthetic image dataset for offline testing
of the 'train_disease_model.py' script.
"""
import os
import numpy as np
from PIL import Image

# --- Configuration ---

OUTPUT_DIR = "data/synthetic_disease"
CLASSES = ["crop_healthy", "crop_disease_a", "crop_disease_b"]
IMAGES_PER_CLASS = 20 # Keep low for fast testing
IMG_SIZE = (224, 224)

def generate*image(class_index):
"""Generates a simple noisy image with a colored block.""" # Create a random noise background
img_array = np.random.rand(\_IMG_SIZE, 3) * 50 + 50

    # Define a color for each class
    colors = [
        [50, 200, 50],  # Healthy (Green)
        [200, 50, 50],  # Disease A (Red)
        [50, 50, 200]   # Disease B (Blue)
    ]
    color = colors[class_index]

    # Add a colored block
    x_start, y_start = np.random.randint(50, 150, 2)
    img_array[x_start:x_start+50, y_start:y_start+50, :] = color

    # Add some "disease" spots
    if class_index > 0:
        for _ in range(30):
            x, y = np.random.randint(0, 224, 2)
            img_array[x:x+3, y:y+3, :] = [200, 200, 50] # Yellow spots

    img_array = np.clip(img_array, 0, 255).astype(np.uint8)
    return Image.fromarray(img_array)

def main():
print(f"[INFO] Generating synthetic image data in: {OUTPUT_DIR}")

    for i, class_name in enumerate(CLASSES):
        class_path = os.path.join(OUTPUT_DIR, class_name)
        os.makedirs(class_path, exist_ok=True)
        print(f"-> Creating class: {class_name}")

        for j in range(IMAGES_PER_CLASS):
            img = generate_image(i)
            img_path = os.path.join(class_path, f"synth_{j:03d}.png")
            img.save(img_path)

    print(f"[SUCCESS] Generated {IMAGES_PER_CLASS * len(CLASSES)} total images.")

if **name** == "**main**":
main()

# train_disease_model.py

"""
(Optional) Offline-only training script for the disease _image_ classifier.
"""
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import argparse
import os

# --- Configuration ---

DATA_DIR = "data/synthetic_disease"
MODEL_SAVE_PATH = "models/disease_model"
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
NUM_EPOCHS = 5 # Keep low for testing

def build_model(num_classes, local_weights_path=None):
"""Builds a MobileNetV2 model, optionally loading local weights."""
print("[INFO] Building MobileNetV2 model...")

    # Use weights=None for 100% offline training by default
    weights_to_load = None

    if local_weights_path:
        print(f"[INFO] Attempting to load local weights from: {local_weights_path}")
        # This allows user to provide their own offline weights file
        weights_to_load = local_weights_path
    else:
        print("[INFO] No local weights provided. Training from scratch (weights=None).")

    base_model = MobileNetV2(
        weights=weights_to_load,
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base layers if we loaded weights
    if local_weights_path:
        for layer in base_model.layers:
            layer.trainable = False

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("[INFO] Model built successfully.")
    return model

def main():
parser = argparse.ArgumentParser(description="Offline Disease Model Trainer")
parser.add_argument(
"--data",
type=str,
default=DATA_DIR,
help="Path to image data directory"
)
parser.add_argument(
"--local-weights",
type=str,
default=None,
help="Path to local .h5 weights file for transfer learning"
)
args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"[ERROR] Data directory not found: {args.data}")
        print("[HINT] Run 'python generate_synthetic_disease_images.py' first.")
        return

    # --- Data Augmentation & Generators ---
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2 # Use 20% of data for validation
    )

    # Train Generator
    train_generator = train_datagen.flow_from_directory(
        args.data,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Validation Generator (no augmentation, just rescaling)
    validation_generator = train_datagen.flow_from_directory(
        args.data,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = len(train_generator.class_indices)
    print(f"[INFO] Found {num_classes} classes: {list(train_generator.class_indices.keys())}")

    # --- Build Model ---
    model = build_model(num_classes, args.local_weights)

    # --- Callbacks ---
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(MODEL_SAVE_PATH, "best_model.h5"),
        save_best_only=True,
        monitor='val_loss',
        verbose=1
    )

    print("[INFO] Starting model training...")
    model.fit(
        train_generator,
        epochs=NUM_EPOCHS,
        validation_data=validation_generator,
        callbacks=[early_stopping, model_checkpoint]
    )

    print(f"[INFO] Training complete. Saving final model to {MODEL_SAVE_PATH}")
    # Save as a SavedModel directory, which is the standard
    model.save(MODEL_SAVE_PATH)
    print("[SUCCESS] Disease model saved.")

if **name** == "**main**":
main()
