# Offline-First Agricultural ML Server

This project is a complete, 100% offline-capable system that runs on a laptop to provide agricultural recommendations. It is designed to receive sensor data (e.g., from an ESP32) and run local ML models to generate predictions.

## Features

- **100% Offline-Capable:** All models, maps, and logic run locally. No internet is needed after setup.
- **Crop Prediction:** A `RandomForestClassifier` trained on 7 sensor features.
- **Smart Fertilizer Model:** An ML classifier that reads N, P, and K values to provide "smart" recommendations (e.g., "DAP") based on the patterns in your dataset.
- **Offline State Lookup:** The server **automatically detects the state name** (e.g., "Karnataka") from the (Lat, Lon) coordinates you provide.
- **Offline Map Generation:** The "History" tab generates a 100% offline map plot showing the entire country of India, with the specific state highlighted in orange and the prediction point marked in red.
- **React Frontend:** A user-friendly web UI that remembers your form data between tabs.
- **ESP32 Ingest:** A Python script that auto-detects a connected ESP32, listens for its JSON data, and posts it to the server.

---

## 1. Required Setup (One-Time Only)

Before running, you must install dependencies and download one critical map file.

### Step 1.1: Install Python & System Dependencies

1.  **Activate your environment**:
    ```bash
    source venv/bin/activate
    ```
2.  **Install Python Libraries**:
    ```bash
    pip install -r requirements.txt
    pip install geopandas fiona pyproj pyserial requests
    ```
    _(Note: `geopandas` and its dependencies are needed for the state lookup and offline map.)_

### Step 1.2: Install Frontend Dependencies

1.  **Navigate to the frontend**:
    ```bash
    cd frontend
    ```
2.  **Install Node libraries**:
    ```bash
    npm install
    ```
3.  **Go back to the root folder**:
    ```bash
    cd ..
    ```

### Step 1.3: Download the State Boundary File (CRITICAL)

The server **cannot** find the state name or draw the map without this file.

1.  **Download the GeoJSON file** and save it in your `data` folder.
    ```bash
    curl -L -o data/india_states.geojson "[https://github.com/Subhash9325/GeoJson-Data-of-Indian-States/raw/master/Indian_States](https://github.com/Subhash9325/GeoJson-Data-of-Indian-States/raw/master/Indian_States)"
    ```

---

## 2. How to Run The Project

You must run **two** servers in **two separate terminals**.

### Terminal 1: Run the Backend (FastAPI) Server

1.  **Activate the environment**:
    ```bash
    source venv/bin/activate
    ```
2.  **Train the Models**: You must run this command to create the "smart" fertilizer model.
    ```bash
    python train_crop_model.py --train-fertilizer-model
    ```
3.  **Run the Server**:
    ```bash
    uvicorn server.main:app --reload
    ```
    _Wait until you see `[INFO] Successfully loaded India state boundaries...` and `...Application startup complete.`_

### Terminal 2: Run the Frontend (React) App

1.  **Navigate to the frontend**:
    ```bash
    cd frontend
    ```
2.  **Run the App**:
    ```bash
    npm start
    ```
    _This will automatically open `http://localhost:3000` in your browser._

---

## 3. How to Test (With ESP32)

You can run a third terminal to simulate or connect to your ESP32.

### Terminal 3: Run the ESP32 Ingest Script

1.  **Activate the environment**:
    ```bash
    source venv/bin/activate
    ```
2.  **To run the simulator** (sends one "fake" prediction):
    ```bash
    python scripts/ingest_from_esp32.py --simulate
    ```
3.  **To connect to your REAL ESP32** (plugged in via USB):
    ```bash
    python scripts/ingest_from_esp32.py
    ```
    _The script will auto-detect the port and start listening for JSON data._
