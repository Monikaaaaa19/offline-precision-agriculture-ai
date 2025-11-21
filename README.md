# Real-Time Offline AI System for Sensor-Integrated Precision Agriculture

This project presents an end-to-end _offline, sensor-driven agricultural intelligence system_ capable of performing real-time soil analysis, crop prediction, fertilizer recommendation, disease risk estimation, and spatial field mapping **without internet dependency**. The system integrates multi-sensor hardware, machine learning models, and local geospatial analytics to support precision agriculture in connectivity-constrained environments.

---

## **1. Overview**

Traditional digital agriculture solutions rely heavily on cloud connectivity and remote inference infrastructure, making them unsuitable for rural agricultural zones with limited network access. This project addresses this gap by enabling **fully offline data acquisition, model inference, mapping, and advisory generation**, powered by:

- Local ML inference (Joblib models)
- Real-time NPK + environmental sensing via Arduino
- Automatic geolocation-based crop suitability
- Local disease risk retrieval
- Offline map rendering and state-boundary overlay
- Manual input and historical analytics interface

The system can operate autonomously in remote fields using local compute hardware (e.g., Raspberry Pi or laptop) paired with an Arduino sensor unit.

---

## **2. Core Contributions**

| Contribution                                 | Description                                                                       |
| -------------------------------------------- | --------------------------------------------------------------------------------- |
| **Offline AI Inference Pipeline**            | All predictions occur locally without cloud APIs or network calls.                |
| **Multi-Sensor Acquisition**                 | Live soil nutrient readings (NPK), pH, temperature, humidity, rainfall, location. |
| **Dataset-Driven Fertilizer Recommendation** | Recommendations derived from empirical dataset rather than static rules.          |
| **Real-Time WebSocket Streaming**            | Streaming pipeline from Arduino → FastAPI → React UI.                             |
| **Offline Geographic Mapping**               | Field boundaries plotted using local GeoJSON + Matplotlib.                        |
| **Spatial Crop Suitability Modelling**       | State lookup via geospatial boundaries & coordinate inference.                    |

---

## **3. System Architecture**

┌───────────┐ USB/Serial ┌──────────────┐
│ Arduino │ ───────────────▶ │ Python Ingest │
│ (NPK + pH)│ │ Script │
└───────────┘ └──────────────┘
(POST / WebSocket)
│
▼
┌──────────────────┐
│ FastAPI Backend │
│ - ML Models │
│ - Geospatial │
│ - History Store │
└──────────────────┘
│ WebSocket
▼
┌──────────────────┐
│ React Frontend │
│ - Live Feed │
│ - Manual Input │
│ - History Maps │
└──────────────────┘

---

## **4. Features**

### **4.1 Real-Time Live Sensor Feed**

- Serial ingestion via Python script
- Calibration-aware corrected values
- Auto-prediction on significant change thresholds

### **4.2 Offline Prediction Engine**

- Model trained on crop–soil–climate dataset
- Scikit-learn based pipeline exported as Joblib

### **4.3 Fertilizer Recommendation**

- Derived directly from the dataset (`arginode_corrected_fertilizers.csv`)
- No default static recommendation

### **4.4 Local Geospatial Analysis**

- State boundary lookup from `india_states.geojson`
- Field polygon → area calculation (acres)
- Matplotlib map export

---

## **5. Tech Stack**

| Layer                  | Technology                                          |
| ---------------------- | --------------------------------------------------- |
| **Frontend**           | React, WebSockets, CSS UI components                |
| **Backend**            | FastAPI, NumPy, scikit-learn, matplotlib, GeoPandas |
| **Hardware Interface** | Arduino + Sensors (NPK, pH, Temp, Humidity, Rain)   |
| **Storage**            | Local JSON-based history logging                    |
| **Deployment Target**  | Offline Raspberry Pi / Local laptop                 |

---

## **6. Setup Instructions**

### **Clone the Repository**

```sh
git clone https://github.com/Monikaaaaa19/offline-precision-agriculture-ai.git
cd offline-precision-agriculture-ai

Create Python Environment

pip install -r requirements.txt

Start Backend

uvicorn server.main:app --reload --port 8000

Start Frontend

npm install
npm start

Start Arduino Ingestion

python scripts/ingest_arduino.py --port "/dev/cu.usbserial-XXXX"


⸻

7. Dataset

This project uses a real-world compiled dataset containing:
	•	N, P, K nutrient values
	•	Soil pH and moisture
	•	Temperature and rainfall
	•	Latitude + longitude + state mapping
	•	Crop labels
	•	Fertilizer recommendations

Location:

data/arginode_corrected_fertilizers.csv

The fertilizer recommendation model performs nearest-neighbor feature search matching sensor inputs to similar historical samples.

⸻


```
