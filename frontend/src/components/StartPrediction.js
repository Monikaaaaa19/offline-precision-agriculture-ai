// frontend/src/components/StartPrediction.js
import React, { useState } from "react";
import axios from "axios";
import { motion } from "framer-motion";
import AnimatedResult from "./AnimatedResult";
import LiveFeed from "./LiveFeed";

const LoadingSpinner = () => (
  <div className="spinner-container">
    <div className="loading-spinner"></div>
  </div>
);

function StartPrediction(props) {
  const {
    N,
    setN,
    P,
    setP,
    K,
    setK,
    pH,
    setpH,
    temperature,
    setTemperature,
    humidity,
    setHumidity,
    rainfall,
    setRainfall,
    latitude,
    setLatitude,
    longitude,
    setLongitude,
    prediction,
    setPrediction,
    error,
    setError,
    isLoading,
    setIsLoading,
  } = props;

  const [isLive, setIsLive] = useState(false);

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    await new Promise((resolve) => setTimeout(resolve, 800));

    try {
      const dataToSend = {
        N: parseFloat(N),
        P: parseFloat(P),
        K: parseFloat(K),
        pH: parseFloat(pH),
        temperature: parseFloat(temperature),
        humidity: parseFloat(humidity),
        rainfall: parseFloat(rainfall),
        latitude: parseFloat(latitude),
        longitude: parseFloat(longitude),
        place_name: "User Form Input",
        polygon: [
          { lat: parseFloat(latitude), lng: parseFloat(longitude) },
          {
            lat: parseFloat(latitude) + 0.001,
            lng: parseFloat(longitude) + 0.001,
          },
          { lat: parseFloat(latitude), lng: parseFloat(longitude) + 0.001 },
        ],
      };

      const response = await axios.post("/predict_crop", dataToSend);
      setPrediction(response.data);
    } catch (err) {
      console.error("Prediction error:", err);
      if (err.response) {
        setError(
          `Server Error: ${err.response.data.detail || err.response.statusText}`
        );
      } else if (err.request) {
        setError("Cannot connect to server. Is the server running?");
      } else {
        setError(`Error: ${err.message}. Please check your input values.`);
      }
    } finally {
      setIsLoading(false);
    }
  };

  const formVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { duration: 0.45 } },
  };

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={formVariants}
      className="prediction-form-container"
      style={{ maxWidth: 920, margin: "0 auto", padding: "24px" }}
    >
      {/* Live feed on top */}
      <div style={{ marginBottom: 32 }}>
        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            marginBottom: 12,
          }}
        >
          <h2 style={{ fontSize: "1.5rem", color: "#2F80ED", margin: 0 }}>
            🌿 Live ESP32 Sensor Feed
          </h2>
          <span
            style={{
              fontSize: "0.9rem",
              background: isLive ? "#e63946" : "#6b7280",
              color: "white",
              padding: "6px 10px",
              borderRadius: 12,
              fontWeight: 600,
              boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
              transition: "all 0.25s ease",
            }}
          >
            {isLive ? "LIVE 🔴" : "OFFLINE ⚫"}
          </span>
        </div>

        <LiveFeed
          wsUrl={`ws://${window.location.hostname}:8000/ws/esp32`}
          onStatusChange={(status) => setIsLive(status === "online")}
        />
      </div>

      {/* Manual input form below (styled like LiveFeed card) */}
      {isLoading && <LoadingSpinner />}

      <div className="manual-card live-like-card">
        <form
          onSubmit={handleFormSubmit}
          className="manual-metrics-layout"
          aria-label="Manual input form"
        >
          <h2 className="manual-title">✋ Manual Input Section</h2>

          {/* Metric-style tiles (two-column grid) */}
          <div className="manual-metrics-grid">
            <div className="input-tile">
              <div className="tile-label">🌿 N</div>
              <input
                type="number"
                value={N}
                onChange={(e) => setN(e.target.value)}
                step="0.1"
                required
                className="tile-input"
              />
            </div>

            <div className="input-tile">
              <div className="tile-label">🌾 P</div>
              <input
                type="number"
                value={P}
                onChange={(e) => setP(e.target.value)}
                step="0.1"
                required
                className="tile-input"
              />
            </div>

            <div className="input-tile">
              <div className="tile-label">🧪 K</div>
              <input
                type="number"
                value={K}
                onChange={(e) => setK(e.target.value)}
                step="0.1"
                required
                className="tile-input"
              />
            </div>

            <div className="input-tile">
              <div className="tile-label">🍋 pH</div>
              <input
                type="number"
                value={pH}
                onChange={(e) => setpH(e.target.value)}
                step="0.01"
                required
                className="tile-input"
              />
            </div>

            <div className="input-tile">
              <div className="tile-label">🌡️ Temp (°C)</div>
              <input
                type="number"
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
                step="0.1"
                required
                className="tile-input"
              />
            </div>

            <div className="input-tile">
              <div className="tile-label">💧 Humidity (%)</div>
              <input
                type="number"
                value={humidity}
                onChange={(e) => setHumidity(e.target.value)}
                step="0.1"
                required
                className="tile-input"
              />
            </div>

            <div className="input-tile">
              <div className="tile-label">🌱 Soil (%)</div>
              <input
                type="number"
                value={""}
                onChange={() => {}}
                placeholder=""
                className="tile-input"
              />
            </div>

            <div className="input-tile">
              <div className="tile-label">🌧️ Rainfall</div>
              <input
                type="number"
                value={rainfall}
                onChange={(e) => setRainfall(e.target.value)}
                step="0.1"
                required
                className="tile-input"
              />
            </div>

            <div className="input-tile">
              <div className="tile-label">📍 Lat</div>
              <input
                type="number"
                value={latitude}
                onChange={(e) => setLatitude(e.target.value)}
                step="0.000001"
                required
                className="tile-input"
              />
            </div>

            <div className="input-tile">
              <div className="tile-label">📍 Lon</div>
              <input
                type="number"
                value={longitude}
                onChange={(e) => setLongitude(e.target.value)}
                step="0.000001"
                required
                className="tile-input"
              />
            </div>
          </div>

          <div className="manual-actions">
            <button
              type="submit"
              className="predict-button"
              disabled={isLoading}
            >
              {isLoading ? "Predicting..." : "Predict Crop"}
            </button>
          </div>
        </form>
      </div>

      {error && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="error-box"
        >
          <h3>Error</h3>
          <pre>{error}</pre>
        </motion.div>
      )}

      {prediction && <AnimatedResult data={prediction} />}
    </motion.div>
  );
}

export default StartPrediction;
