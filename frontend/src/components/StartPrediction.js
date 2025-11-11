// frontend/src/components/StartPrediction.js
import React from "react"; // We removed 'useState'
import axios from "axios";
import AnimatedResult from "./AnimatedResult";
import { motion } from "framer-motion"; // <-- FIX: Was 'in', now 'from'

const LoadingSpinner = () => (
  <div className="spinner-container">
    <div className="loading-spinner"></div>
  </div>
);

// We now receive ALL state as props from App.js
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

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    await new Promise((resolve) => setTimeout(resolve, 1000));

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
    visible: { opacity: 1, y: 0, transition: { duration: 0.5 } },
  };

  return (
    <motion.div
      initial="hidden"
      animate="visible"
      variants={formVariants}
      className="prediction-form-container"
    >
      {isLoading && <LoadingSpinner />}

      <form onSubmit={handleFormSubmit} className="sensor-input-grid">
        {/* All these inputs now use props, e.g., value={N} and onChange={e => setN(e.target.value)} */}
        <div className="input-group">
          <label htmlFor="N">🌿 Nitrogen (N)</label>
          <input
            type="number"
            id="N"
            value={N}
            onChange={(e) => setN(e.target.value)}
            step="0.1"
            required
          />
        </div>
        <div className="input-group">
          <label htmlFor="P">🌾 Phosphorus (P)</label>
          <input
            type="number"
            id="P"
            value={P}
            onChange={(e) => setP(e.target.value)}
            step="0.1"
            required
          />
        </div>
        <div className="input-group">
          <label htmlFor="K">🧪 Potassium (K)</label>
          <input
            type="number"
            id="K"
            value={K}
            onChange={(e) => setK(e.target.value)}
            step="0.1"
            required
          />
        </div>
        <div className="input-group">
          <label htmlFor="pH">🍋 pH Level</label>
          <input
            type="number"
            id="pH"
            value={pH}
            onChange={(e) => setpH(e.target.value)}
            step="0.1"
            required
          />
        </div>
        <div className="input-group">
          <label htmlFor="temperature">🌡️ Temperature (°C)</label>
          <input
            type="number"
            id="temperature"
            value={temperature}
            onChange={(e) => setTemperature(e.target.value)}
            step="0.1"
            required
          />
        </div>
        <div className="input-group">
          <label htmlFor="humidity">💧 Soil Moisture (%)</label>
          <input
            type="number"
            id="humidity"
            value={humidity}
            onChange={(e) => setHumidity(e.target.value)}
            step="0.1"
            required
          />
        </div>
        <div className="input-group">
          <label htmlFor="latitude">📍 Latitude</label>
          <input
            type="number"
            id="latitude"
            value={latitude}
            onChange={(e) => setLatitude(e.target.value)}
            step="0.001"
            required
          />
        </div>
        <div className="input-group">
          <label htmlFor="longitude">📍 Longitude</label>
          <input
            type="number"
            id="longitude"
            value={longitude}
            onChange={(e) => setLongitude(e.target.value)}
            step="0.001"
            required
          />
        </div>
        <div className="input-group full-width">
          <label htmlFor="rainfall">🌧️ Rainfall (mm)</label>
          <input
            type="number"
            id="rainfall"
            value={rainfall}
            onChange={(e) => setRainfall(e.target.value)}
            step="0.1"
            required
          />
        </div>

        <button
          type="submit"
          disabled={isLoading}
          className="predict-button full-width"
        >
          {isLoading ? "Predicting..." : "Predict Crop"}
        </button>
      </form>

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
