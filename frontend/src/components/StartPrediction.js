// frontend/src/components/StartPrediction.js
import React, { useState } from 'react';
import axios from 'axios';
import AnimatedResult from './AnimatedResult';
import { motion } from 'framer-motion';

// New component for the loading spinner
const LoadingSpinner = () => (
  <div className="spinner-container">
    <div className="loading-spinner"></div>
  </div>
);

function StartPrediction() {
  // --- State for each input field (now empty by default) ---
  const [N, setN] = useState('');
  const [P, setP] = useState('');
  const [K, setK] = useState('');
  const [pH, setpH] = useState('');
  const [temperature, setTemperature] = useState('');
  const [humidity, setHumidity] = useState('');
  const [rainfall, setRainfall] = useState('');
  const [latitude, setLatitude] = useState('');
  const [longitude, setLongitude] = useState('');
  // --------------------------------------------------------

  // --- Prediction Result States ---
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFormSubmit = async (event) => {
    event.preventDefault();
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    // We'll add a short delay so you can see the spinner
    await new Promise(resolve => setTimeout(resolve, 1000));

    try {
      // Construct the data payload from individual states
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
        
        // Add placeholder metadata so the server is happy
        place_name: "User Form Input",
        polygon: [
          {"lat": parseFloat(latitude), "lng": parseFloat(longitude)},
          {"lat": parseFloat(latitude) + 0.001, "lng": parseFloat(longitude) + 0.001},
          {"lat": parseFloat(latitude), "lng": parseFloat(longitude) + 0.001}
        ]
      };
      
      const response = await axios.post('/predict_crop', dataToSend);
      setPrediction(response.data);

    } catch (err) { 
      console.error("Prediction error:", err);
      if (err.response) {
        setError(`Server Error: ${err.response.data.detail || err.response.statusText}`);
      } else if (err.request) {
        setError("Cannot connect to server. Is the server running?");
      } else {
        setError(`Error: ${err.message}. Please check your input values.`);
      }
    } finally {
      setIsLoading(false);
    }
  };
  
  // Animation variants
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
      
      {/* Show spinner *over* the form when loading */}
      {isLoading && <LoadingSpinner />}
      
      <form onSubmit={handleFormSubmit} className="sensor-input-grid">
        
        <div className="input-group">
          <label htmlFor="N">🌿 Nitrogen (N)</label>
          <input type="number" id="N" value={N} onChange={(e) => setN(e.target.value)} step="0.1" required />
        </div>
        <div className="input-group">
          <label htmlFor="P">🌾 Phosphorus (P)</label>
          <input type="number" id="P" value={P} onChange={(e) => setP(e.target.value)} step="0.1" required />
        </div>
        <div className="input-group">
          <label htmlFor="K">🧪 Potassium (K)</label>
          <input type="number" id="K" value={K} onChange={(e) => setK(e.target.value)} step="0.1" required />
        </div>
        <div className="input-group">
          <label htmlFor="pH">🍋 pH Level</label>
          <input type="number" id="pH" value={pH} onChange={(e) => setpH(e.target.value)} step="0.1" required />
        </div>
        <div className="input-group">
          <label htmlFor="temperature">🌡️ Temperature (°C)</label>
          <input type="number" id="temperature" value={temperature} onChange={(e) => setTemperature(e.target.value)} step="0.1" required />
        </div>
        <div className="input-group">
          <label htmlFor="humidity">💧 Humidity (%)</label>
          <input type="number" id="humidity" value={humidity} onChange={(e) => setHumidity(e.target.value)} step="0.1" required />
        </div>

        <div className="input-group">
          <label htmlFor="latitude">📍 Latitude</label>
          <input type="number" id="latitude" value={latitude} onChange={(e) => setLatitude(e.target.value)} step="0.001" required />
        </div>
        <div className="input-group">
          <label htmlFor="longitude">📍 Longitude</label>
          <input type="number" id="longitude" value={longitude} onChange={(e) => setLongitude(e.target.value)} step="0.001" required />
        </div>

        <div className="input-group full-width">
          <label htmlFor="rainfall">🌧️ Rainfall (mm)</label>
          <input type="number" id="rainfall" value={rainfall} onChange={(e) => setRainfall(e.target.value)} step="0.1" required />
        </div>
        
        <button type="submit" disabled={isLoading} className="predict-button full-width">
          {isLoading ? 'Predicting...' : 'Predict Crop'}
        </button>
      </form>

      {/* --- Results Section --- */}
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

      {/* The 3D result will appear here */}
      {prediction && (
        <AnimatedResult data={prediction} />
      )}
    </motion.div>
  );
}

export default StartPrediction;