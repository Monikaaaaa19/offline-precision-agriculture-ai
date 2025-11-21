// src/components/StartPrediction.js
import React, { useState } from "react";
import AnimatedResult from "./AnimatedResult";
import "../App.css";

const initialForm = {
  N: "",
  P: "",
  K: "",
  pH: "",
  temperature: "",
  humidity: "",
  rainfall: "",
  latitude: "",
  longitude: "",
};

const API_BASE = process.env.REACT_APP_API_BASE || "http://localhost:8000";

function StartPrediction() {
  const [form, setForm] = useState(initialForm);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setForm((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setIsSubmitting(true);

    try {
      const payload = {
        N: Number(form.N),
        P: Number(form.P),
        K: Number(form.K),
        pH: Number(form.pH),
        temperature: Number(form.temperature),
        humidity: Number(form.humidity),
        rainfall: Number(form.rainfall),
        latitude: Number(form.latitude),
        longitude: Number(form.longitude),
      };

      const res = await fetch(`${API_BASE}/predict_crop`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(payload),
      });

      if (!res.ok) {
        const txt = await res.text();
        console.error("Prediction error response:", txt);
        throw new Error(
          `Server error (${res.status}): ${txt || res.statusText}`
        );
      }

      const data = await res.json();
      console.log("Prediction response:", data);
      setPrediction(data);
      setError(null);
    } catch (err) {
      console.error("Prediction request failed:", err);
      setPrediction(null);
      setError(
        "Could not get a prediction from the server. Check if the backend (FastAPI) is running and see its console for errors."
      );
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="tab-panel manual-tab">
      <div className="live-card manual-card">
        {/* Header row for this card */}
        <div className="card-header">
          <div>
            <h2>Manual Prediction</h2>
          </div>
          <span className="badge badge-manual">Manual Input</span>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="grid-form">
          {/* N */}
          <div className="form-group">
            <label className="input-label" htmlFor="N">
              🌱 N
            </label>
            <input
              id="N"
              name="N"
              type="number"
              value={form.N}
              onChange={handleChange}
              required
            />
          </div>

          {/* P */}
          <div className="form-group">
            <label className="input-label" htmlFor="P">
              🌿 P
            </label>
            <input
              id="P"
              name="P"
              type="number"
              value={form.P}
              onChange={handleChange}
              required
            />
          </div>

          {/* K */}
          <div className="form-group">
            <label className="input-label" htmlFor="K">
              🌾 K
            </label>
            <input
              id="K"
              name="K"
              type="number"
              value={form.K}
              onChange={handleChange}
              required
            />
          </div>

          {/* pH */}
          <div className="form-group">
            <label className="input-label" htmlFor="pH">
              🍋 pH
            </label>
            <input
              id="pH"
              name="pH"
              type="number"
              step="0.01"
              value={form.pH}
              onChange={handleChange}
              required
            />
          </div>

          {/* Temperature */}
          <div className="form-group">
            <label className="input-label" htmlFor="temperature">
              🌡 Temp (°C)
            </label>
            <input
              id="temperature"
              name="temperature"
              type="number"
              step="0.1"
              value={form.temperature}
              onChange={handleChange}
              required
            />
          </div>

          {/* Humidity */}
          <div className="form-group">
            <label className="input-label" htmlFor="humidity">
              💧 Humidity (%)
            </label>
            <input
              id="humidity"
              name="humidity"
              type="number"
              step="0.1"
              value={form.humidity}
              onChange={handleChange}
              required
            />
          </div>

          {/* Rainfall */}
          <div className="form-group">
            <label className="input-label" htmlFor="rainfall">
              🌧 Rainfall (mm)
            </label>
            <input
              id="rainfall"
              name="rainfall"
              type="number"
              step="0.1"
              value={form.rainfall}
              onChange={handleChange}
              required
            />
          </div>

          {/* Latitude */}
          <div className="form-group">
            <label className="input-label" htmlFor="latitude">
              📍 Lat
            </label>
            <input
              id="latitude"
              name="latitude"
              type="number"
              step="0.000001"
              value={form.latitude}
              onChange={handleChange}
              required
            />
          </div>

          {/* Longitude */}
          <div className="form-group">
            <label className="input-label" htmlFor="longitude">
              📍 Lon
            </label>
            <input
              id="longitude"
              name="longitude"
              type="number"
              step="0.000001"
              value={form.longitude}
              onChange={handleChange}
              required
            />
          </div>

          {/* Errors */}
          {error && <div className="form-error">{error}</div>}

          {/* Button */}
          <button type="submit" className="primary-btn" disabled={isSubmitting}>
            {isSubmitting ? "Predicting…" : "Predict Crop"}
          </button>
        </form>

        {/* Result card (only shows once we have a prediction or an error) */}
        <AnimatedResult
          prediction={prediction}
          loading={isSubmitting}
          error={error}
        />
      </div>
    </div>
  );
}

export default StartPrediction;
