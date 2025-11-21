// src/components/AnimatedResult.js
import React from "react";
import "./AnimatedResult.css";

function AnimatedResult({ prediction, loading, error }) {
  if (loading) {
    return (
      <div className="prediction-card prediction-card-empty">
        <p className="prediction-title">Running prediction…</p>
        <p className="prediction-subtitle">
          Analysing your soil and environment inputs.
        </p>
      </div>
    );
  }

  if (!prediction && !error) {
    // Nothing to show yet
    return null;
  }

  if (error && !prediction) {
    return (
      <div className="prediction-card prediction-card-empty">
        <p className="prediction-title">Prediction Failed</p>
        <p className="prediction-error-text">{error}</p>
      </div>
    );
  }

  const {
    predicted_crop,
    confidence,
    fertilizer_recommendation,
    disease_alerts = [],
    received_data,
  } = prediction;

  const state = received_data?.state ?? "Unknown";
  const confPercent =
    typeof confidence === "number"
      ? Math.round(confidence * 100)
      : null;

  return (
    <div className="prediction-card">
      <div className="prediction-header-row">
        <div>
          <p className="prediction-title">Prediction Successful</p>
          <p className="prediction-subtitle">
            {received_data?.place_name
              ? `User Form Input – ${received_data.place_name}`
              : "User Form Input"}
          </p>
        </div>

        <div className="prediction-chip">
          <div className="prediction-chip-label">Predicted Crop</div>
          <div className="prediction-chip-value">
            {predicted_crop || "—"}
          </div>
        </div>
      </div>

      {/* Confidence bar */}
      {confPercent !== null && (
        <div className="prediction-confidence">
          <div className="prediction-confidence-label">
            Model Confidence
          </div>
          <div className="prediction-confidence-bar">
            <div
              className="prediction-confidence-fill"
              style={{ width: `${Math.min(100, confPercent)}%` }}
            />
          </div>
          <div className="prediction-confidence-value">
            {confPercent}%
          </div>
        </div>
      )}

      {/* Details grid */}
      <div className="prediction-details-grid">
        <div className="prediction-detail-row">
          <span className="prediction-detail-label">State / Region</span>
          <span className="prediction-detail-value">{state}</span>
        </div>

        <div className="prediction-detail-row">
          <span className="prediction-detail-label">Predicted Crop</span>
          <span className="prediction-detail-value">
            {predicted_crop || "—"}
          </span>
        </div>

        <div className="prediction-detail-row">
          <span className="prediction-detail-label">
            Fertilizer Recommendation
          </span>
          <span className="prediction-detail-value">
            {fertilizer_recommendation || "—"}
          </span>
        </div>

        <div className="prediction-detail-row prediction-detail-row--diseases">
          <span className="prediction-detail-label">Disease Alerts</span>
          <div className="prediction-disease-tags">
            {disease_alerts && disease_alerts.length > 0 ? (
              disease_alerts.map((d) => (
                <span key={d} className="prediction-tag">
                  {d}
                </span>
              ))
            ) : (
              <span className="prediction-detail-value">—</span>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default AnimatedResult;