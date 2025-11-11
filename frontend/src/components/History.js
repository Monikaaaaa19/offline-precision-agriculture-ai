// frontend/src/components/History.js
import React, { useState, useEffect } from "react";
import axios from "axios";

// A simple loading spinner
const LoadingSpinner = () => (
  <div className="history-spinner-container">
    <div className="loading-spinner"></div>
  </div>
);

function History() {
  const [history, setHistory] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);

  // This useEffect runs once when the component mounts
  useEffect(() => {
    const fetchHistory = async () => {
      setIsLoading(true);
      setError(null);
      try {
        // Fetch data from our server's /history endpoint
        const response = await axios.get("/history");
        // We reverse the array so the newest predictions are first
        setHistory(response.data.reverse());
      } catch (err) {
        console.error("Error fetching history:", err);
        setError("Could not connect to server to fetch history.");
      } finally {
        setIsLoading(false);
      }
    };

    fetchHistory();
  }, []);

  if (isLoading) {
    return <LoadingSpinner />;
  }

  if (error) {
    return <div className="error-box">{error}</div>;
  }

  if (history.length === 0) {
    return (
      <div className="history-container">
        <h2>Prediction History</h2>
        <p>No predictions have been saved yet.</p>
        <p>Run a prediction from the "Start" tab to see it here.</p>
      </div>
    );
  }

  // --- Helper to render the OFFLINE map image ---
  const renderMap = (record) => {
    const { latitude, longitude } = record.received_data;

    // Check if the record has location data
    if (latitude && longitude) {
      // The image URL now points to our new FastAPI endpoint
      // This is 100% OFFLINE
      const imageUrl = `/history_map/${record.id}.png`;
      return (
        <img
          src={imageUrl}
          alt={`Map for Prediction ${record.id}`}
          className="offline-map-image" // We'll add a style for this
        />
      );
    }
    // This is for old records that had no location
    return (
      <div className="history-map-placeholder">No location data provided.</div>
    );
  };

  return (
    <div className="history-container">
      <h2>Prediction History</h2>
      <p>Showing the {history.length} most recent predictions.</p>

      <div className="history-list">
        {history.map((record) => (
          <div className="history-card" key={record.id}>
            <div className="history-card-details">
              <h3>{record.received_data.place_name || "Unnamed Prediction"}</h3>

              {/* --- NEW: Display State Name --- */}
              <p>
                <strong>State:</strong> {record.received_data.state || "N/A"}
              </p>
              {/* ----------------------------- */}

              <p>
                <strong>Predicted Crop:</strong> {record.predicted_crop}
              </p>
              <p>
                <strong>Alerts:</strong> {record.disease_alerts.join(", ")}
              </p>
              <p>
                <strong>Fertilizer:</strong> {record.fertilizer_recommendation}
              </p>
              <p className="history-timestamp">
                Saved at: {new Date(record.saved_at).toLocaleString()}
              </p>
            </div>
            <div className="history-card-map">{renderMap(record)}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default History;
