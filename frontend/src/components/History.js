// frontend/src/components/History.js
import React, { useState, useEffect } from "react";
import axios from "axios";
import { MapContainer, TileLayer, Marker, Popup } from "react-leaflet";
import "leaflet/dist/leaflet.css";
import L from "leaflet"; // Import L to fix a known issue with icons

// --- FIX for missing Leaflet marker icons ---
// This is a common bug when using react-leaflet with bundlers
import icon from "leaflet/dist/images/marker-icon.png";
import iconShadow from "leaflet/dist/images/marker-shadow.png";

let DefaultIcon = L.icon({
  iconUrl: icon,
  shadowUrl: iconShadow,
  iconAnchor: [12, 41], // Manually set anchor
});

L.Marker.prototype.options.icon = DefaultIcon;
// --- End of icon fix ---

// A simple loading spinner (can be the same as StartPrediction's)
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
  }, []); // The empty array [] means "run this only once"

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

  // --- Helper to render a map ---
  // We check if the data has the lat/lon to display
  const renderMap = (record) => {
    const { latitude, longitude, place_name } = record.received_data;

    if (latitude && longitude) {
      const position = [latitude, longitude];
      return (
        <div className="history-map-container">
          <MapContainer
            center={position}
            zoom={13}
            style={{ height: "200px", width: "100%" }}
          >
            {/* This TileLayer uses OpenStreetMap. It *will* try to connect
              to the internet. This is the one part that isn't fully
              offline by default, but it will degrade gracefully and just
              show a gray box if no internet is present.
              For a *truly* offline app, you'd need local map tiles.
            */}
            <TileLayer
              attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
              url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
            />
            <Marker position={position}>
              <Popup>{place_name || "Prediction Location"}</Popup>
            </Marker>
          </MapContainer>
        </div>
      );
    }
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
