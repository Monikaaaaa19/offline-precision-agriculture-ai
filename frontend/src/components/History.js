// src/components/History.js
import React, { useEffect, useState } from "react";
import "../App.css";

const History = () => {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        setLoading(true);
        setError(null);

        // Use absolute backend URL; change to "/history" if you proxy through Vite/CRA
        const res = await fetch("http://localhost:8000/history");
        if (!res.ok) {
          throw new Error(`Server responded with ${res.status}`);
        }

        const data = await res.json();
        if (Array.isArray(data)) {
          // newest first
          setItems([...data].reverse());
        } else {
          setItems([]);
        }
      } catch (err) {
        console.error("Failed to load history", err);
        setError("Could not load prediction history.");
      } finally {
        setLoading(false);
      }
    };

    fetchHistory();
  }, []);

  const formatTimestamp = (raw) => {
    if (!raw) return "—";
    try {
      // handle epoch seconds or ISO strings
      if (typeof raw === "number") {
        return new Date(raw * 1000).toLocaleString();
      }
      return new Date(raw).toLocaleString();
    } catch {
      return String(raw);
    }
  };

  return (
    <div className="history-card">
      <div className="history-header-row">
        <div>
          <div className="history-title">Prediction History</div>
          <div className="history-subtitle">
            Showing the {items.length} most recent predictions.
          </div>
        </div>
      </div>

      {loading && (
        <div className="history-spinner-container">
          <div className="loading-spinner" />
        </div>
      )}

      {error && <div className="history-error">{error}</div>}

      {!loading && !error && items.length === 0 && (
        <div className="history-helper-text">
          No predictions saved yet. Run a manual prediction or wait for live
          Arduino data to start building history.
        </div>
      )}

      {!loading && !error && items.length > 0 && (
        <div className="history-list">
          {items.map((item, idx) => {
            const rd = item.received_data || {};
            const place = rd.place_name || "Arduino Field";
            const state = rd.state || "Unknown";
            const crop = item.predicted_crop || "-";
            const fertilizer = item.fertilizer_recommendation || "-";
            const alerts = Array.isArray(item.disease_alerts)
              ? item.disease_alerts
              : [];
            const ts =
              item.saved_at || item.created_at || item.timestamp || item.ts;
            const formattedTs = formatTimestamp(ts);

            const id = item.id != null ? item.id : idx;
            const mapUrl =
              item.id != null ? `/history_map/${item.id}.png` : null;

            return (
              <div key={id} className="history-item-card">
                <div className="history-item-main">
                  <div className="history-item-title">{place}</div>

                  <div className="history-row">
                    <span className="history-label">State:</span>
                    <span>{state}</span>
                  </div>

                  <div className="history-row">
                    <span className="history-label">Predicted Crop:</span>
                    <span>{crop}</span>
                  </div>

                  <div className="history-row">
                    <span className="history-label">Alerts:</span>
                    <span>
                      {alerts.length ? alerts.join(", ") : "No major alerts"}
                    </span>
                  </div>

                  <div className="history-row">
                    <span className="history-label">Fertilizer:</span>
                    <span>{fertilizer}</span>
                  </div>

                  <div className="history-row-muted">
                    <span className="history-label">Saved at: </span>
                    <span>{formattedTs}</span>
                  </div>
                </div>

                <div className="history-item-map">
                  {mapUrl ? (
                    <img
                      src={mapUrl}
                      alt={`Location map for ${place}`}
                      className="offline-map-image"
                    />
                  ) : (
                    <div className="history-map-placeholder">
                      No map available for this prediction.
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default History;
