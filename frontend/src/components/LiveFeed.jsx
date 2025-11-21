// src/components/LiveFeed.jsx
import React, { useEffect, useRef, useState } from "react";
import "../App.css";

const LiveFeed = () => {
  const [wsStatus, setWsStatus] = useState("connecting"); // connecting | connected | error
  const [deviceOnline, setDeviceOnline] = useState(false);
  const [statusUpdatedAt, setStatusUpdatedAt] = useState(null);

  const [latestSensor, setLatestSensor] = useState(null);
  const [lastCalibration, setLastCalibration] = useState(null);
  const [recentReadings, setRecentReadings] = useState([]);
  const [latestPrediction, setLatestPrediction] = useState(null);

  // Only show predictions that come *after* this page has seen a live sensor packet
  const hasLiveSensorRef = useRef(false);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const wsUrl = `${protocol}://${window.location.hostname}:8000/ws/esp32`;

    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      setWsStatus("connected");
    };

    socket.onerror = () => {
      setWsStatus("error");
    };

    socket.onclose = () => {
      setWsStatus("error");
    };

    socket.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        const { type, data } = msg;

        if (!type) return;

        switch (type) {
          case "status": {
            const online = !!data?.online;
            setDeviceOnline(online);
            if (data?.updated_at) {
              setStatusUpdatedAt(new Date(data.updated_at * 1000));
            }
            break;
          }

          case "calibration": {
            setLastCalibration(data || null);
            break;
          }

          case "history": {
            if (Array.isArray(data)) {
              const mapped = data
                .slice(-10)
                .reverse()
                .map((item) => mapSensorPacket(item));
              setRecentReadings(mapped);
            }
            break;
          }

          case "sensor": {
            const mapped = mapSensorPacket(data);
            hasLiveSensorRef.current = true;
            setLatestSensor(mapped);

            setRecentReadings((prev) => {
              const next = [mapped, ...prev];
              return next.slice(0, 8);
            });
            break;
          }

          case "prediction": {
            // Ignore predictions until we’ve seen at least one live sensor packet
            if (!hasLiveSensorRef.current) return;

            const base = data || {};
            const rec = base.received_data || {};

            const pred = {
              crop: base.predicted_crop || "-",
              confidence: base.confidence != null ? base.confidence : null,
              fertilizer: base.fertilizer_recommendation || "-",
              diseases: Array.isArray(base.disease_alerts)
                ? base.disease_alerts
                : [],
              place: rec.place_name || "Arduino Field",
              state: rec.state || "Unknown",
            };

            setLatestPrediction(pred);
            break;
          }

          default:
            break;
        }
      } catch (err) {
        console.error("WS message parse error", err);
      }
    };

    return () => {
      try {
        socket.close();
      } catch (e) {
        /* ignore */
      }
    };
  }, []);

  // ---------- Helper mapping from backend packet → flat object ----------
  function mapSensorPacket(packet) {
    const data = packet || {};
    const corrected = data.corrected || {};
    const raw = data.raw || {};

    const getFirst = (...candidates) => {
      for (const c of candidates) {
        if (c !== undefined && c !== null) return c;
      }
      return null;
    };

    const ts =
      typeof data.ts === "number"
        ? new Date(data.ts * 1000)
        : data.ts
        ? new Date(data.ts)
        : null;

    return {
      ts,
      calibrated: !!data.calibrated,

      n: toNumber(
        getFirst(corrected.n, corrected.N, data.n, data.N, raw.n, raw.N)
      ),
      p: toNumber(
        getFirst(corrected.p, corrected.P, data.p, data.P, raw.p, raw.P)
      ),
      k: toNumber(
        getFirst(corrected.k, corrected.K, data.k, data.K, raw.k, raw.K)
      ),

      ph: toNumber(getFirst(data.ph, data.pH, data.soil_ph)),
      temperature: toNumber(
        getFirst(data.temperature, data.temp, data.avg_temp_c)
      ),
      humidity: toNumber(
        getFirst(data.humidity, data.h, data.soil_moisture_pct)
      ),
      rainfall: toNumber(getFirst(data.rainfall, data.annual_rainfall_mm, 0)),
      lat: toNumber(getFirst(data.latitude, data.lat)),
      lon: toNumber(getFirst(data.longitude, data.lon)),
    };
  }

  function toNumber(value) {
    if (value === null || value === undefined) return null;
    const num = Number(value);
    return Number.isFinite(num) ? num : null;
  }

  const formatTs = (ts) => {
    if (!ts) return "—";
    try {
      return ts.toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        second: "2-digit",
      });
    } catch {
      return "—";
    }
  };

  const formatDateTime = (ts) => {
    if (!ts) return "—";
    try {
      return ts.toLocaleString();
    } catch {
      return "—";
    }
  };

  // ---------- RENDER HELPERS ----------
  // label ABOVE pill, value INSIDE pill
  const renderSensorField = (label, value, unit, mutedIfMissing = false) => {
    const isMissing = value === null || value === undefined;

    return (
      <div className="live-sensor-field">
        <div className="live-sensor-label-outside">{label}</div>

        <div
          className={
            "live-sensor-pill" +
            (isMissing && mutedIfMissing ? " live-sensor-pill-muted" : "")
          }
        >
          <div className="live-sensor-value">
            {isMissing ? (
              <span className="live-sensor-dash">—</span>
            ) : (
              <>
                <span>{value}</span>
                {unit && <span className="live-sensor-unit">{unit}</span>}
              </>
            )}
          </div>
        </div>
      </div>
    );
  };

  // ---------- JSX ----------
  // Use an empty object so the grid always renders “boxes” even before data
  const sensor = latestSensor || {};

  return (
    <div className="live-card">
      {/* Header */}
      <div className="live-card-header">
        <div>
          <h2 className="live-title">Live Arduino Sensor Feed</h2>
          <p className="live-subtitle">
            Streaming calibrated NPK values and environment readings from your
            Arduino board.
          </p>

          <div className="live-status-row">
            <span
              className={
                "status-pill " +
                (wsStatus === "connected"
                  ? "status-pill-ok"
                  : "status-pill-bad")
              }
            >
              ● WebSocket:{" "}
              {wsStatus === "connected" ? "Connected" : "Disconnected"}
            </span>
            <span
              className={
                "status-pill " +
                (deviceOnline ? "status-pill-ok" : "status-pill-bad")
              }
            >
              ● Device {deviceOnline ? "Online" : "Offline"}
            </span>
          </div>
        </div>

        <div className="live-arduino-pill">
          <span>ARDUINO · WEBSOCKET</span>
        </div>
      </div>

      {/* Main 2-column row */}
      <div className="live-main-row">
        {/* Left: Latest Sensor Reading */}
        <div className="live-panel live-panel-sensors">
          <div className="live-panel-header">
            <h3>Latest Sensor Reading</h3>
            <span
              className={
                "calibration-pill " +
                (sensor.calibrated
                  ? "calibration-pill-ok"
                  : "calibration-pill-bad")
              }
            >
              {sensor.calibrated ? "Calibrated" : "Not Calibrated"}
            </span>
          </div>

          {/* Always show the grid, values default to dashes */}
          <div className="live-sensor-grid">
            {renderSensorField(
              "🌱 N",
              sensor.n != null ? sensor.n.toFixed(1) : null,
              "mg/kg"
            )}
            {renderSensorField(
              "🌿 P",
              sensor.p != null ? sensor.p.toFixed(1) : null,
              "mg/kg"
            )}
            {renderSensorField(
              "🌾 K",
              sensor.k != null ? sensor.k.toFixed(1) : null,
              "mg/kg"
            )}

            {renderSensorField(
              "🍋 pH",
              sensor.ph != null ? sensor.ph.toFixed(1) : null,
              ""
            )}
            {renderSensorField(
              "🌡 Temp (°C)",
              sensor.temperature != null ? sensor.temperature.toFixed(1) : null,
              "°C"
            )}
            {renderSensorField(
              "💧 Humidity (%)",
              sensor.humidity != null ? sensor.humidity.toFixed(1) : null,
              "%"
            )}

            {renderSensorField(
              "🌧 Rainfall (mm)",
              sensor.rainfall != null ? sensor.rainfall.toFixed(1) : null,
              "mm",
              true
            )}
            {renderSensorField(
              "📍 Lat",
              sensor.lat != null ? sensor.lat.toFixed(3) : null,
              "",
              true
            )}
            {renderSensorField(
              "📍 Lon",
              sensor.lon != null ? sensor.lon.toFixed(3) : null,
              "",
              true
            )}
          </div>

          {!latestSensor && (
            <p className="live-empty-text">
              No readings yet… Start the Arduino ingestion script to see live
              values.
            </p>
          )}

          <div className="live-raw-line">
            Raw:{" "}
            {latestSensor
              ? `N ${latestSensor.n ?? "—"}, P ${latestSensor.p ?? "—"}, K ${
                  latestSensor.k ?? "—"
                }`
              : "N —, P —, K —"}
          </div>
        </div>

        {/* Right: Latest Live Prediction */}
        <div className="live-panel live-panel-prediction">
          <div className="live-panel-header">
            <h3>Latest Live Prediction</h3>
            <span className="live-panel-sub">
              Generated automatically from live Arduino readings.
            </span>
          </div>

          {latestPrediction ? (
            <div className="live-prediction-body">
              <div className="live-prediction-title">
                {latestPrediction.crop}
              </div>

              <div className="live-prediction-row">
                <span className="live-prediction-label">State / Region:</span>
                <span>{latestPrediction.state}</span>
              </div>
              <div className="live-prediction-row">
                <span className="live-prediction-label">Predicted Crop:</span>
                <span>{latestPrediction.crop}</span>
              </div>
              <div className="live-prediction-row">
                <span className="live-prediction-label">Fertilizer:</span>
                <span>{latestPrediction.fertilizer}</span>
              </div>
              <div className="live-prediction-row">
                <span className="live-prediction-label">Confidence:</span>
                <span>
                  {latestPrediction.confidence != null
                    ? `${(latestPrediction.confidence * 100).toFixed(1)}%`
                    : "—"}
                </span>
              </div>

              <div className="live-prediction-row">
                <span className="live-prediction-label">Disease Risks:</span>
                <div className="live-disease-tags">
                  {latestPrediction.diseases.length ? (
                    latestPrediction.diseases.map((d) => (
                      <span key={d} className="pill-tag">
                        {d}
                      </span>
                    ))
                  ) : (
                    <span>—</span>
                  )}
                </div>
              </div>
            </div>
          ) : (
            <p className="live-empty-text">No live prediction yet…</p>
          )}

          <div className="live-status-footer">
            <span>Last status update</span>
            <span> {formatDateTime(statusUpdatedAt)} </span>
          </div>
        </div>
      </div>

      {/* Calibration */}
      <div className="live-calibration-section">
        <h4>Calibration</h4>
        {lastCalibration ? (
          <>
            <p>
              Average raw NPK:{" "}
              <strong>
                {lastCalibration.avg_raw?.n?.toFixed
                  ? lastCalibration.avg_raw.n.toFixed(1)
                  : lastCalibration.avg_raw?.n ?? "—"}
                {" / "}
                {lastCalibration.avg_raw?.p?.toFixed
                  ? lastCalibration.avg_raw.p.toFixed(1)
                  : lastCalibration.avg_raw?.p ?? "—"}
                {" / "}
                {lastCalibration.avg_raw?.k?.toFixed
                  ? lastCalibration.avg_raw.k.toFixed(1)
                  : lastCalibration.avg_raw?.k ?? "—"}
              </strong>
              {" · "}
              calibrated at{" "}
              {lastCalibration.calibrated_at
                ? new Date(
                    lastCalibration.calibrated_at * 1000
                  ).toLocaleTimeString()
                : "—"}
            </p>
          </>
        ) : (
          <p>No calibration info received yet.</p>
        )}
      </div>

      {/* Recent Sensor Readings */}
      <div className="live-recent-section">
        <h4>Recent Sensor Readings</h4>
        {recentReadings.length === 0 ? (
          <p>No history yet.</p>
        ) : (
          <ul className="live-recent-list">
            {recentReadings.map((item, idx) => (
              <li key={idx} className="live-recent-item">
                <span className="live-recent-main">
                  N {item.n ?? "—"} · P {item.p ?? "—"} · K {item.k ?? "—"}
                </span>
                <span className="live-recent-time">{formatTs(item.ts)}</span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
};

export default LiveFeed;