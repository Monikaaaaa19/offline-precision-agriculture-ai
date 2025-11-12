// frontend/src/components/LiveFeed.jsx
import React, { useEffect, useRef, useState, useCallback } from "react";
import styles from "./LiveFeed.module.css";
import { motion } from "framer-motion";

const DEFAULT_WS_URL = `ws://${window.location.hostname}:8000/ws/esp32`;

export default function LiveFeed({ wsUrl = DEFAULT_WS_URL }) {
  const [status, setStatus] = useState("offline");
  const [latest, setLatest] = useState(null);
  const [log, setLog] = useState([]);
  const [paused, setPaused] = useState(false);
  const wsRef = useRef(null);
  const reconnectTimerRef = useRef(null);

  const connect = useCallback(() => {
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {}
    }
    setStatus("connecting");
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => setStatus("online");
    ws.onclose = () => {
      setStatus("offline");
      scheduleReconnect();
    };
    ws.onerror = () => {
      setStatus("offline");
      scheduleReconnect();
    };
    ws.onmessage = (e) => {
      if (paused) return;
      // be tolerant: server sends {"type":"sensor","data":{...}} or {"type":"prediction","data":{...}}
      try {
        const msg = JSON.parse(e.data);
        if (msg && msg.type === "sensor") {
          const data = msg.data;
          setLatest(data);
          setLog((prev) => {
            const next = [data, ...prev];
            if (next.length > 100) next.pop();
            return next;
          });
        } else if (msg && msg.type === "prediction") {
          // Dispatch prediction event to be consumed by StartPrediction
          const prediction = msg.data;
          try {
            window.dispatchEvent(new CustomEvent("live-prediction", { detail: prediction }));
          } catch (err) {
            console.warn("Could not dispatch live-prediction event", err);
          }
        } else {
          // fallback: try parse as sensor payload
          const data = msg;
          if (data && typeof data === "object") {
            setLatest(data);
            setLog((prev) => {
              const next = [data, ...prev];
              if (next.length > 100) next.pop();
              return next;
            });
          }
        }
      } catch (err) {
        console.warn("LiveFeed parse error:", err);
      }
    };
  }, [paused, wsUrl]);

  const scheduleReconnect = useCallback(() => {
    clearTimeout(reconnectTimerRef.current);
    reconnectTimerRef.current = setTimeout(() => {
      connect();
    }, 3000);
  }, [connect]);

  useEffect(() => {
    connect();
    return () => {
      clearTimeout(reconnectTimerRef.current);
      if (wsRef.current) wsRef.current.close();
    };
  }, [connect]);

  const onPause = () => setPaused((p) => !p);
  const onClear = () => { setLog([]); setLatest(null); };
  const onCopy = async () => {
    const payload = JSON.stringify(log, null, 2);
    await navigator.clipboard.writeText(payload);
  };
  const onReconnect = () => connect();

  const metric = (icon, label, value, unit) => (
    <div className={styles.metric} key={label}>
      <div className={styles.metricLabel}>
        <span className={styles.icon}>{icon}</span> {label}
      </div>
      <div className={styles.metricValue}>
        {value ?? "—"}{" "}
        {value != null && unit ? (
          <span className={styles.metricUnit}>{unit}</span>
        ) : null}
      </div>
    </div>
  );

  return (
    <motion.section
      className={styles.card}
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5 }}
    >
      <header className={styles.header}>
        <h2 className={styles.title}>🌿 Live ESP32 Sensor Feed</h2>
        <div
          className={`${styles.status} ${
            status === "online"
              ? styles.statusOnline
              : status === "connecting"
              ? styles.statusConnecting
              : styles.statusOffline
          }`}
        >
          {status === "online"
            ? "🟢 Online"
            : status === "connecting"
            ? "🌀 Connecting..."
            : "🔴 Offline"}
        </div>
      </header>

      <div className={styles.metricsGrid}>
        {metric("🌿", "N", latest?.N, "mg/kg")}
        {metric("🌾", "P", latest?.P, "mg/kg")}
        {metric("🧪", "K", latest?.K, "mg/kg")}
        {metric("🍋", "pH", latest?.pH, "")}
        {metric("🌡️", "Temp", latest?.temp, "°C")}
        {metric("💧", "Humidity", latest?.humidity, "%")}
        {metric("🌱", "Soil", latest?.soil, "%")}
        {metric("🌧️", "Rainfall", latest?.rainfall, "mm")}
        {metric("📍", "Lat", latest?.lat, "")}
        {metric("📍", "Lon", latest?.lon, "")}
      </div>

      <div className={styles.toolbar}>
        <button
          onClick={onPause}
          className={`${styles.btn} ${paused ? styles.btnSecondary : styles.btnPrimary}`}
        >
          {paused ? "▶ Resume" : "⏸ Pause"}
        </button>
        <button onClick={onClear} className={styles.btn}>
          🧹 Clear
        </button>
        <button onClick={onCopy} className={styles.btn}>
          📋 Copy JSON
        </button>
        <button onClick={onReconnect} className={`${styles.btn} ${styles.btnPrimary}`}>
          🔄 Reconnect
        </button>
      </div>

      <div className={styles.log}>
        {log.length === 0 ? (
          <div className={styles.logEmpty}>
            {status === "offline"
              ? "Device not connected."
              : "Waiting for live data..."}
          </div>
        ) : (
          log.map((m, i) => (
            <div key={i} className={styles.logRow}>
              <span className={styles.logTime}>
                {new Date(m.ts || Date.now()).toLocaleTimeString()}
              </span>
              <span className={styles.logText}>
                N:{m.N ?? "—"} P:{m.P ?? "—"} K:{m.K ?? "—"} pH:{m.pH ?? "—"} T:
                {m.temp ?? "—"}°C H:{m.humidity ?? "—"}% S:{m.soil ?? "—"}% R:
                {m.rainfall ?? "—"}mm
              </span>
            </div>
          ))
        )}
      </div>
    </motion.section>
  );
}