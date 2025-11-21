// src/App.js
import React, { useState } from "react";
import "./App.css";

import LiveFeed from "./components/LiveFeed.jsx";
import StartPrediction from "./components/StartPrediction";
import History from "./components/History";

const TABS = {
  LIVE: "live",
  MANUAL: "manual",
  HISTORY: "history",
};

function App() {
  const [activeTab, setActiveTab] = useState(TABS.LIVE);

  return (
    <div className="app-root">
      {/* ───────────── HEADER ───────────── */}
      <header className="app-header">
        <div>
          <h1>
            Real-Time Offline AI System for Sensor-Integrated Precision
            Agriculture
          </h1>
          <p>
            Real-time crop, fertilizer, and disease-risk advisory from live
            Arduino sensor data
          </p>
        </div>
      </header>

      {/* ───────────── TABS ───────────── */}
      <div className="nav-tabs">
        <button
          type="button"
          className={`nav-tab ${activeTab === TABS.LIVE ? "active" : ""}`}
          onClick={() => setActiveTab(TABS.LIVE)}
        >
          Live Arduino
        </button>

        <button
          type="button"
          className={`nav-tab ${activeTab === TABS.MANUAL ? "active" : ""}`}
          onClick={() => setActiveTab(TABS.MANUAL)}
        >
          Manual Input
        </button>

        <button
          type="button"
          className={`nav-tab ${activeTab === TABS.HISTORY ? "active" : ""}`}
          onClick={() => setActiveTab(TABS.HISTORY)}
        >
          History
        </button>
      </div>

      {/* ───────────── MAIN CONTENT ───────────── */}
      <main className="app-main">
        {/* ---- LIVE TAB ---- */}
        <div
          className={
            activeTab === TABS.LIVE
              ? "tab-panel tab-panel-active"
              : "tab-panel tab-panel-hidden"
          }
        >
          <LiveFeed />
        </div>

        {/* ---- MANUAL TAB ---- */}
        <div
          className={
            activeTab === TABS.MANUAL
              ? "tab-panel tab-panel-active"
              : "tab-panel tab-panel-hidden"
          }
        >
          <StartPrediction />
        </div>

        {/* ---- HISTORY TAB ---- */}
        <div
          className={
            activeTab === TABS.HISTORY
              ? "tab-panel tab-panel-active"
              : "tab-panel tab-panel-hidden"
          }
        >
          <History />
        </div>
      </main>
    </div>
  );
}

export default App;
