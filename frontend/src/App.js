// frontend/src/App.js
import React from "react";
// We need these components to create different "pages"
import { BrowserRouter, Routes, Route, NavLink } from "react-router-dom";

// --- Import our real components ---
import StartPrediction from "./components/StartPrediction";
import History from "./components/History"; // <-- THIS IS THE NEW LINE

function App() {
  return (
    // BrowserRouter is what handles the URL changes
    <BrowserRouter>
      <div className="App">
        {/* This is our main navigation bar */}
        <nav>
          <NavLink
            to="/"
            className={({ isActive }) => (isActive ? "active" : "")}
          >
            Start
          </NavLink>
          <NavLink
            to="/history"
            className={({ isActive }) => (isActive ? "active" : "")}
          >
            History
          </NavLink>
        </nav>

        {/* This tells the router where to render the "page" */}
        <Routes>
          {/* Route for the main "Start" page */}
          <Route path="/" element={<StartPrediction />} />

          {/* Route for the "History" page */}
          {/* This now renders our real History component */}
          <Route path="/history" element={<History />} />
        </Routes>
      </div>
    </BrowserRouter>
  );
}

export default App;
