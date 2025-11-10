// frontend/src/index.js
import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import "./App.css"; // We import our global CSS here

// This is the standard React 18 entry point
const root = ReactDOM.createRoot(document.getElementById("root"));
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);
