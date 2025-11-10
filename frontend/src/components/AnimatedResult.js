// frontend/src/components/AnimatedResult.js
import React, { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Text } from "@react-three/drei";
import { motion } from "framer-motion";

// A simple 3D shape (a spinning box)
function SpinningBox() {
  return (
    <mesh rotation={[0.5, 0.7, 0]}>
      <boxGeometry args={[2, 2, 2]} />
      <meshStandardMaterial color="#007aff" />
    </mesh>
  );
}

function AnimatedResult({ data }) {
  const {
    predicted_crop,
    confidence,
    fertilizer_recommendation,
    disease_alerts,
  } = data;

  // Animation variants for framer-motion
  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    visible: { opacity: 1, y: 0, transition: { staggerChildren: 0.1 } },
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -10 },
    visible: { opacity: 1, x: 0 },
  };

  return (
    <motion.div
      style={styles.container}
      variants={containerVariants}
      initial="hidden"
      animate="visible"
    >
      <h2 style={styles.header}>Prediction Successful!</h2>

      {/* 3D Viewer Section */}
      <div style={styles.canvasContainer}>
        <Canvas>
          <Suspense fallback={null}>
            <ambientLight intensity={0.5} />
            <pointLight position={[10, 10, 10]} />
            <SpinningBox />
            <Text
              position={[0, 0, 1.1]}
              fontSize={0.3}
              color="white"
              anchorX="center"
              anchorY="middle"
            >
              {predicted_crop}
            </Text>
            <OrbitControls enableZoom={false} autoRotate={true} />
          </Suspense>
        </Canvas>
      </div>

      {/* Results Details Section */}
      <motion.div style={styles.details} variants={itemVariants}>
        <strong>Predicted Crop:</strong>
        <span>
          {predicted_crop} ({(confidence * 100).toFixed(1)}%)
        </span>
      </motion.div>

      <motion.div style={styles.details} variants={itemVariants}>
        <strong>Fertilizer:</strong>
        <span>{fertilizer_recommendation}</span>
      </motion.div>

      <motion.div style={styles.details} variants={itemVariants}>
        <strong>Disease Alerts:</strong>
        <span>{disease_alerts.join(", ")}</span>
      </motion.div>
    </motion.div>
  );
}

// Basic styling to make it look better
const styles = {
  container: {
    marginTop: "20px",
    padding: "20px",
    backgroundColor: "#f9f9f9",
    borderRadius: "8px",
    border: "1px solid #eee",
    textAlign: "left",
  },
  header: {
    textAlign: "center",
    color: "#333",
    borderBottom: "1px solid #ddd",
    paddingBottom: "10px",
  },
  canvasContainer: {
    height: "250px",
    width: "100%",
    backgroundColor: "#e0e0e0",
    borderRadius: "4px",
    marginBottom: "20px",
  },
  details: {
    display: "flex",
    justifyContent: "space-between",
    padding: "10px 0",
    borderBottom: "1px solid #eee",
    fontSize: "0.95rem",
  },
};

export default AnimatedResult;
