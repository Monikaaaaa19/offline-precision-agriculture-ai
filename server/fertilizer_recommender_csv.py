# server/fertilizer_recommender_csv.py
"""
Fertilizer recommender that uses the original CSV dataset and all sensor
features (N, P, K, pH, rainfall, temperature, moisture) plus crop name.

API is compatible with existing code:

    recommender.get_recommendation(crop_name, features=features_np.flatten())

It:
  - Filters rows by crop
  - Computes distance in feature space
  - Returns fertilizer_recommendation from the closest row

If something goes wrong or data is missing, it gracefully falls back to the
most common fertilizer or "Normal Fertilizers Needed".
"""

from __future__ import annotations

import os
from typing import Iterable, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Base implementation
# ---------------------------------------------------------------------


class _BaseFertilizerRecommender:
  def __init__(self, csv_path: Optional[str] = None):
    """
    csv_path is optional so this stays compatible with whatever
    models_loader.py is doing. If not provided, we default to:
        ../data/arginode_corrected_fertilizers.csv
    """
    if csv_path is None:
      # server/ -> project root -> data/...
      base_dir = os.path.dirname(os.path.dirname(__file__))
      csv_path = os.path.join(
        base_dir, "data", "arginode_corrected_fertilizers.csv"
      )

    self.csv_path = csv_path

    try:
      df = pd.read_csv(csv_path)
    except Exception as e:
      print(f"[FERTILIZER] Failed to load CSV at {csv_path}: {e}")
      df = pd.DataFrame()

    if df.empty:
      print("[FERTILIZER] WARNING: Fertilizer CSV is empty or missing.")
      self.df = df
      return

    required_cols = [
      "N",
      "P",
      "K",
      "soil_ph",
      "annual_rainfall_mm",
      "avg_temp_c",
      "soil_moisture_pct",
      "crop",
      "fertilizer_recommendation",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
      print(f"[FERTILIZER] WARNING: CSV missing columns: {missing}")

    # Keep only rows that have crop + fertilizer
    df = df.dropna(subset=["crop", "fertilizer_recommendation"])
    self.df = df.reset_index(drop=True)

    print(
      f"[FERTILIZER] Loaded {len(self.df)} rows from "
      f"{os.path.basename(csv_path)}"
    )

  # -------------------------------------------------------------------
  # Public API used by main.py
  # -------------------------------------------------------------------
  def get_recommendation(
      self,
      crop_name: Optional[str],
      features: Optional[Iterable[float]] = None,
  ) -> str:
    """
    crop_name: string crop label (e.g. 'rice', 'cotton')
    features: iterable of numeric features from main.py in order:
        [N, P, K, soil_ph, annual_rainfall_mm, avg_temp_c, soil_moisture_pct, ...]

    Returns the fertilizer_recommendation string from the nearest row
    in the CSV. If anything fails, falls back to the most common
    fertilizer or "Normal Fertilizers Needed".
    """
    if self.df is None or self.df.empty:
      return "Normal Fertilizers Needed"

    crop_name = (crop_name or "").strip().lower()
    df = self.df

    # 1) Filter by crop if possible
    if crop_name:
      subset = df[df["crop"].str.lower() == crop_name]
      if subset.empty:
        subset = df  # fallback to all crops
    else:
      subset = df

    # 2) If we don't have numeric features, just return the most common fert
    if features is None:
      return self._most_common_fertilizer(subset)

    try:
      feat = np.asarray(list(features), dtype=float).ravel()
    except Exception:
      return self._most_common_fertilizer(subset)

    # We only care about the first 7 numbers: N, P, K, soil_ph, rainfall, temp, moisture
    numeric_cols = [
      "N",
      "P",
      "K",
      "soil_ph",
      "annual_rainfall_mm",
      "avg_temp_c",
      "soil_moisture_pct",
    ]

    if feat.size < len(numeric_cols):
      # Not enough features, just fallback
      return self._most_common_fertilizer(subset)

    feat = feat[: len(numeric_cols)]

    # Drop rows missing any of the numeric cols
    valid = subset.dropna(subset=numeric_cols)
    if valid.empty:
      return self._most_common_fertilizer(subset)

    # 3) Compute Euclidean distance in feature space
    try:
      mat = valid[numeric_cols].to_numpy(dtype=float)
      diffs = mat - feat
      dists = np.linalg.norm(diffs, axis=1)
      best_idx = int(dists.argmin())
      fert = valid.iloc[best_idx]["fertilizer_recommendation"]
      return str(fert)
    except Exception as e:
      print(f"[FERTILIZER] Distance calculation failed: {e}")
      return self._most_common_fertilizer(subset)

  # -------------------------------------------------------------------
  # Helpers
  # -------------------------------------------------------------------
  def _most_common_fertilizer(self, df: pd.DataFrame) -> str:
    try:
      counts = df["fertilizer_recommendation"].value_counts()
      if not counts.empty:
        return str(counts.idxmax())
    except Exception:
      pass
    return "Normal Fertilizers Needed"


# ---------------------------------------------------------------------
# Aliases so existing imports keep working (we don't know which one was
# used in models_loader.py, so we provide several common names).
# ---------------------------------------------------------------------


class FertilizerRecommenderCSV(_BaseFertilizerRecommender):
  pass


class CSVFertilizerRecommender(_BaseFertilizerRecommender):
  pass


class FertilizerRecommender(_BaseFertilizerRecommender):
  pass