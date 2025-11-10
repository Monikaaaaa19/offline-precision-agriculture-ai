# server/db_json.py
"""
Simple offline persistence using a local JSON file.

Handles appending new predictions and reading the history.
"""

import json
import os
from typing import List, Dict, Any
from datetime import datetime

# --- Configuration ---
DB_DIR = "db"
DB_PATH = os.path.join(DB_DIR, "predictions.json")

# Ensure the 'db' directory exists
os.makedirs(DB_DIR, exist_ok=True)


def read_prediction_history() -> List[Dict[str, Any]]:
    """
    Reads the list of all predictions from the JSON file.
    Returns an empty list if the file doesn't exist.
    """
    try:
        with open(DB_PATH, 'r') as f:
            # Read the entire file content
            content = f.read()
            if not content:
                # File is empty, return empty list
                return []
            
            # The file is expected to contain a list of JSON objects
            history = json.loads(content)
            return history
            
    except FileNotFoundError:
        # This is not an error, just means no history yet
        return []
    except json.JSONDecodeError:
        # This means the file is corrupted or empty
        print(f"[ERROR] Could not decode JSON from {DB_PATH}. Returning empty list.")
        return []
    except Exception as e:
        print(f"[ERROR] Error reading prediction history: {e}")
        return []

def save_prediction(prediction_data: Dict[str, Any]) -> bool:
    """
    Saves a single new prediction to the JSON file.
    
    This function reads the existing history, appends the new
    prediction, and writes the entire list back to the file.
    """
    
    # First, read the current history
    history = read_prediction_history()
    
    # Add a server-side timestamp to the record
    prediction_data["id"] = f"pred_{len(history)}"
    prediction_data["saved_at"] = datetime.now().isoformat()
    
    # Append the new prediction
    history.append(prediction_data)
    
    # Write the entire updated list back to the file
    try:
        with open(DB_PATH, 'w') as f:
            # 'indent=4' makes the JSON file human-readable
            json.dump(history, f, indent=4)
        return True
    except IOError as e:
        print(f"[ERROR] Could not write to database file {DB_PATH}: {e}")
        return False
    except Exception as e:
        print(f"[ERROR] Error saving prediction: {e}")
        return False