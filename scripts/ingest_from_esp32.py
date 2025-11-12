# scripts/ingest_from_esp32.py
"""
Reads text sensor data from an ESP32 (auto-detecting the port),
parses it into JSON, and posts it to the local prediction server.
"""
import serial
import serial.tools.list_ports
import time
import requests
import json
import sys
import argparse

# --- Configuration ---
DEFAULT_POST_URL = "http://127.0.0.1:8000/predict_crop"
DEFAULT_BAUD_RATE = 115200

# This "map" translates the ESP32's text labels (like 'Temp')
# into the JSON keys your server expects (like 'temperature').
KEY_MAP = {
    'N': 'N',
    'P': 'P',
    'K': 'K',
    'pH': 'pH',
    'Temp': 'temperature',
    'Hum': 'humidity',
    'Rain': 'rainfall',
    'Lat': 'latitude',
    'Lon': 'longitude'
}
# The 9 keys the server *requires* to make a prediction
REQUIRED_KEYS = [
    'N', 'P', 'K', 'pH', 'temperature', 
    'humidity', 'rainfall', 'latitude', 'longitude'
]

def find_esp32_port():
    """Automatically find the ESP32 serial port."""
    print("[INFO] Searching for ESP32...")
    ports = serial.tools.list_ports.comports()
    for port in ports:
        if (port.vid == 0x10C4 and port.pid in [0xEA60, 0xEA61]) or \
           (port.vid == 0x1A86 and port.pid == 0x7523) or \
           "esp32" in port.description.lower() or \
           "silicon labs" in port.description.lower():
            print(f"[INFO] ESP32 found on port: {port.device}")
            return port.device
            
    print("[WARN] ESP32 not found. Will not be able to listen.")
    return None

def parse_and_post_line(line: str, url: str):
    """
    Parses the ESP32 text line (e.g., "N: 90, P: 45...")
    into a JSON payload and posts it to the server.
    """
    try:
        payload = {}
        # 1. Split the line into individual parts: "N: 90", " P: 45", ...
        parts = line.split(',')
        
        if len(parts) < len(REQUIRED_KEYS):
            print(f"[WARN] Incomplete line received. Expected {len(REQUIRED_KEYS)} parts, got {len(parts)}. Line: '{line}'")
            return

        # 2. Parse each part
        for part in parts:
            kv = part.split(':')
            if len(kv) != 2:
                print(f"[WARN] Malformed part, skipping: '{part}'")
                continue
                
            key_from_esp = kv[0].strip()   # e.g., "Temp"
            value_from_esp = float(kv[1].strip()) # e.g., 28.5
            
            # 3. Translate the key
            if key_from_esp in KEY_MAP:
                json_key = KEY_MAP[key_from_esp]
                payload[json_key] = value_from_esp
        
        # 4. Check if we have all required data for the server
        if all(key in payload for key in REQUIRED_KEYS):
            payload['place_name'] = "Live ESP32 Data"
            
            print(f"[INFO] Parsed JSON payload: {payload}")
            
            # 5. Post to the server
            response = requests.post(url, json=payload, timeout=5)
            
            if response.status_code == 200:
                print("[SUCCESS] Data posted. Server responded:")
                print(f"  Crop: {response.json().get('predicted_crop')}")
                print(f"  State: {response.json().get('received_data', {}).get('state')}")
            else:
                print(f"[ERROR] Server returned status: {response.status_code}")
        else:
            print(f"[WARN] Parsed data is missing required fields. Not posting.")

    except ValueError:
        print(f"[ERROR] Could not convert a value to a number. Line: '{line}'")
    except requests.exceptions.ConnectionError:
        print(f"[ERROR] Could not connect to server at {url}")
        print("[HINT] Is the 'uvicorn' server running in Terminal 1?")
    except Exception as e:
        print(f"[ERROR] Failed to parse or post line: {e}")


def read_esp32_data(url):
    """This function runs the REAL device listener."""
    try:
        port = find_esp32_port()
        if not port:
            raise Exception("ESP32 not found. Aborting.")
            
        print(f"ESP32 found on port: {port}")
    except Exception as e:
        print(e)
        return

    ser = serial.Serial(
        port=port,
        baudrate=DEFAULT_BAUD_RATE,
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1,
        rtscts=False,
        dsrdtr=False
    )
    
    time.sleep(2)
    print(f"Reading data from ESP32 on {port}... (Press Ctrl+C to stop)")
    
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip()
                
                # Check if the line looks like our data
                if "N:" in line and "P:" in line and "K:" in line:
                    print(f"Received: {line}")
                    # Parse the line and send it to the server
                    parse_and_post_line(line, url)
                else:
                    # Show other ESP32 messages (like "Booting...")
                    print(f"[ESP32_MSG] {line}")
                
    except KeyboardInterrupt:
        print("\nStopped by user.")
    except serial.SerialException:
        print(f"[ERROR] Device on {port} disconnected.")
    finally:
        ser.close()
        print("Serial connection closed.")

if __name__ == "__main__":
    # Check for 'requests' library
    try:
        import requests
    except ImportError:
        print("[ERROR] The 'requests' library is required.")
        print("[HINT] Please install it: pip install requests")
        sys.exit(1)

    read_esp32_data(DEFAULT_POST_URL)