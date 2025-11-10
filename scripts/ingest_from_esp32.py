# scripts/ingest_from_esp32.py
"""
Script to read sensor JSON from an ESP32 (via Serial or UDP)
and post it to the local prediction server.

Also includes a simulator mode for testing.

--- ESP32 JSON Format Example ---
(Must be a single line of text sent over serial)
{
  "device_id": "esp32_01", "timestamp": "2025-11-09T09:00:00Z",
  "N": 34, "P": 12, "K": 45, "pH": 6.5, "temperature": 28.6,
  "humidity": 73.5, "rainfall": 2.4, "latitude": 13.021,
  "longitude": 80.198, "place_name": "Field A",
  "polygon": [
    {"lat":13.021,"lng":80.198},
    {"lat":13.022,"lng":80.199},
    {"lat":13.0215,"lng":80.1995}
  ]
}
---------------------------------
"""
import argparse
import json
import serial
import requests # This library is for making HTTP requests
import time
import sys

# --- Configuration ---
DEFAULT_POST_URL = "http://127.0.0.1:8000/predict_crop"
DEFAULT_SERIAL_PORT = "/dev/tty.usbserial-0001" # Common for macOS
DEFAULT_BAUD_RATE = 115200

# --- THIS IS THE UPDATED BLOCK ---
# We've added latitude and longitude to the simulated data
SIMULATED_JSON_DATA = {
    "N": 34, "P": 12, "K": 45, "pH": 6.5, "temperature": 28.6,
    "humidity": 73.5, "rainfall": 2.4,
    "latitude": 13.021,    # <-- ADDED
    "longitude": 80.198,   # <-- ADDED
    "place_name": "Field A (Simulated)",
    "polygon": [
      {"lat":13.021,"lng":80.198},
      {"lat":13.022,"lng":80.199},
      {"lat":13.0215,"lng":80.1995}
    ]
}
# ---------------------------------

def post_to_server(data_payload: dict, url: str):
    """Sends the given data payload to the prediction server."""
    try:
        # The 'json' parameter automatically sets
        # 'Content-Type: application/json'
        response = requests.post(url, json=data_payload, timeout=5)
        
        # Check if the request was successful
        if response.status_code == 200:
            print("[INFO] Successfully posted data. Server response:")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"[ERROR] Server returned status code {response.status_code}:")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print(f"\n[ERROR] Connection refused.")
        print(f"[ERROR] Could not connect to the server at {url}")
        print("[HINT] Is the 'uvicorn server.main:app' running?\n")
    except requests.exceptions.Timeout:
        print(f"[ERROR] Request timed out when connecting to {url}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")

def start_serial_listener(port: str, baud: int, url: str):
    """Listens on a serial port for newline-terminated JSON."""
    print(f"[INFO] Starting serial listener on {port} at {baud} baud.")
    print(f"[INFO] Posting data to: {url}")
    print("[INFO] Waiting for data from ESP32... (Press Ctrl+C to stop)")
    
    try:
        # Open the serial port
        ser = serial.Serial(port, baud, timeout=1)
        
        while True:
            try:
                # Read one line from the serial port
                # .strip() removes whitespace and newline characters
                line = ser.readline().decode('utf-8').strip()
                
                if line:
                    print(f"\n[DATA] Received line: {line}")
                    
                    # Try to parse the line as JSON
                    try:
                        data_payload = json.loads(line)
                        # Post the valid JSON to the server
                        post_to_server(data_payload, url)
                        
                    except json.JSONDecodeError:
                        print(f"[WARN] Received invalid JSON. Ignoring line.")
                
            except serial.SerialException as e:
                print(f"[ERROR] Serial error: {e}. Reconnecting...")
                ser.close()
                time.sleep(5)
                ser.open()
            except KeyboardInterrupt:
                print("\n[INFO] Stopping serial listener.")
                break
                
    except serial.SerialException as e:
        print(f"[CRITICAL] Could not open serial port {port}: {e}")
        print("[HINT] Is the ESP32 plugged in? Is the port name correct?")
        print("[HINT] On macOS, find ports with: ls /dev/tty.*")
    finally:
        if 'ser' in locals() and ser.is_open:
            ser.close()

def run_simulator(url: str):
    """Sends the simulated data packet to the server."""
    print("[INFO] --- Running in Simulator Mode ---")
    print(f"[INFO] Sending one test packet to: {url}")
    print("[INFO] Data to be sent:")
    print(json.dumps(SIMULATED_JSON_DATA, indent=2))
    
    post_to_server(SIMULATED_JSON_DATA, url)
    print("[INFO] --- Simulator run complete ---")

def main():
    parser = argparse.ArgumentParser(description="ESP32 Data Ingest Script")
    
    # --- Serial Port Arguments ---
    parser.add_argument(
        "--serial-port",
        type=str,
        default=DEFAULT_SERIAL_PORT,
        help="The serial port to listen on (e.g., /dev/tty.usbserial-XXXX)"
    )
    parser.add_argument(
        "--baud",
        type=int,
        default=DEFAULT_BAUD_RATE,
        help="The baud rate for the serial connection"
    )
    
    # --- Server URL Argument ---
    parser.add_argument(
        "--post-url",
        type=str,
        default=DEFAULT_POST_URL,
        help="The URL of the prediction server endpoint"
    )
    
    # --- Simulator Mode ---
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run in simulator mode. Sends one test packet and exits."
    )
    
    args = parser.parse_args()
    
    # Check if the requests library is installed (it's not in reqs)
    try:
        import requests
    except ImportError:
        print("[ERROR] The 'requests' library is required for this script.")
        print("[HINT] Please install it by running:")
        print("pip install requests")
        sys.exit(1)

    if args.simulate:
        run_simulator(args.post_url)
    else:
        start_serial_listener(args.serial_port, args.baud, args.post_url)

if __name__ == "__main__":
    main()