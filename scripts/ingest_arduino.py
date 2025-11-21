#!/usr/bin/env python3
import time, argparse, glob, re, statistics, sys, requests

# Lab reference values (from soil lab results)
LAB = {"n": 125.44, "p": 12.82, "k": 165.36}

# Default fallback location (only used if Arduino DOESN'T send la/lo)
FIELD_LAT = 13.169116
FIELD_LON = 77.558304
FIELD_NAME = "Arduino Field"

# Default calibration logic settings
JUNK_SECONDS = 120
MIN_BUFFER = 10
STABILITY_WINDOW = 20
STD_THRESH = {"n": 5, "p": 2, "k": 5}

POST_SENSOR = "/api/sensor-data"
POST_STATUS = "/api/status"
POST_CALIB = "/api/calibration"

try:
    import serial
except:
    serial = None

# --------------------------------
# HTTP Helper
# --------------------------------
def post(url, payload):
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        print("[HTTP ERROR]", e)

# --------------------------------
# Parse Arduino Lines: *key:value*
#   Example lines from your sketch:
#   *ph:6.80*
#   *n :10*
#   *p :14*
#   *k :29*
#   *t :26.3*
#   *h :61.0*
#   *s :60*
#   *la:13.17*
#   *lo:77.56*
# --------------------------------
STAR_PATTERN = re.compile(r"\*([a-zA-Z]+)\s*:\s*([0-9.]+)\*")

def parse_star_line(line):
    m = STAR_PATTERN.fullmatch(line.strip())
    if not m:
        return None
    key, value = m.groups()
    return key.lower(), float(value)

# --------------------------------
# Detect Serial port on macOS
# --------------------------------
def find_mac_port():
    return glob.glob("/dev/tty.usb*")

# --------------------------------
# Stability check for calibration
# --------------------------------
def stable_enough(buffer):
    if len(buffer) < MIN_BUFFER:
        return False

    tail = buffer[-STABILITY_WINDOW:]

    def std(key):
        vals = [item[key] for item in tail]
        if len(vals) < 3:
            return 999
        return statistics.pstdev(vals)

    return (
        std("n") <= STD_THRESH["n"]
        and std("p") <= STD_THRESH["p"]
        and std("k") <= STD_THRESH["k"]
    )

# --------------------------------
# POST cleaned data
# --------------------------------
def process_cycle(cycle, correction, backend):
    raw_n = cycle["n"]
    raw_p = cycle["p"]
    raw_k = cycle["k"]

    corrected = {
        "n": raw_n * correction["n"],
        "p": raw_p * correction["p"],
        "k": raw_k * correction["k"],
    }

    # read optional extras from cycle (Arduino sends these)
    ph = cycle.get("ph")
    temperature = cycle.get("t")
    humidity = cycle.get("h")
    moisture = cycle.get("s")
    lat = cycle.get("la")
    lon = cycle.get("lo")

    # safe numeric conversion + fallback if Arduino somehow omitted them
    def safe_float(val, default=None):
        if val is None:
            return default
        try:
            return float(val)
        except Exception:
            return default

    payload = {
        "ts": time.time(),
        "raw": {"n": raw_n, "p": raw_p, "k": raw_k},
        "corrected": corrected,
        "calibrated": True,
        "ph": safe_float(ph),
        "temperature": safe_float(temperature),
        "humidity": safe_float(humidity),
        "moisture": safe_float(moisture),
        # frontend expects `lat` / `lon`
        "lat": safe_float(lat, FIELD_LAT),
        "lon": safe_float(lon, FIELD_LON),
        "place_name": FIELD_NAME,
    }

    print("[POSTED]", payload)
    post(backend + POST_SENSOR, payload)

# ---------------------------------------------
# SERIAL MODE – REAL ARDUINO
# ---------------------------------------------
def run_serial(port, baud, backend):
    if serial is None:
        print("pyserial missing → run: pip install pyserial")
        sys.exit(1)

    print("Connecting to", port)
    ser = serial.Serial(port, baudrate=baud, timeout=1)
    time.sleep(2)

    post(backend + POST_STATUS, {"online": True})

    buffer = []
    calibrated = False
    correction = None
    cycle = {}
    start_time = time.time()

    # keys we need AFTER calibration (full sensor frame)
    FULL_KEYS = {"n", "p", "k", "ph", "t", "h", "s", "la", "lo"}

    try:
        while True:
            line = ser.readline().decode(errors="ignore").strip()
            if not line:
                continue

            if line == "*start*":
                cycle = {}
                continue

            parsed = parse_star_line(line)
            if parsed:
                key, value = parsed
                cycle[key] = value

            # -------------------
            #  CALIBRATION PHASE
            # -------------------
            if not calibrated:
                if {"n", "p", "k"}.issubset(cycle.keys()):
                    elapsed = time.time() - start_time
                    buffer.append({"n": cycle["n"], "p": cycle["p"], "k": cycle["k"]})

                    if elapsed < JUNK_SECONDS:
                        continue

                    if stable_enough(buffer):
                        avg_n = statistics.mean([b["n"] for b in buffer])
                        avg_p = statistics.mean([b["p"] for b in buffer])
                        avg_k = statistics.mean([b["k"] for b in buffer])

                        if avg_n <= 0 or avg_p <= 0 or avg_k <= 0:
                            print("[WARN] Skipping calibration (zero values detected)")
                            continue

                        correction = {
                            "n": LAB["n"] / avg_n,
                            "p": LAB["p"] / avg_p,
                            "k": LAB["k"] / avg_k,
                        }

                        calibrated = True
                        print("\n[CALIBRATION COMPLETE]")
                        print("Average raw:", avg_n, avg_p, avg_k)
                        print("Correction:", correction)
                        print()

                        post(
                            backend + POST_CALIB,
                            {
                                "calibrated_at": time.time(),
                                "avg_raw": {"n": avg_n, "p": avg_p, "k": avg_k},
                                "correction": correction,
                                "lab_values": LAB,
                            },
                        )
                # if not calibrated yet, keep reading next lines
                continue

            # -------------------
            #  NORMAL STREAMING
            # -------------------
            if calibrated and FULL_KEYS.issubset(cycle.keys()):
                process_cycle(cycle, correction, backend)
                cycle = {}

    except KeyboardInterrupt:
        print("Stopping ingestion...")

    finally:
        ser.close()
        post(backend + POST_STATUS, {"online": False})

# ---------------------------------------------
# CSV SIMULATION MODE
# ---------------------------------------------
def run_csv(path, backend):
    print("Simulating from CSV →", path)

    with open(path) as f:
        lines = [ln.strip() for ln in f]

    buffer = []
    calibrated = False
    correction = None
    cycle = {}
    start_time = time.time()

    FULL_KEYS = {"n", "p", "k", "ph", "t", "h", "s", "la", "lo"}

    for line in lines:

        if line == "*start*":
            cycle = {}
            continue

        parsed = parse_star_line(line)
        if parsed:
            key, value = parsed
            cycle[key] = value

        if not calibrated:
            if {"n", "p", "k"}.issubset(cycle.keys()):
                elapsed = time.time() - start_time
                buffer.append({"n": cycle["n"], "p": cycle["p"], "k": cycle["k"]})

                if elapsed < JUNK_SECONDS:
                    continue

                if stable_enough(buffer):
                    avg_n = statistics.mean([b["n"] for b in buffer])
                    avg_p = statistics.mean([b["p"] for b in buffer])
                    avg_k = statistics.mean([b["k"] for b in buffer])

                    if avg_n <= 0 or avg_p <= 0 or avg_k <= 0:
                        print("[WARN] Skipping calibration (zero values detected)")
                        continue

                    correction = {
                        "n": LAB["n"] / avg_n,
                        "p": LAB["p"] / avg_p,
                        "k": LAB["k"] / avg_k,
                    }

                    calibrated = True
                    print("\n[CALIBRATION COMPLETE]")
                    print("Average raw:", avg_n, avg_p, avg_k)
                    print("Correction:", correction)
                    print()

                    post(
                        backend + POST_CALIB,
                        {
                            "calibrated_at": time.time(),
                            "avg_raw": {"n": avg_n, "p": avg_p, "k": avg_k},
                            "correction": correction,
                            "lab_values": LAB,
                        },
                    )
            continue

        if calibrated and FULL_KEYS.issubset(cycle.keys()):
            process_cycle(cycle, correction, backend)
            cycle = {}

# ---------------------------------------------
# MAIN
# ---------------------------------------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--port", default=None)
    p.add_argument("--baud", type=int, default=9600)
    p.add_argument("--backend", default="http://127.0.0.1:8000")
    p.add_argument("--test-csv", default=None)
    p.add_argument("--junk-seconds", type=int, default=120)
    p.add_argument("--min-buffer", type=int, default=10)

    args = p.parse_args()

    JUNK_SECONDS = args.junk-seconds if hasattr(args, "junk-seconds") else args.junk_seconds
    MIN_BUFFER = args.min_buffer

    if args.test_csv:
        run_csv(args.test_csv, args.backend)
    else:
        if not args.port:
            ports = find_mac_port()
            if not ports:
                print("⚠ No Arduino detected")
                sys.exit(1)
            args.port = ports[0]

        run_serial(args.port, args.baud, args.backend)