import serial
import serial.tools.list_ports
import time

def find_esp32_port():
    """Automatically find the ESP32 serial port."""
    ports = serial.tools.list_ports.comports()
    for port in ports:
        # Common ESP32 VID:PID pairs (update if your board differs)
        if (port.vid == 0x10C4 and port.pid in [0xEA60, 0xEA61]) or \
           (port.vid == 0x1A86 and port.pid == 0x7523) or \
           "esp32" in port.description.lower() or \
           "silicon labs" in port.description.lower():  # CP210x chips
            return port.device
    raise Exception("ESP32 not found! Connect it and try again.")

def read_esp32_data():
    port = find_esp32_port()
    print(f"ESP32 found on port: {port}")
    
    # Open serial connection (common settings for ESP32)
    ser = serial.Serial(
        port=port,
        baudrate=115200,      # Most common baudrate (change if your ESP32 uses different)
        bytesize=serial.EIGHTBITS,
        parity=serial.PARITY_NONE,
        stopbits=serial.STOPBITS_ONE,
        timeout=1,            # Timeout for readline()
        rtscts=False,          # Disable hardware flow control (important for ESP32!)
        dsrdtr=False
    )
    
    # Small delay to let ESP32 boot
    time.sleep(2)
    
    print("Reading data from ESP32... (Press Ctrl+C to stop)")
    try:
        while True:
            if ser.in_waiting > 0:
                line = ser.readline().decode('utf-8').rstrip()  # Read line, decode, remove trailing newline
                print(f"Received: {line}")
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        ser.close()
        print("Serial connection closed.")

if __name__ == "__main__":
    read_esp32_data()