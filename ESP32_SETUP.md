# ESP32 Traffic Light Hardware Setup Guide

This guide will help you set up the ESP32 hardware to control physical LED traffic lights for the 4-lane intersection system.

## 📋 Components Required

- ESP32 Development Board (ESP32 DevKit)
- 8 LEDs (4 Red + 4 Green)
- 8 × 220Ω Resistors
- Breadboard and jumper wires
- USB cable (Micro-B) for ESP32
- Laptop/PC with Arduino IDE

## 🔌 Wiring Diagram

### LED Connections

Connect each LED as follows:

| Lane | LED Color | ESP32 Pin | GPIO | Long Leg (+) → Resistor → Pin | Short Leg (-) → GND |
|------|-----------|-----------|------|-------------------------------|---------------------|
| A    | Red       | D13       | GPIO13 | Resistor → GPIO13            | GND                 |
| A    | Green     | D14       | GPIO14 | Resistor → GPIO14            | GND                 |
| B    | Red       | D27       | GPIO27 | Resistor → GPIO27            | GND                 |
| B    | Green     | D26       | GPIO26 | Resistor → GPIO26            | GND                 |
| C    | Red       | D25       | GPIO25 | Resistor → GPIO25            | GND                 |
| C    | Green     | D18       | GPIO18 | Resistor → GPIO18            | GND                 |
| D    | Red       | D19       | GPIO19 | Resistor → GPIO19            | GND                 |
| D    | Green     | D21       | GPIO21 | Resistor → GPIO21            | GND                 |

### Connection Steps

1. **Place 220Ω resistors** in series with each LED's positive (long) leg
2. **Connect resistor ends** to the respective ESP32 GPIO pins
3. **Connect all LED negative legs** to ESP32 GND (you can use a common GND rail on the breadboard)
4. **Power ESP32** via USB cable

**Important Notes:**
- LEDs have polarity: Long leg = Positive (+), Short leg = Negative (-)
- Always use resistors (220Ω) to protect LEDs from excessive current
- ESP32 GPIO pins are 3.3V logic level
- Never connect LEDs directly to pins without resistors

## 💻 Software Setup

### Step 1: Install Arduino IDE

1. Download and install [Arduino IDE](https://www.arduino.cc/en/software) (version 1.8.x or 2.x)
2. Open Arduino IDE

### Step 2: Install ESP32 Board Support

1. Open Arduino IDE Preferences
2. Add this URL to "Additional Board Manager URLs":
   ```
   https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
   ```
3. Go to **Tools → Board → Boards Manager**
4. Search for "ESP32" and install **"esp32 by Espressif Systems"**

### Step 3: Configure Arduino IDE for ESP32

1. Connect ESP32 to your computer via USB
2. Go to **Tools → Board** and select **"ESP32 Dev Module"**
3. Select the correct COM port: **Tools → Port → [Your ESP32 Port]**
4. Set these settings:
   - **Upload Speed**: 115200
   - **CPU Frequency**: 240MHz
   - **Flash Frequency**: 80MHz
   - **Flash Mode**: QIO
   - **Flash Size**: 4MB
   - **Partition Scheme**: Default 4MB

### Step 4: Install Required Libraries

1. Go to **Sketch → Include Library → Manage Libraries**
2. Search and install:
   - **WiFi** (usually included by default)
   - **WebServer** (usually included by default)

### Step 5: Upload Code to ESP32

1. Open the file: `esp32_traffic_controller/esp32_traffic_controller.ino`
2. **IMPORTANT**: Update WiFi credentials in the code:
   ```cpp
   const char* ssid = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
   Replace `YOUR_WIFI_SSID` and `YOUR_WIFI_PASSWORD` with your actual WiFi network credentials.

3. Click **Verify** (✓) to compile the code
4. Click **Upload** (→) to upload to ESP32
5. Open **Serial Monitor** (Tools → Serial Monitor) at 115200 baud rate
6. You should see:
   - "Connecting to WiFi..."
   - "WiFi connected!"
   - **IP address**: (e.g., 192.168.1.100) ← **Copy this IP!**

### Step 6: Test ESP32

1. Open a web browser
2. Navigate to: `http://[ESP32_IP_ADDRESS]/test`
   - Example: `http://192.168.1.100/test`
   - This will test all LEDs in sequence

3. Test individual lanes:
   - `http://[ESP32_IP]/control?lane=1&state=GREEN` - Set Lane 1 to Green
   - `http://[ESP32_IP]/control?lane=1&state=RED` - Set Lane 1 to Red
   - Repeat for lanes 2, 3, 4

4. Check status:
   - `http://[ESP32_IP]/status` - Get current LED states

### Step 7: Configure Flask App

1. Edit `esp32_config.json`:
   ```json
   {
     "esp32_ip": "192.168.1.100",
     "esp32_port": 80,
     "enabled": true,
     "timeout": 2
   }
   ```
   Replace `"192.168.1.100"` with your ESP32's actual IP address from Step 5.

2. If ESP32 is not connected, you can disable it by setting:
   ```json
   {
     "enabled": false
   }
   ```

### Step 8: Install Python Dependencies

Make sure you have the required Python packages:
```bash
pip install -r requirements.txt
```

The `requests` library is already added for ESP32 communication.

### Step 9: Run the System

1. Start the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser to: `http://localhost:5000`

3. Upload videos for all 4 lanes

4. Click "Run Project"

5. The ESP32 LEDs should now automatically control based on the active lane!

## 🔧 Troubleshooting

### ESP32 Won't Connect to WiFi

- **Check credentials**: Ensure WiFi SSID and password are correct
- **2.4GHz only**: ESP32 only supports 2.4GHz WiFi (not 5GHz)
- **Signal strength**: Move ESP32 closer to router
- **Serial Monitor**: Check Serial Monitor for error messages

### ESP32 Shows IP Address but Flask Can't Connect

- **Check IP address**: Verify ESP32 IP in Serial Monitor
- **Network connectivity**: Ensure PC and ESP32 are on the same WiFi network
- **Firewall**: Check if firewall is blocking connections
- **Test manually**: Try accessing `http://[ESP32_IP]/status` in browser

### LEDs Don't Light Up

- **Check wiring**: Verify all connections are correct
- **Polarity**: Ensure LED long leg (+) is connected via resistor to GPIO pin
- **GND**: Ensure all LED short legs (-) are connected to GND
- **Resistors**: Verify 220Ω resistors are in place
- **Power**: Ensure ESP32 is powered via USB

### LEDs Always Off/On

- **Test LEDs**: Use `/test` endpoint to verify wiring
- **Check code**: Verify correct GPIO pins in code match your wiring
- **Serial Monitor**: Check for error messages

### Flask Shows ESP32 Connection Errors

- **Check config**: Verify `esp32_config.json` has correct IP
- **Network**: Ensure PC and ESP32 on same network
- **ESP32 running**: Verify ESP32 is powered and connected to WiFi
- **Disable if needed**: Set `"enabled": false` in config to run without hardware

## 📡 ESP32 API Endpoints

The ESP32 provides these HTTP endpoints:

- `GET /` - Home page with status
- `GET /status` - Get current LED states (JSON)
- `GET /control?lane=X&state=RED|GREEN` - Control LEDs
- `GET /test` - Test all LEDs in sequence

## 🔄 How It Works

1. Flask app detects vehicles using YOLO v8
2. Traffic light logic determines which lane should be green
3. Flask app sends HTTP GET request to ESP32: `/control?lane=X&state=GREEN`
4. ESP32 receives command and sets corresponding LEDs (Red/Green)
5. Other lanes are automatically set to RED

## 📝 Additional Notes

- **Access Point Mode**: If WiFi connection fails, ESP32 creates an AP named "ESP32-Traffic-Light" with password "12345678"
- **LED Current**: Each LED draws ~10-20mA, total ~80-160mA (well within ESP32 limits)
- **Power**: ESP32 can be powered via USB or external 5V supply
- **Range**: ESP32 WiFi range is typically 30-100 meters indoors

## ✅ Verification Checklist

- [ ] All 8 LEDs wired correctly with resistors
- [ ] ESP32 connected to WiFi
- [ ] ESP32 IP address noted
- [ ] `esp32_config.json` updated with correct IP
- [ ] Test endpoints work in browser
- [ ] Flask app can connect to ESP32
- [ ] LEDs respond to Flask app commands

---

**Need Help?** Check the Serial Monitor output for detailed debugging information.

