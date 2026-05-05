# 🚦 Density-Based Traffic Management System using YOLOv8

An intelligent traffic management system that uses YOLOv8 computer vision to estimate vehicle density in real-time and dynamically control traffic lights. This project includes a Flask-based web interface for managing a 4-lane intersection and integrates an ESP32 microcontroller to control physical LED traffic lights.

## ✨ Features

- **Real-Time Object Detection:** Utilizes YOLOv8 to detect and count vehicles (bicycles, buses, cars, motorbikes).
- **Dynamic Traffic Control:** Smart logic switches green lights based on real-time vehicle presence and a maximum time allocation (30 seconds).
- **Web Dashboard:** A Flask web application to upload lane videos, start/stop the system, and monitor real-time video feeds with bounding boxes.
- **Hardware Integration:** Connects with an ESP32 microcontroller via HTTP to toggle physical Red/Green LEDs for 4 individual lanes.
- **Parallel Processing:** Uses multi-threading in Python for smooth video playback and rapid HTTP commands to the ESP32.

## 📋 Prerequisites

### Software Dependencies
- Python 3.8+
- PyTorch
- Ultralytics (YOLOv8)
- OpenCV (`opencv-python`)
- Flask, Flask-CORS
- Requests

### Hardware Requirements
- 1× ESP32 Development Board (e.g., ESP32 DevKit)
- 8× LEDs (4 Red, 4 Green)
- 8× 220Ω Resistors
- Breadboard and Jumper Wires
- Micro-USB Cable

## 🚀 Installation & Setup

### 1. Software Setup

Clone the repository and install the required Python packages:

```bash
git clone https://github.com/your-username/Density-Based-Traffic-Management-System-Using-YOLO-V8.git
cd Density-Based-Traffic-Management-System-Using-YOLO-V8
pip install -r requirements.txt
```

**YOLOv8 Model:** Ensure you have your trained YOLOv8 model weights saved at the following path relative to the project root:
`runs/detect/yolo26s_train/weights/best.pt`

### 2. Hardware Setup (ESP32)

To simulate the traffic lights physically, you'll need to wire 8 LEDs to the ESP32 and upload the controller sketch.

- **Wiring:** See the complete wiring diagram in ESP32_WIRING.md.
- **Flashing the ESP32:** Read ESP32_SETUP.md for step-by-step instructions on compiling and uploading the `.ino` sketch using the Arduino IDE. 

### 3. ESP32 Configuration

Once your ESP32 is running and connected to WiFi, it will display an IP address in the Arduino Serial Monitor. Update the `esp32_config.json` file in the project root with this IP address:

```json
{
  "esp32_ip": "192.168.1.100", 
  "esp32_port": 80,
  "enabled": true,
  "timeout": 2
}
```
*(Note: If you want to test the software without the hardware connected, simply set `"enabled": false` in this file).*

## 💻 Usage

1. **Start the Flask Server:**
   ```bash
   python app.py
   ```
2. **Open the Dashboard:** Open your web browser and navigate to `http://localhost:5000`.
3. **Upload Videos:** Upload sample traffic footage for all 4 lanes.
4. **Run the System:** Click "Start" on the dashboard. The system will:
   - Begin processing video frame-by-frame.
   - Detect vehicles and allocate a green light to the active lane.
   - Send an HTTP signal to your ESP32 to switch physical LEDs.
   - Automatically rotate lanes if time expires (30s max) or if a lane becomes empty.

## 🔄 How the Logic Works

1. The **Flask app** maintains a background thread for traffic logic processing.
2. **YOLOv8** processes frames every 2 seconds for the currently active lane to count bounding boxes matching vehicle classes.
3. If the vehicle count is `> 0`, the lane remains green up to a `MAX_GREEN_TIME` of 30 seconds.
4. If no vehicles are detected, or the maximum time is reached, the active lane shifts to the next lane (1 → 2 → 3 → 4 → Loop).
5. A state change triggers an HTTP request to the **ESP32**, which toggles the specific GPIO pins hooked to the red/green LEDs.

## 📁 Project Structure

```text
📂 Density-Based-Traffic-Management-System-Using-YOLO-V8
 ├── app.py                  # Main Flask backend and traffic logic
 ├── esp32_config.json       # IP configuration for ESP32 connectivity
 ├── ESP32_SETUP.md          # Guide for setting up ESP32 software
 ├── ESP32_WIRING.md         # Guide for wiring ESP32 physical pins
 ├── /templates              # HTML files for the Web interface
 ├── /uploads                # Directory for user-uploaded videos
 └── /runs/detect/...        # Directory containing trained YOLOv8 weights
```

## ⚠️ Troubleshooting

- **ESP32 timeouts / lagging lights:** Ensure your computer and ESP32 are on the same 2.4GHz WiFi network. Check if the IP changed.
- **Model fails to load:** Ensure PyTorch is correctly installed and the file path `runs/detect/yolo26s_train/weights/best.pt` exactly matches your model location.

---
*Developed as an academic project showcasing Computer Vision and IoT Integration.*