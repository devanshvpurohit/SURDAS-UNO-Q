# Arduino UNO Q Deployment Guide: SURDAS

Complete deployment and operations guide for running **SURDAS** on the **Arduino UNO Q** hybrid computing platform.

---

## 🏗️ Hardware Architecture

The Arduino UNO Q features a dual-processor architecture:

1. **Qualcomm Dragonwing QRB2210 MPU (Debian Linux)**:
   - Python 3 application runtime (`surdas_brain.py`)
   - Computer vision pipeline (V4L2 USB camera, YOLOv8 object detection, MiDaS inverse depth estimation)
   - Spatial memory engine (SQLite) & indoor navigation state machine
   - Unified FastAPI telemetry & static Caregiver Dashboard server (`http://0.0.0.0:8000`)
   - Offline wake word & bilingual voice assistant

2. **STM32U585 Microcontroller (Zephyr OS / Arduino Core)**:
   - 50 Hz deterministic I2C IMU sampling & multi-phase fall detection state machine
   - UART GPS NMEA streaming & coordinate processing ($GPRMC, $GPGGA)
   - SPI TFT graphical dashboard driver
   - Hard-wired emergency stop interrupt pin & motor power cutoff relay
   - 10 Hz JSON telemetry streaming across the internal `/dev/ttyRPMSG0` bridge

```
┌────────────────────────────────────────────────────────────────────────┐
│                   Arduino UNO Q Hardware Platform                      │
├────────────────────────────────────┬───────────────────────────────────┤
│    Qualcomm QRB2210 Linux MPU      │     STM32U585 Microcontroller     │
│   • V4L2 USB Camera Pipeline       │   • 50 Hz IMU Fall State Machine  │
│   • YOLOv8 Obstacle Perception     │   • UART GPS Stream ($GPRMC)      │
│   • MiDaS Relative Depth           │   • SPI TFT Display Dashboard     │
│   • Indoor Navigation Engine       │   • Hardware E-Stop Pin & Relay   │
│   • FastAPI & Caregiver Dashboard  │   • 10 Hz Bridge Telemetry Stream │
└─────────────────┬──────────────────┴─────────────────▲─────────────────┘
                  │         Internal Serial Bridge     │
                  └─────────────── /dev/ttyRPMSG0 ─────┘
```

---

## 🔌 Conflict-Free Pinout Configuration

The standard Arduino UNO Q header has been mapped to prevent bus and pin contention:

| Peripheral | Signal | Arduino UNO Q Pin | Protocol | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **I2C IMU (MPU6050 / LSM6DSOX)** | SDA | **A4** (or dedicated SDA) | I2C (400 kHz) | Acceleration & tilt sampling |
| | SCL | **A5** (or dedicated SCL) | I2C (400 kHz) | Pullups on sensor module |
| **UART GPS (NEO-6M / NEO-M8N)** | TX (Module TX) | **D0** (RX / `Serial1`) | UART (9600 baud) | Connects to STM32 RX |
| | RX (Module RX) | **D1** (TX / `Serial1`) | UART (9600 baud) | Connects to STM32 TX |
| **SPI TFT Display (ST7789/ST7735)** | SCK / CLK | **D13** | Hardware SPI | SPI Clock line |
| | MOSI / SDA | **D11** | Hardware SPI | Data Out from MCU |
| | CS | **D10** | Digital Output | Active-low Chip Select |
| | DC / RS | **D9** | Digital Output | Data / Command register |
| | RST | **D6** | Digital Output | Hardware display reset |
| | BLK | **D5** | PWM / Digital Out | Backlight control |
| **Emergency Stop Button** | E-STOP | **D2** (INT0) | External Interrupt | Active-low; instantaneous cutoff |
| **Motor Relay** | RELAY_EN | **D7** | Digital Output | Active-high motor driver enable |
| **Haptic Buzzer** | BUZZER | **D8** | Digital Output | Audible safety & collision alarm |
| **Status LED** | LED | **A0** | Digital Output | Reassigned to A0 to avoid D13 SPI conflict |

---

## 🚀 Quick Deployment (3 Steps)

### Step 1: Clone from GitHub on Arduino UNO Q
SSH into the Arduino UNO Q Debian terminal and clone the repository:
```bash
git clone https://github.com/devanshvpurohit/SURDAS-UNO-Q.git /opt/surdas
cd /opt/surdas
```

### Step 2: Run Automated Provisioning
Run the automated installation script to configure system packages, virtual environment, dependencies, and permissions:
```bash
sudo ./scripts/install_uno_q.sh
```

### Step 3: Flash STM32 Firmware & Start Service
Compile and flash the microcontroller firmware and enable the background systemd service:
```bash
# Flash the STM32 microcontroller:
sudo ./scripts/flash_firmware.sh

# Deploy and start background service:
sudo ./scripts/deploy_service.sh
```

---

## 🖥️ Accessing the Caregiver Dashboard

Once deployed, the FastAPI backend automatically serves the production React Caregiver Dashboard:

- **Local Network URL**: `http://<arduino-uno-q-ip>:8000/`
- **On-board URL**: `http://localhost:8000/`

Features available in the dashboard:
- Live MJPEG video stream from `/video_feed`
- Real-time WebSocket telemetry on `/ws` (FPS, obstacle summary, confidence)
- Standardized IMU fall monitoring from `/imu` (with `STM32 HARDWARE` badge)
- Emergency Stop trigger with physical relay cutoff latching
- Room, landmark, and spatial memory browser

---

## ⚙️ Service Operations & Maintenance

| Action | Command |
| :--- | :--- |
| **Check service status** | `sudo systemctl status surdas` |
| **View real-time logs** | `sudo journalctl -u surdas -f` |
| **Restart system** | `sudo systemctl restart surdas` |
| **Stop system** | `sudo systemctl stop surdas` |

---

## 🧪 Manual Execution & Single-Command Launcher

To start the system manually with a single command without the background systemd service:

```bash
# Standard startup (auto-detects hardware and starts dashboard on :8000):
./start.sh

# Or force simulation mode on a development computer:
./start.sh --sim
```

All 61 hardware-independent test cases can be verified at any time:
```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -v
```
