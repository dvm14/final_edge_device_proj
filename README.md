# Automatic Pill Dispenser with Voice Control and Computer Vision

## Project Overview

An AI-powered medication dispenser that uses voice commands to receive instructions and computer vision to accurately count and dispense the correct number of pills. The system combines speech recognition, object detection, and servo control to create a hands-free, assistive device for daily medication management.

**Motivation:** Taking daily vitamins is easy to forget, especially when they're not readily available on-the-go. This device automates the dispensing process and ensures the correct dosage through real-time visual verification.

---

## Components Used

### Hardware
- **Raspberry Pi** (with GPIO)
- **PiCamera** - For capturing images and counting pills
- **Servo Motor (SG90)** - For tipping the pill bottle to dispense pills
- **I2C LCD Display (16x2)** - For user feedback and system status
- **Microphone** - For voice command input
- **Pill Container** - Flat, large container for pills to land in

### Software
- **Python 3**
- **YOLOv5** - Object detection model for counting pills (trained on Google Colab, deployed to Pi)
- **Vosk** - Offline speech recognition
- **picamzero** - Camera interface
- **gpiozero** - GPIO control for servo, sensors, and button
- **PyTorch** - Deep learning framework for YOLOv5
- **OpenCV** - Image processing
- **sounddevice** - Audio recording

---

## Project Structure

```
edge_device_proj/
├── README.md                               # This file
├── final_main.py                           # Main script (voice + camera)
├── capture_train_imgs.py                   # Script to collect training images for YOLO
├── train_pill_yolov5_colab.ipynb           # Google Colab notebook for training YOLOv5
├── best.pt                                 # Trained YOLOv5 weights (copied from Colab)
├── training_images/                        # Collected images for YOLOv5 training
│   └── pill_*.jpg                          # ~150+ images with varying pill counts
├── yolov5/                                 # YOLOv5 repository (cloned, used for inference)
│   ├── detect.py                           # YOLOv5 inference script
│   └── ...
├── Pill-Detection-2/                       # Roboflow dataset (annotated images)
│   ├── data.yaml                           # Dataset configuration
│   ├── train/                              # Training images + labels
│   ├── valid/                              # Validation images + labels
│   └── test/                              # Test images + labels
└── models/                                # Vosk speech models (one level up: ../models/)
    └── vosk-model-small-en-us-0.15/       # English speech recognition model
```

---

## Setup Instructions

### 1. Hardware Setup

**GPIO Pin Connections:**
- **Servo Motor:**
  - Signal → GPIO 16 (BCM)
  - VCC → 5V
  - GND → GND

- **I2C LCD Display:**
  - SDA → GPIO 2 (SDA)
  - SCL → GPIO 3 (SCL)
  - VCC → 5V
  - GND → GND

- **PiCamera:**
  - Connect to camera port on Raspberry Pi
  - Position camera above pill container

### 2. Software Installation (Raspberry Pi)

#### Install System Dependencies
```bash
sudo apt update
sudo apt install python3-pip python3-venv portaudio19-dev python3-opencv pigpio i2c-tools

# Start pigpio daemon (for servo control)
sudo pigpiod
```

#### Create Virtual Environment
```bash
cd ~/Desktop/edge_device_proj
python3 -m venv .venv
source .venv/bin/activate
```

#### Install Python Packages
```bash
# Core dependencies
pip install gpiozero picamzero lcd-i2c

# Computer vision and ML
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install opencv-python pillow pyyaml requests tqdm matplotlib seaborn pandas scipy ultralytics

# Speech recognition
pip install vosk sounddevice numpy
```

#### Clone YOLOv5 Repository (for inference only)
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..
```

#### Download Vosk Model
```bash
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ..
```

---

## Data Collection and Model Training

> **Note:** Model training is done on Google Colab (free GPU), not on the Raspberry Pi. The Pi is only used for inference. Training on the Pi takes 4+ hours; Colab takes ~10 minutes.

### Step 1: Collect Training Images
```bash
python capture_train_imgs.py
# Press ENTER to capture images with 0, 1, 2, 3, 4, 5+ pills
# Capture ~150-200 images total
```

### Step 2: Annotate Images on Roboflow
1. Go to [roboflow.com](https://roboflow.com) and create an account
2. Create new project: "Pill Detection" (Object Detection)
3. Upload images from `training_images/`
4. Draw bounding boxes around each pill
5. Use auto-labeling after ~30 manual annotations
6. Generate dataset version (YOLOv5 PyTorch format)
7. Note your API key and workspace/project names

### Step 3: Train on Google Colab
1. Open `train_pill_yolov5_colab.ipynb` in [colab.research.google.com](https://colab.research.google.com)
2. Go to `Runtime > Change runtime type > T4 GPU`
3. Fill in your Roboflow API key and project details in cell 3
4. Run all cells in order
5. The final cell downloads `best.pt` to your computer

A well-trained model should reach **mAP50 ≥ 0.90** by epoch 100.

### Step 4: Deploy Weights to Raspberry Pi
```bash
# Copy best.pt from your laptop to the project root on the Pi
scp ~/Downloads/best.pt dvm14@raspberrypi.local:~/Desktop/edge_device_proj/
```

---

## Running the Project

### Manual Start
```bash
source .venv/bin/activate
sudo pigpiod  # Start pigpio daemon if not already running
python final_main.py
```

**Usage:**
1. LCD shows: "Ready. Say: Hi there..."
2. User says: "Hi there, I need 3 pills"
3. System acknowledges and starts dispensing
4. Camera counts pills in real-time
5. LCD updates: "Pouring... 2/3 pills"
6. Stops when target count reached
7. LCD shows: "Done! 3 pills dispensed"
8. Returns to ready state after 5 seconds

### Autostart on Boot (Recommended)

The project is configured to start automatically when the Pi is powered on using systemd services.

#### pigpiod service
```bash
# Create /etc/systemd/system/pigpiod.service
sudo nano /etc/systemd/system/pigpiod.service
```
```ini
[Unit]
Description=Pigpio Daemon
After=network.target

[Service]
Type=forking
ExecStart=/usr/local/bin/pigpiod
ExecStop=/bin/systemctl kill pigpiod

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable pigpiod
sudo systemctl start pigpiod
```

#### pill-dispenser service
```bash
sudo nano /etc/systemd/system/pill-dispenser.service
```
```ini
[Unit]
Description=Pill Dispenser
After=network.target pigpiod.service

[Service]
ExecStartPre=/bin/sleep 10
ExecStart=/home/dvm14/Desktop/edge_device_proj/.venv/bin/python /home/dvm14/Desktop/edge_device_proj/final_main.py
WorkingDirectory=/home/dvm14/Desktop/edge_device_proj
User=dvm14
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload
sudo systemctl enable pill-dispenser.service
sudo systemctl start pill-dispenser.service
```

Once configured, simply plug in the Pi and the dispenser will start automatically within ~20 seconds.

#### Useful service commands
```bash
# Check status
sudo systemctl status pill-dispenser.service

# View logs
journalctl -u pill-dispenser.service -f

# Stop manually
sudo systemctl stop pill-dispenser.service

# Restart
sudo systemctl restart pill-dispenser.service
```

---

## Configuration

### Adjustable Parameters

In `final_main.py`:

```python
# Pin assignments
SERVO_PIN = 16

# Model paths
YOLO_MODEL_PATH = "./best.pt"  # Place best.pt in the project root
VOSK_MODEL_DIR = "../models/vosk-model-small-en-us-0.15"

# Voice recognition settings
WAKE_PHRASE = "hi there"
RECORD_SECONDS = 5

# Servo control
increment = 4    # Angle increment per step (degrees)
max_angle = 176  # Maximum tipping angle
```

---

## Testing the YOLO Model
```bash
# Test inference on a single image
python yolov5/detect.py \
  --weights best.pt \
  --source training_images/pill_20260329_141820.jpg \
  --conf 0.4

# Results saved to yolov5/runs/detect/
```

---

## Troubleshooting

### Camera Issues
```bash
# Check if camera is detected
vcgencmd get_camera

# Enable camera in raspi-config
sudo raspi-config
# Interface Options → Camera → Enable
```

### Servo Jitter
```bash
# Make sure pigpio daemon is running
sudo systemctl status pigpiod

# Start it if not running
sudo systemctl start pigpiod
```

### Microphone Not Working
```bash
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test recording
python -c "import sounddevice as sd; import numpy as np; audio = sd.rec(16000, samplerate=16000, channels=1); sd.wait(); print('Recording complete')"
```

### Model Loading Errors
- Ensure model path in `final_main.py` matches where `best.pt` was copied
- Verify the file exists: `ls yolov5/runs/train/pill_detection_colab/weights/`

### LCD Not Displaying
```bash
# Check I2C devices
sudo i2cdetect -y 1
```

### Script Not Starting on Boot
```bash
# Check both services are active
sudo systemctl status pigpiod
sudo systemctl status pill-dispenser.service

# View detailed logs
journalctl -xeu pill-dispenser.service --no-pager | tail -30
```

---

## Project Evolution

This project builds upon a midterm version that used an ultrasonic distance sensor for binary pill detection (present/absent). The final version replaces the ultrasonic sensor with computer vision for precise pill counting and adds voice control for a fully hands-free experience.

### Key Upgrades
- **Added PiCamera:** Real-time pill counting with YOLOv5 object detection
- **Added Microphone:** Voice command input with Vosk speech recognition
- **Wake phrase:** "Hi there" triggers listening mode
- **Pill counting:** Stops dispensing at least target count
- **Enhanced feedback:** Real-time count display on LCD (e.g., "Pouring... 2/3 pills")
- **Colab training:** YOLOv5 trained on Google Colab GPU for 100 epochs (mAP50 ~0.995), weights deployed to Pi
- **Autostart:** Systemd services ensure the dispenser runs automatically on power-on

---

## Future Improvements

- **Multi-medication support:** Dispense different pills from multiple bottles
- **Scheduling:** Automatic reminders at set times
- **Mobile app:** Remote monitoring and control
- **Cloud logging:** Track medication adherence over time
- **Pill identification:** Use computer vision to verify correct medication
- **Voice feedback:** Speak status updates instead of just LCD

---

## Credits and Citations

- **YOLOv5:** [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- **Vosk Speech Recognition:** [alphacephei.com/vosk](https://alphacephei.com/vosk/)
- **Roboflow:** Dataset annotation and augmentation
- **Google Colab:** GPU training environment
- **Course:** AIPI 590 - AI in the Physical World, Duke University

---

## License

This project is for educational purposes as part of AIPI 590 - AI in the Physical World coursework.

---

## Contact

Diya Mirji  
Duke University  
AIPI 590 - Spring 2026