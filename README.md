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
- **YOLOv5** - Object detection model for counting pills
- **Vosk** - Offline speech recognition
- **picamzero** - Camera interface
- **gpiozero** - GPIO control for servo, sensors, and button
- **PyTorch** - Deep learning framework for YOLOv5
- **OpenCV** - Image processing
- **sounddevice** - Audio recording
- **scikit-learn** - For ultrasonic sensor model training (midterm)

---

## Project Structure

```
edge_device_proj/
├── README.md                          # This file
├── final_pill_dispenser.py            # Main script (voice + camera)
├── capture_train_imgs.py              # Script to collect training images for YOLO
├── train_model.py                     # Script to download dataset and train YOLOv5
├── training_images/                   # Collected images for YOLOv5 training
│   └── pill_*.jpg                     # ~150+ images with varying pill counts
├── yolov5/                            # YOLOv5 repository (cloned)
│   ├── train.py                       # YOLOv5 training script
│   ├── detect.py                      # YOLOv5 inference script
│   └── runs/train/pill_detection3/    # Training results
│       └── weights/
│           ├── best.pt                # Best trained model (use this!)
│           └── last.pt                # Final epoch model
├── Pill-Detection-2/                  # Roboflow dataset (annotated images)
│   ├── data.yaml                      # Dataset configuration
│   ├── train/                         # Training images + labels
│   ├── valid/                         # Validation images + labels
│   └── test/                          # Test images + labels
└── models/                            # Vosk speech models
    └── vosk-model-small-en-us-0.15/   # English speech recognition model
```

---

## Setup Instructions

### 1. Hardware Setup

**GPIO Pin Connections:**
- **Servo Motor:**
  - Signal → GPIO 21 (BCM)
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

### 2. Software Installation

#### Install System Dependencies
```bash
sudo apt update
sudo apt install python3-pip python3-venv portaudio19-dev python3-opencv pigpio

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

# Data science (for midterm)
pip install scikit-learn

# Roboflow (for dataset management)
pip install roboflow
```

#### Clone YOLOv5 Repository
```bash
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..
```

#### Download Vosk Model
```bash
# Download and extract Vosk speech model
mkdir -p models
cd models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
cd ..
```

### 3. Data Collection and Model Training

#### Collect Training Images
```bash
python capture_train_imgs.py
# Press ENTER to capture images with 0, 1, 2, 3, 4, 5+ pills
# Capture ~150-200 images total
```

#### Annotate Images (on Roboflow)
1. Go to [roboflow.com](https://roboflow.com) and create account
2. Create new project: "Pill Detection" (Object Detection)
3. Upload images from `training_images/`
4. Draw bounding boxes around each pill
5. Use auto-labeling after ~30 manual annotations
6. Generate dataset version (YOLOv5 PyTorch format)
7. Export and note the API key

#### Train YOLOv5 Model
```bash
# Update train_model.py with your Roboflow API key
# Then run training (use tmux to prevent disconnection)
tmux new -s training
source .venv/bin/activate
python train_model.py

# Detach from tmux: Ctrl+B then D
# Reattach later: tmux attach -t training
```

Training takes 30-60 minutes on Raspberry Pi. Model will be saved to:
`yolov5/runs/train/pill_detection3/weights/best.pt`

---

## Running the Project

```bash
source .venv/bin/activate
sudo pigpiod  # Start pigpio daemon
python final_pill_dispenser.py
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

### Testing the YOLO Model
```bash
# Test inference on a single image
python yolov5/detect.py \
  --weights yolov5/runs/train/pill_detection3/weights/best.pt \
  --source training_images/pill_20260329_141820.jpg \
  --conf 0.4

# Results saved to yolov5/runs/detect/
```

---

## Configuration

### Adjustable Parameters

In `final_pill_dispenser.py`:

```python
# Pin assignments
SERVO_PIN = 21

# Model paths
YOLO_MODEL_PATH = "yolov5/runs/train/pill_detection3/weights/best.pt"
VOSK_MODEL_DIR = "../models/vosk-model-small-en-us-0.15"

# Voice recognition settings
WAKE_PHRASE = "hi there"
RECORD_SECONDS = 5

# Servo control
increment = 1  # Angle increment per step (degrees)
max_angle = 180  # Maximum tipping angle
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
sudo pigpiod

# Check if it's running
ps aux | grep pigpio
```

### Microphone Not Working
```bash
# List audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"

# Test recording
python -c "import sounddevice as sd; import numpy as np; audio = sd.rec(16000, samplerate=16000, channels=1); sd.wait(); print('Recording complete')"
```

### Model Loading Errors
- Ensure model path is correct in script
- Check that training completed successfully
- Verify `best.pt` exists in weights folder

### LCD Not Displaying
```bash
# Check I2C devices
sudo i2cdetect -y 1

# Install I2C tools if needed
sudo apt install i2c-tools
```

---

## Project Evolution

This project builds upon a midterm version that used an ultrasonic distance sensor for binary pill detection (present/absent). The final version replaces the ultrasonic sensor with computer vision for precise pill counting and adds voice control for a fully hands-free experience.

### Key Upgrades
- **Added PiCamera:** Real-time pill counting with YOLOv5 object detection
- **Added Microphone:** Voice command input with Vosk speech recognition
- **Wake phrase:** "Hi there" triggers listening mode
- **Precise counting:** Stops dispensing at exact target count
- **Enhanced feedback:** Real-time count display on LCD (e.g., "Pouring... 2/3 pills")

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
- **Course:** AIPI 590 - AI in the Physical World, Duke University

---

## License

This project is for educational purposes as part of AIPI 590 coursework.

---

## Contact

Diya Mirji  
Duke University  
AIPI 590 - Spring 2026
