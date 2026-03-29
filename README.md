# final_edge_device_proj


cd ~/Desktop/edge_device_proj

# Create virtual environment
python3 -m venv pill_env

# Activate it
source pill_env/bin/activate

# Now install packages
pip install roboflow

# Clone YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt
cd ..
