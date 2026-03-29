#!/usr/bin/env python3
"""
Train YOLOv5 model for pill detection
"""

from roboflow import Roboflow
import os

# Download dataset from Roboflow
print("Downloading dataset from Roboflow...")
rf = Roboflow(api_key="JGIi3Cm7KbfdE8vASs8j")
project = rf.workspace("diyas-workspace-hsyu7").project("pill-detection-dtahz")
version = project.version(2)
dataset = version.download("yolov5")

print(f"\nDataset downloaded to: {dataset.location}")
print(f"data.yaml location: {dataset.location}/data.yaml")

# Train the model
print("\nStarting training...")
print("This will take a while on Raspberry Pi (30-60 minutes)")

os.system(f"""
python yolov5/train.py \
    --img 640 \
    --batch 8 \
    --epochs 50 \
    --data {dataset.location}/data.yaml \
    --weights yolov5n.pt \
    --cache \
    --name pill_detection
""")

print("\nTraining complete!")
print("Model saved to: yolov5/runs/train/pill_detection/weights/best.pt")