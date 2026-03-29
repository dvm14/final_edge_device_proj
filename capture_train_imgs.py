#!/usr/bin/env python3
"""
Image collection script for pill detection training data.
Captures images from PiCamera for labeling (no preview version).
"""

from picamzero import Camera
import cv2
import os
from datetime import datetime
import time

# Configuration
SAVE_DIR = "training_images"

# Create save directory if it doesn't exist
os.makedirs(SAVE_DIR, exist_ok=True)

def setup_camera():
    """Initialize and configure the camera."""
    cam = Camera()
    
    # Let camera warm up
    time.sleep(2)
    
    return cam

def main():
    print("Starting pill image collection...")
    print(f"Images will be saved to: {SAVE_DIR}/")
    print("\nControls:")
    print("  Press ENTER to capture image")
    print("  Type 'q' and press ENTER to quit")
    print("\nTips for good training data:")
    print("  - Capture images with 0, 1, 2, 3, 4, 5+ pills")
    print("  - Vary pill positions and orientations")
    print("  - Include different lighting conditions")
    print("  - Capture some with pills touching/overlapping")
    print()
    
    picam2 = setup_camera()
    image_count = len([f for f in os.listdir(SAVE_DIR) if f.endswith('.jpg')])
    
    try:
        while True:
            user_input = input(f"[{image_count} images] Press ENTER to capture (or 'q' to quit): ").strip().lower()
            
            if user_input == 'q':
                print(f"\nCollection complete! {image_count} images saved.")
                break
            
            # Capture frame as numpy array
            frame = picam2.capture_array()
            
            # picamzero returns RGB, convert to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pill_{timestamp}.jpg"
            filepath = os.path.join(SAVE_DIR, filename)
            
            cv2.imwrite(filepath, frame_bgr)
            image_count += 1
            print(f"✓ Saved: {filename}")
                
    except KeyboardInterrupt:
        print(f"\nCollection interrupted. {image_count} images saved.")

if __name__ == "__main__":
    main()