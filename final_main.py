from time import sleep
from gpiozero import Servo
from gpiozero.pins.pigpio import PiGPIOFactory
import time
from picamzero import Camera
import torch
import cv2
import numpy as np
from lcd_i2c import LCD_I2C
import re
import json
import numpy as np
import sounddevice as sd
from vosk import Model, KaldiRecognizer

# Pin configuration
SERVO_PIN = 16

# Hardware setup
factory = PiGPIOFactory()
servo = None
camera = None
lcd = None
yolo_model = None
vosk_model = None
vosk_recognizer = None

# Wake phrase configuration
WAKE_PHRASE = "hi there"

# Vosk settings
VOSK_MODEL_DIR = "vosk-model-small-en-us-0.15"  # Update path as needed
SAMPLE_RATE = 44100
CHANNELS = 1
RECORD_SECONDS = 5
YOLO_MODEL_PATH = "./best.pt"

def setup_hardware():
    """Initialize all hardware components."""
    global servo, camera, lcd, vosk_model, vosk_recognizer
    
    servo = Servo(SERVO_PIN, pin_factory=factory)
    camera = Camera()
    lcd = LCD_I2C(39, 16, 2)
    lcd.backlight.on()
    lcd.blink.off()
    
    # Setup Vosk speech recognition
    print(f"Loading Vosk model from {VOSK_MODEL_DIR}...")
    vosk_model = Model(VOSK_MODEL_DIR)
    vosk_recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    print("Vosk model loaded.")
    
    # Let camera warm up
    time.sleep(2)
    print("Hardware initialized.")

def load_yolo_model():
    """Load the trained YOLOv5 pill detection model."""
    global yolo_model
    
    print(f"Loading model from {YOLO_MODEL_PATH}...")
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', 
                                path=YOLO_MODEL_PATH, force_reload=True)
    yolo_model.conf = 0.4  # Confidence threshold
    print("Model loaded successfully.")

def set_lcd_text(text):
    """Write text to LCD display (max 32 chars, 2 lines of 16)."""
    global lcd
    lcd.clear()
    lcd.cursor.setPos(0, 0)
    lcd.write_text(text[:16])
    if len(text) > 16:
        lcd.cursor.setPos(1, 0)
        lcd.write_text(text[16:32])

def set_servo_angle(angle):
    """Set servo to specific angle (0-180 degrees)."""
    value = (angle / 90.0) - 1  # Convert to -1 to 1 range
    servo.value = value

def count_pills_in_frame():
    """
    Capture frame from camera and count pills using YOLOv5.
    Returns: number of pills detected
    """
    global camera, yolo_model
    
    # Capture frame
    frame = camera.capture_array()
    print("Frame captured for pill counting.")
    
    # Convert RGB to BGR for OpenCV/YOLO
    if len(frame.shape) == 3 and frame.shape[2] == 3:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        frame_bgr = frame
    print("Frame converted to BGR format.")
    # Run inference
    results = yolo_model(frame_bgr)
    print("Inference completed.")
    
    # Count detections
    detections = results.pandas().xyxy[0]  # Pandas DataFrame
    pill_count = len(detections)
    print(f"Detected {pill_count} pills.")
    
    return pill_count

def record_audio(seconds):
    """Record audio from microphone and return numpy array."""
    frames = int(SAMPLE_RATE * seconds)
    audio = sd.rec(frames, samplerate=SAMPLE_RATE, channels=CHANNELS, dtype="float32")
    sd.wait()
    return audio

def float_to_pcm16(audio_float):
    """Convert float32 audio to 16-bit PCM bytes for Vosk."""
    x = audio_float.reshape(-1)
    x = np.clip(x, -1.0, 1.0)
    pcm = (x * 32767).astype(np.int16)
    return pcm.tobytes()

def transcribe_with_vosk(audio_float):
    """
    Transcribe audio using Vosk.
    Returns: text string
    """
    global vosk_recognizer
    
    # Reset recognizer for new audio
    vosk_recognizer = KaldiRecognizer(vosk_model, SAMPLE_RATE)
    
    # Convert to PCM bytes
    pcm_bytes = float_to_pcm16(audio_float)
    
    # Process audio
    vosk_recognizer.AcceptWaveform(pcm_bytes)
    result_json = vosk_recognizer.Result()
    result = json.loads(result_json)
    text = result.get("text", "")
    
    return text

def get_voice_command():
    """
    Listen for wake phrase 'Hi there' followed by pill count command.
    Returns: target pill count (int) or None
    """
    set_lcd_text("Say: Hi there, I need X pills")
    print("Listening for wake phrase...")
    
    try:
        # Record audio
        set_lcd_text("Listening...")
        audio = record_audio(RECORD_SECONDS)
        
        # Transcribe with Vosk
        set_lcd_text("Processing...")
        text = transcribe_with_vosk(audio).lower()
        print(f"Heard: '{text}'")
        
        if text.strip() == "":
            print("No speech detected.")
            set_lcd_text("Didn't hear you Try again")
            sleep(2)
            return None
        
        # Check if wake phrase is present
        if WAKE_PHRASE not in text:
            print(f"Wake phrase '{WAKE_PHRASE}' not detected. Ignoring.")
            set_lcd_text("Say 'Hi there' first!")
            sleep(2)
            return None
        
        # Extract number from command
        target_count = voice_to_number(text)
        
        if target_count is None:
            print("No number detected in command.")
            set_lcd_text("Didn't catch    that. Try again")
            sleep(2)
            return None
        
        print(f"Detected target: {target_count} pills")
        return target_count
        
    except Exception as e:
        print(f"Voice recognition error: {e}")
        set_lcd_text("Error. Try     again")
        sleep(2)
        return None

def voice_to_number(text):
    """
    Extract number (1-5) from voice command as either digits or words.
    """
    # Define the mapping
    number_map = {
        "one": 1, "1": 1,
        "two": 2, "2": 2,
        "three": 3, "3": 3,
        "four": 4, "4": 4,
        "five": 5, "5": 5
    }
    
    # Convert text to lowercase to catch "One" or "ONE"
    text = text.lower()
    
    # Create a regex pattern that looks for any of the keys in our map
    # This joins them like: (one|1|two|2|...)
    pattern = r'\b(' + '|'.join(number_map.keys()) + r')\b'
    
    match = re.search(pattern, text)
    if match:
        return number_map[match.group(0)]
        
    return None

def dispense_pills(target_count):
    """
    Main dispensing logic: tip bottle while counting pills with camera.
    Stops when target count is reached.
    """
    current_angle = 0
    increment = 4
    max_angle = 176
    
    set_servo_angle(current_angle)
    sleep(1)
    
    set_lcd_text(f"Pouring...      0/{target_count} pills")
    
    try:
        while current_angle <= max_angle:
            # Increment servo angle
            current_angle += increment
            set_servo_angle(current_angle)
            
            # Count pills in frame
            pill_count = count_pills_in_frame()
            
            # Update LCD with current count
            set_lcd_text(f"Pouring...      {pill_count}/{target_count} pills")
            print(f"Angle: {current_angle}° | Pills detected: {pill_count}/{target_count}")
            
            # Check if target reached
            if pill_count >= target_count:
                print(f"Target reached! {pill_count} pills detected.")
                break
            
            sleep(0.25)  # Small delay between increments
            
    except KeyboardInterrupt:
        print("Dispensing interrupted.")
    
    # Return to starting position
    #set_servo_angle(0)
    sleep(1)
    
    return pill_count

def main():
    """Main program loop."""
    print("Starting Pill Dispenser...")
    
    # Initialize hardware and model
    setup_hardware()
    load_yolo_model()
    set_servo_angle(0)
    
    try:
        while True:
            # State 1: Ready/Waiting
            set_lcd_text("Ready. Say:    Hi there...")
            print("\nWaiting for voice command...")
            
            # Get voice command with wake phrase detection
            target_count = get_voice_command()
            
            if target_count is None or target_count <= 0:
                # Invalid or no command, loop back
                continue
            
            # State 2: Acknowledge and start
            set_lcd_text(f"Got it! Pouring{target_count} pills...")
            print(f"Target: {target_count} pills")
            sleep(2)
            
            # State 3: Dispense
            final_count = dispense_pills(target_count)
            
            # State 4: Success
            set_lcd_text(f"Done! {final_count} pills  dispensed")
            print(f"Dispensing complete. {final_count} pills.")
            sleep(5)
            
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        cleanup()

def cleanup():
    """Clean up hardware on exit."""
    print("Cleaning up...")
    set_lcd_text("Shutting down...")
    set_servo_angle(0)
    sleep(1)
    servo.close()
    lcd.clear()
    lcd.backlight.off()
    print("Cleanup complete.")

if __name__ == "__main__":
    main()