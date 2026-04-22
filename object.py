import torch
import cv2
import numpy as np
import pyttsx3
import threading  # For non-blocking TTS
from ultralytics import YOLO

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

model = YOLO('yolov5su.pt')

# Connect to laptop camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Set buffer size to 1 (minimizes delay)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Function to run text-to-speech in a separate thread
def speak(text):
    def _speak():
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=_speak)
    thread.start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Resize frame for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Convert BGR (OpenCV) to RGB (YOLO expects RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Perform object detection
    results = model(frame_rgb)

    detected_objects = []  # Store detected object names

    # Extract detections from ultralytics YOLO format
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0].item()
            class_id = int(box.cls[0].item())
            name = model.names[class_id]

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display label and confidence
            label = f"{name} {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Add object name to list
            detected_objects.append(name)

    # Convert detected objects to speech in a separate thread
    if detected_objects:
        text = "I see " + ", ".join(set(detected_objects))  # Avoid repeating names
        speak(text)  # Run in background to prevent lag

    # Show the frame
    cv2.imshow("Real-Time Object Detection (DroidCam)", frame)

    # Press 'q' to exit or close the window
    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Real-Time Object Detection (DroidCam)", cv2.WND_PROP_VISIBLE) < 1:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()