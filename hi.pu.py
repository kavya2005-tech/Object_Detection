import cv2
import numpy as np
import pyttsx3  # Text-to-Speech
import time
from ultralytics import YOLO

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed
engine.setProperty('volume', 1.0)  # Maximum volume

# Load YOLOv5 Model
model = YOLO('yolov5s.pt')

# Connect to camera (change '0' for external camera or provide IP camera URL)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not found!")
    exit()

# Object Detection Loop
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Perform object detection
    results = model(frame)

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

            # Add detected object to list
            detected_objects.append(name)

    # Convert detected objects to speech
    if detected_objects:
        text = "I see " + ", ".join(set(detected_objects))
        engine.say(text)
        engine.runAndWait()
        time.sleep(1)  # Avoid overlapping speech

    # Show the frame
    cv2.imshow("Object Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
