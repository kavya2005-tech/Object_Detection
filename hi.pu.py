import torch
import cv2
import numpy as np
import pyttsx3  # Text-to-Speech
import time

# Initialize TTS engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speech speed
engine.setProperty('volume', 1.0)  # Maximum volume

# Load YOLOv5 Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

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

    # Extract detections
    detections = results.pandas().xyxy[0]

    detected_objects = []  # Store detected object names

    for _, row in detections.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        confidence, name = row['confidence'], row['name']

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
