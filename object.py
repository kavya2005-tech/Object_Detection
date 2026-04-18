import torch
import cv2
import numpy as np
import pyttsx3
import threading  # For non-blocking TTS

# Initialize Text-to-Speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True, trust_repo=True)

# Connect to iPhone camera via DroidCam
droidcam_url = "http://192.168.22.162:4747/video"  # Replace with your IP
cap = cv2.VideoCapture(droidcam_url)

# Set buffer size to 1 (minimizes delay)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Function to run text-to-speech in a separate thread
def speak(text):
    thread = threading.Thread(target=lambda: engine.say(text) or engine.runAndWait())
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

        # Add object name to list
        detected_objects.append(name)

    # Convert detected objects to speech in a separate thread
    if detected_objects:
        text = "I see " + ", ".join(set(detected_objects))  # Avoid repeating names
        speak(text)  # Run in background to prevent lag

    # Show the frame
    cv2.imshow("Real-Time Object Detection (DroidCam)", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
