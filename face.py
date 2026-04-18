import cv2
import numpy as np
import pyttsx3
import os

# Initialize Text-to-Speech (TTS) engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Adjust speed
engine.setProperty('volume', 1.0)  # Full volume

# Load OpenCV's pre-trained deep learning face detector model
face_detector = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt",  # Model architecture
    "res10_300x300_ssd_iter_140000.caffemodel"  # Pre-trained weights
)

# Load known faces from images (using OpenCV instead of `face_recognition`)
known_faces = {}
faces_dir = "faces"

for filename in os.listdir(faces_dir):
    image_path = os.path.join(faces_dir, filename)
    image = cv2.imread(image_path)
    known_faces[filename.split(".")[0]] = image  # Store with name

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to blob for OpenCV's DNN model
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:  # Threshold to filter weak detections
            box = detections[0, 0, i, 3:7] * np.array(
                [frame.shape[1], frame.shape[0], frame.shape[1], frame.shape[0]]
            )
            (startX, startY, endX, endY) = box.astype("int")

            # Extract face from frame
            face_crop = frame[startY:endY, startX:endX]

            # Try to match face with known faces (very basic matching method)
            name = "Unknown"
            for known_name, known_face in known_faces.items():
                diff = cv2.absdiff(cv2.resize(face_crop, (100, 100)), cv2.resize(known_face, (100, 100)))
                if np.mean(diff) < 50:  # Simple similarity check
                    name = known_name
                    break

            # Draw bounding box and label
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
            cv2.putText(frame, name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Speak out the recognized name
            engine.say(f"Hello, {name}")
            engine.runAndWait()

    # Show the frame
    cv2.imshow("Face Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
