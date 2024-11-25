# Face tracking using the meanshift algorithm  
# Author: Bakpen Kombat 

import cv2
import numpy as np

"""1. Load the video file"""
video_path = 'video2.mp4'  # Video path
video_capture = cv2.VideoCapture(video_path)
if not video_capture.isOpened():
    print("Error: Could not open video.")
    exit()



"""2. Get Video frames and convert to grayscale"""
ret, first_frame = video_capture.read()
if not ret:
    print("Failed to read the video file.")
    video_capture.release()
    exit()
gray_first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)


"""3. Detect the face using Haar Cascade"""
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_first_frame, scaleFactor=1.1, minNeighbors=5)
if len(faces) == 0:
    print("No face detected. Exiting.")
    video_capture.release()
    exit()
    

"""4. Perform the tracking """
(x, y, w, h) = faces[0]
print(f"Initial face detected at: {(x, y, w, h)}")
while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5)
    if len(faces) > 0:
        (x, y, w, h) = faces[0]  
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
    cv2.imshow('Face Tracking', frame)

    key = cv2.waitKey(30) 
    if key == 27:
        break

video_capture.release()
cv2.destroyAllWindows()
