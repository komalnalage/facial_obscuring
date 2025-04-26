import cv2
import numpy as np
import os

def detect_and_obscure_faces_image(image, method="blur"):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 
                                         "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        face_region = image[y:y+h, x:x+w]
        if method == "Blur":
            face_region = cv2.GaussianBlur(face_region, (99, 99), 30)
        elif method == "Pixelate":
            face_region = pixelate(face_region)
        image[y:y+h, x:x+w] = face_region
    return image

def pixelate(image, blocks=10):
    h, w = image.shape[:2]
    x_steps = w // blocks
    y_steps = h // blocks
    for y in range(0, h, y_steps):
        for x in range(0, w, x_steps):
            roi = image[y:y+y_steps, x:x+x_steps]
            if roi.size == 0: continue
            color = roi.mean(axis=(0, 1)).astype(int)
            image[y:y+y_steps, x:x+x_steps] = color
    return image

def extract_frames(video_path, output_dir, interval=1):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:03}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1
    cap.release()
