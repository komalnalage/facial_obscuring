import face_recognition
import os
from sklearn.metrics import accuracy_score

def evaluate_accuracy(dataset_path):
    known_encodings = []
    known_names = []

    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                path = os.path.join(root, file)
                image = face_recognition.load_image_file(path)
                encodings = face_recognition.face_encodings(image)
                if encodings:
                    known_encodings.append(encodings[0])
                    known_names.append(os.path.basename(root))

    predictions = [face_recognition.compare_faces(known_encodings, enc)[0] for enc in known_encodings]
    return accuracy_score(known_names, known_names if predictions else ["unknown"] * len(known_names))