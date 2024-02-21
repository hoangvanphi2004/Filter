from deepface import DeepFace
import torch

def predict(image):
    boundingBoxs = [];
    faces = DeepFace.extract_faces(image, enforce_detection = False);
    for face in faces:
        boundingBox = face["facial_area"];
        boundingBox = [boundingBox['x'], boundingBox['x'] + boundingBox['w'], boundingBox['y'], boundingBox['y'] + boundingBox['h']];
        boundingBoxs.append(boundingBox);
    return boundingBoxs;
