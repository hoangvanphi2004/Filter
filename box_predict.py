import torch
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image

app = None

def init_model():
    global app
    app = FaceAnalysis(name="buffalo_sc");
    app.prepare(ctx_id=0)

def predict(image):
    boundingBoxs = [];
    faces = app.get(image)
    for face in faces:
        boundingBox = face["bbox"].astype(int);
        boundingBox = [boundingBox[0], boundingBox[2], boundingBox[1], boundingBox[3]];
        boundingBoxs.append(boundingBox);
    return boundingBoxs;