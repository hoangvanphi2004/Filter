import torch;
import matplotlib.pyplot as plt;
import model;
from image_aug import imageAug, scalePoints;
from config import PREDICT_SIZE;
import Base.config as configBase;

def loadModel():
    net = model.Model();
    net.load_state_dict(torch.load("resnet-for-face-points-recognize-state-dict.pth"));
    return net;
    
def predictKeypoints(image, boundingBoxs, net):
    keypoints = [];
    for boundingBox in boundingBoxs:
        cropImage = imageAug(image = image, boundingBox = boundingBox);
        
        predict = net(torch.tensor(cropImage).unsqueeze(0))[0];
        predict = torch.stack([predict[:configBase.X_COLS_LEN], predict[configBase.X_COLS_LEN:]], dim = 0);
        predict = predict.swapaxes(0, 1);
        
        predict = scalePoints(points = predict, boundingBox = boundingBox, initialSize = PREDICT_SIZE);
        keypoints.append(predict);
    return torch.stack(keypoints, dim = 0);