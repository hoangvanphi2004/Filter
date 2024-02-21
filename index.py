import cv2;
import numpy as np;
import torch;
from Base import config as configBase;
import config;
from model_loading import loadModel;
from model_loading import predictKeypoints;
from apply_filter import applyHatFilter;
from sys import argv;
import box_predict;

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW);
net = loadModel();

while(True):
    res, image = cam.read();
    
    predictBoundingBoxs = box_predict.predict(image = image);
    
    predictPointsOfAllFaces = predictKeypoints(image = image, boundingBoxs = predictBoundingBoxs, net = net);
    
    if(len(argv) > 1 and argv[1] == "filter"):
        for predictPoints in predictPointsOfAllFaces:
            image = applyHatFilter(image = image,\
                predict = predictPoints,\
                path = argv[2] if len(argv) > 2 else config.PATH,\
                height = int(argv[3]) if len(argv) > 3 else 5);
    
    if(len(argv) == 1 or argv[1] == "keypoints"):
        for predictPoints in predictPointsOfAllFaces:
            predictPoints = predictPoints.type(torch.int);
            xPoints = predictPoints[:, 0].clone();
            yPoints = predictPoints[:, 1].clone();
            xPoints[xPoints >= config.INPUT_IMAGE_WIDTH_SIZE - 1] = config.INPUT_IMAGE_WIDTH_SIZE - 2;
            yPoints[yPoints >= config.INPUT_IMAGE_HEIGHT_SIZE - 1] = config.INPUT_IMAGE_HEIGHT_SIZE - 2;
            xPoints[xPoints < 0] = 0;
            yPoints[yPoints < 0] = 0;
            
            image[yPoints, xPoints] = (0, 0, 255);
            image[yPoints + 1, xPoints] = (0, 0, 255)
            image[yPoints, xPoints + 1] = (0, 0, 255)
            image[yPoints + 1, xPoints + 1] = (0, 0, 255)
        for boudingBox in predictBoundingBoxs:
            points = [(boudingBox[0], boudingBox[2]), (boudingBox[1], boudingBox[3])];
            image = cv2.rectangle(image, points[0], points[1], (255, 0, 0), 2);
    
    cv2.namedWindow("Filter", cv2.WINDOW_NORMAL);
    cv2.imshow('Filter', image);

    if cv2.waitKey(1) == ord('q'):
        break;

    if cv2.getWindowProperty("Filter", cv2.WND_PROP_VISIBLE) < 1:
        break;
    
cam.release();
cv2.destroyAllWindows();
    