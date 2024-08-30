import cv2;
import numpy as np;
import torch;
from Base import config as configBase;
import config;
from keypoints_predict import loadModel;
from keypoints_predict import predictKeypoints;
from apply_filter import applyHatFilter, applyMaskFilter, draw;
from sys import argv;
import box_predict;
import json
from smoothing_motion import local_search, create_kalman_filter, kalman_filter_update
import writeVideo

import time

cam = cv2.VideoCapture(0);
cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
box_predict.init_model()
net = loadModel();
previousPointsOfAllFaces = []
previousoundingBoxs = []
cnt = 0;

if("save-video" in argv):
    videoWriter = writeVideo.VideoWriter()

def insideFrame(boundingBox):
    if(boundingBox[0] - 20 > 0 and boundingBox[1] + 20 < image.shape[1] and boundingBox[2] - 20 > 0 and boundingBox[3] + 20 < image.shape[0]):
        return True
    else:
        return False
    
while(True):
    res, image = cam.read();
    
    image = cv2.copyMakeBorder(image, 200, 200, 200, 200, cv2.BORDER_CONSTANT, value = (255, 255, 255))

    predictBoundingBoxs = box_predict.predict(image = image);
    temp = []
    for boundingBox in predictBoundingBoxs:
            if(insideFrame(boundingBox)):
                temp.append(boundingBox)
    predictBoundingBoxs = temp

    predictPointsOfAllFaces = predictKeypoints(image = image, boundingBoxs = predictBoundingBoxs, net = net)

    aveTime = 3
    
    for i in range(aveTime - 1):
        predictPointsOfAllFaces1 = predictKeypoints(image = image, boundingBoxs = predictBoundingBoxs, net = net)
        predictPointsOfAllFaces += predictPointsOfAllFaces1
    
    if(len(predictPointsOfAllFaces) != 0):
        predictPointsOfAllFaces /= aveTime
    
    predictBoundingBoxs = np.array(predictBoundingBoxs, dtype = np.int64)
    
    firstPoints = predictPointsOfAllFaces
    firstBoundingBoxs = predictBoundingBoxs

    if(len(previousPointsOfAllFaces) == len(predictPointsOfAllFaces)):
        predictPointsOfAllFacesInNumpy = np.array(predictPointsOfAllFaces);
        predictPointsOfAllFaces, predictBoundingBoxs = kalman_filter_update(predictPointsOfAllFacesInNumpy, predictBoundingBoxs)
    else:
        create_kalman_filter(len(predictPointsOfAllFaces), argv)
        cnt = 0;
        pass

    previousPointsOfAllFaces = predictPointsOfAllFaces
    previousoundingBoxs = predictBoundingBoxs

    if("floating-mask" not in argv):
        if cnt < 10:
            predictPointsOfAllFaces = firstPoints
            predictBoundingBoxs = firstBoundingBoxs
            cnt += 1

    ###-----------------Options--------------------###
    if(len(argv) > 1 and argv[1] == "hat-filter"):
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
            xPoints[xPoints >= image.shape[1] - 1] = image.shape[1] - 2;
            yPoints[yPoints >= image.shape[0] - 1] = image.shape[0] - 2;
            xPoints[xPoints < 0] = 0;
            yPoints[yPoints < 0] = 0;

            image[yPoints, xPoints] = (0, 0, 255);
            image[yPoints + 1, xPoints] = (0, 0, 255)
            image[yPoints, xPoints + 1] = (0, 0, 255)
            image[yPoints + 1, xPoints + 1] = (0, 0, 255)
        for boudingBox in predictBoundingBoxs:
            points = [(boudingBox[0], boudingBox[2]), (boudingBox[1], boudingBox[3])];
            image = cv2.rectangle(image, points[0], points[1], (255, 0, 0), 2);
    
    if(len(argv) > 1 and argv[1] == "mask-filter"):
        
        for i in range(len(predictBoundingBoxs)):
            frameBoudingBox = np.copy(predictBoundingBoxs[i])
            framePredictKeypoints = np.copy(predictPointsOfAllFaces[i])
            
            if(not insideFrame(frameBoudingBox)):
                continue;
            
            mask = cv2.imread("./mask.png")
            maskKeypoints = np.array(json.load(open("keypoints_mask.json")))
            width = mask.shape[1]
            height = mask.shape[0]
            maskBoundingBox = np.array([10, width - 10, 10, height - 10])
            image = applyMaskFilter(image, frameBoudingBox, framePredictKeypoints, mask, maskBoundingBox, maskKeypoints);
            
    image = image[200: -200, 200: -200]

    cv2.namedWindow("Filter", cv2.WINDOW_NORMAL);
    cv2.imshow('Filter', image);

    if("save-video" in argv):
        videoWriter.write(image)
        
    if cv2.waitKey(1) == ord('q'):
        break;

    if cv2.getWindowProperty("Filter", cv2.WND_PROP_VISIBLE) < 1:
        break;

if("save-video" in argv):  
    videoWriter.release()

cam.release();
cv2.destroyAllWindows();

