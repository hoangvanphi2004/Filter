import cv2;
import numpy as np;
import torch;
from Base import config as configBase;
import config;
from image_aug import imageAug;
from model_loading import loadModel;
from model_loading import predictKeypoints;
from apply_filter import applyHatFilter;

cam = cv2.VideoCapture(0);
net = loadModel();

while(True):
    res, image = cam.read();
    inputImage = imageAug(image);
    predictPoints = predictKeypoints(image = inputImage, net = net);
    
    pointList = torch.stack([predictPoints[:configBase.X_COLS_LEN], predictPoints[configBase.X_COLS_LEN:]], dim = 0);
    predictPoints = [None for i in range(configBase.X_COLS_LEN)];
    for index in range(pointList.size()[1]):
        predictPoints[index] = [pointList[0, index].item() * config.INPUT_IMAGE_HEIGHT_SIZE / configBase.IMG_SIZE + config.PADDING_WIDTH,\
                                pointList[1, index].item() * config.INPUT_IMAGE_HEIGHT_SIZE / configBase.IMG_SIZE\
                            ];
    image = applyHatFilter(image = image, predict = predictPoints, path = "LuffyHat.png");
    # ### ----- predict points ----- ###
    # for point in predictPoints:
    #     image[int(point[1]), int(point[0])] = (0, 0, 255);
    #     image[int(point[1]) + 1, int(point[0])] = (0, 0, 255);
    #     image[int(point[1]), int(point[0]) + 1] = (0, 0, 255);
    #     image[int(point[1]) + 1, int(point[0]) + 1] = (0, 0, 255);
    
    cv2.namedWindow("Filter", cv2.WINDOW_NORMAL);
    cv2.resizeWindow("Filter", 720, 720);
    cv2.imshow('Filter', image);

    if cv2.waitKey(1) == ord('q'):
        break;

cam.release();
cv2.destroyAllWindows();
    