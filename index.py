import cv2;
import numpy as np;
import torch;
from Base import config as configBase;
import config;
from image_aug import imageAug;
from model_loading import loadModel;
from model_loading import predictKeypoints;
from apply_filter import applyHatFilter;
from sys import argv;

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW);
net = loadModel();

while(True):
    res, image = cam.read();
    inputImage = imageAug(image);
    predictPoints = predictKeypoints(image = inputImage, net = net);
    image = image[:, config.PADDING_WIDTH: config.PADDING_WIDTH + config.INPUT_IMAGE_HEIGHT_SIZE, :];
    
    pointList = torch.stack([predictPoints[:configBase.X_COLS_LEN], predictPoints[configBase.X_COLS_LEN:]], dim = 0);
    predictPoints = [None for i in range(configBase.X_COLS_LEN)];
    for index in range(pointList.size()[1]):
        predictPoints[index] = [pointList[0, index].item() * config.INPUT_IMAGE_HEIGHT_SIZE / configBase.IMG_SIZE,\
                                pointList[1, index].item() * config.INPUT_IMAGE_HEIGHT_SIZE / configBase.IMG_SIZE\
                            ];
    if(len(argv) > 1 and argv[1] == "filter"):
        image = applyHatFilter(image = image,\
            predict = predictPoints,\
            path = argv[2] if len(argv) > 2 else config.PATH,\
            height = int(argv[3]) if len(argv) > 3 else 0);
    
    if(len(argv) == 1 or argv[1] == "keypoints"):
        for point in predictPoints:
            image[int(point[1]), int(point[0])] = (0, 0, 255);
            image[int(point[1]) + 1, int(point[0])] = (0, 0, 255);
            image[int(point[1]), int(point[0]) + 1] = (0, 0, 255);
            image[int(point[1]) + 1, int(point[0]) + 1] = (0, 0, 255);
    
    cv2.namedWindow("Filter", cv2.WINDOW_NORMAL);
    cv2.resizeWindow("Filter", 720, 720);
    cv2.imshow('Filter', image);

    if cv2.waitKey(1) == ord('q'):
        break;

    if cv2.getWindowProperty("Filter", cv2.WND_PROP_VISIBLE) < 1:
        break;
    
cam.release();
cv2.destroyAllWindows();
    