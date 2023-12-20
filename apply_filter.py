import cv2;
import albumentations as A
from albumentations.pytorch import ToTensorV2;
from albumentations.augmentations import geometric, crops;
from PIL import Image;
import numpy as np;
import matplotlib.pyplot as plt;
import math;

def overlapIamge(background, object, x, y):
    # object background : black
    # background, object : (height, width, 3);
    # (x, y) : top - left pixel of object;
    width = object.shape[1];
    height = object.shape[0];
    backgroundWidth = background.shape[1];
    backgroundHeight = background.shape[0];
    
    objectRegion = object[max(0, -y): min(height, backgroundHeight - y), max(0, -x): min(width, backgroundWidth - x)];
    objectMask = np.copy(objectRegion[:, :, 0]);
    width = objectMask.shape[1];
    height = objectMask.shape[0];
    objectMask[objectMask > 0] = 255;

    backgroundRegion = background[max(y, 0): max(y, 0) + height, max(x, 0): max(x, 0) + width];

    objectMask = cv2.bitwise_not(objectMask);
    filterRegion = cv2.bitwise_and(backgroundRegion, backgroundRegion, mask = objectMask);
    fullFill = cv2.add(objectRegion, filterRegion);
    background[max(y, 0): max(y, 0) + height, max(x, 0): max(x, 0) + width] = fullFill;

    return background;

def alpha(point1, point2):
    return math.degrees(math.atan((point2[1] - point1[1]) / (point2[0] - point1[0])));
    
def cornerPoint(point1, point2, distance, object):
    middlePoint = ((point1[0] + point2[0]) / 2, (point1[1] + point2[1]) / 2);
    turnDeg = alpha(point1 = point1, point2 = point2) - 90;
    middleObject = (middlePoint[0] + distance * math.cos(math.radians(turnDeg)), middlePoint[1] + distance * math.sin(math.radians(turnDeg)));
    cornerPoint = (int(middleObject[0] - object.shape[1] / 2), int(middleObject[1] - object.shape[0] / 2));
    return cornerPoint;

def turnAndPadding(point1, point2, object):
    turnDeg = - alpha(point1 = point1, point2 = point2);
    padding = int(object.shape[0] / 2  * (abs(math.sin(math.radians(turnDeg))) + abs(math.cos(math.radians(turnDeg))) - 1));  
    # Padding
    object = cv2.copyMakeBorder(object, padding, padding, padding, padding, cv2.BORDER_CONSTANT, None, value = 0)
    # Rotate Image
    center = (object.shape[1] / 2, object.shape[0] / 2);
    rotateMatrix = cv2.getRotationMatrix2D(center = center, angle = turnDeg, scale = 1);
    return cv2.warpAffine(object, rotateMatrix, (object.shape[1], object.shape[0]));

def applyHatFilter(image, predict, path):
    top_leftEye = predict[19];
    top_rightEye = predict[24];
    distance = int(math.dist(top_leftEye, top_rightEye) * 5);
    hatImage = np.array(Image.open(path).convert("RGB"));
    hatImage = cv2.cvtColor(hatImage, cv2.COLOR_RGB2BGR);
    hatImage[hatImage == np.array([255, 255, 255])] = [0, 0, 0] * (int(hatImage[hatImage == np.array([255, 255, 255])].shape[0] / 3));
    if(hatImage.shape[0] > hatImage.shape[1]):
        padding = int((hatImage.shape[0] - hatImage.shape[1]) / 2);
        hatImage = cv2.copyMakeBorder(hatImage, top = 0, left = padding, bottom = 0, right = padding, borderType = cv2.BORDER_CONSTANT, dst = None, value = 0);
    else:
        padding = int((hatImage.shape[1] - hatImage.shape[0]) / 2);
        hatImage = cv2.copyMakeBorder(hatImage, top = padding, left = 0, bottom = padding, right = 0, borderType = cv2.BORDER_CONSTANT, dst = None, value = 0);
    hatImage = cv2.resize(hatImage, (distance, distance), interpolation = cv2.INTER_LINEAR);
    
    paddingImage = turnAndPadding(top_leftEye, top_rightEye, hatImage);
    corner = cornerPoint(top_leftEye, top_rightEye, distance / 8, paddingImage);
    fullImage = overlapIamge(image, paddingImage, corner[0], corner[1]);
    return fullImage;
 

    