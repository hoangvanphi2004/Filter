import torch;
import config;
from Base import config as configBase
import albumentations as A
from albumentations.pytorch import ToTensorV2;
from albumentations.augmentations import geometric, crops;
import numpy as np;

def imageAug(image, boundingBox):
    transform = A.Compose([
        crops.transforms.Crop(x_min = boundingBox[0], y_min = boundingBox[2], x_max = boundingBox[1], y_max = boundingBox[3]),
        geometric.resize.Resize(configBase.IMG_SIZE, configBase.IMG_SIZE, always_apply = True),
        A.transforms.Normalize(),
        ToTensorV2()
    ]);
    image = transform(image = image)['image'];
    return image;

def scalePoints(points, boundingBox, initialSize):
    width = boundingBox[1] - boundingBox[0];
    height = boundingBox[3] - boundingBox[2];
    widthRatio = width / initialSize;
    heightRatio = height / initialSize;
    points[:, 0] = points[:, 0] * widthRatio + boundingBox[0];
    points[:, 1] = points[:, 1] * heightRatio + boundingBox[2];
    return points;
    