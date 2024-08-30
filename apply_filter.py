import cv2;
import albumentations as A
from albumentations.pytorch import ToTensorV2;
from albumentations.augmentations import geometric, crops;
from PIL import Image;
import numpy as np;
import matplotlib.pyplot as plt;
import math;
import box_predict;
import json
import torch
from keypoints_predict import loadModel, predictKeypoints
import time

###--------------------Hat Filter--------------------###
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
    if point2[0] - point1[0] == 0:
        return 0
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

def applyHatFilter(image, predict, path, height = 0):
    top_leftEye = predict[19];
    top_rightEye = predict[24];
    distance = max(int(math.dist(top_leftEye, top_rightEye) * 5), 1);
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
    corner = cornerPoint(top_leftEye, top_rightEye, (distance * (height / 4)) / 8, paddingImage);
    fullImage = overlapIamge(image, paddingImage, corner[0], corner[1]);
    return fullImage;
 
###-----------------Mask Filter------------------###
def draw(bounding_box, keypoints, frame):
    result = frame
    cv2.rectangle(result, (int(bounding_box[0]), int(bounding_box[2])), (int(bounding_box[1]), int(bounding_box[3])), (255, 0, 0), 4)
    
    for keypoint in keypoints:
        cv2.circle(result, (int(keypoint[0]), int(keypoint[1])), radius = 2, color = (0, 255, 0), thickness = -1)

# Apply a mask to a SINGLE face
def applyMaskFilter(frame, frameBoundingBox, frameKeypoints, maskFilter, maskBoundingBox, maskKeypoints):
    switch = False
    if((maskFilter[0, 0] == (255, 255, 255)).all()):
        maskFilter = 255 - maskFilter
        switch = True

    frameBoundingBox = np.array(frameBoundingBox, dtype = np.int32)
    maskBoundingBox = np.array(maskBoundingBox, dtype = np.int32)
    maskBoundingBoxXYWH = [int(maskBoundingBox[0]), \
                           int(maskBoundingBox[2]), \
                           int(maskBoundingBox[1] - maskBoundingBox[0]), \
                           int(maskBoundingBox[3] - maskBoundingBox[2])]
    frameBoundingBoxXYWH = [int(frameBoundingBox[0]), \
                           int(frameBoundingBox[2]), \
                           int(frameBoundingBox[1] - frameBoundingBox[0]), \
                           int(frameBoundingBox[3] - frameBoundingBox[2])]
    maskBoundingBoxPoints = [(int(maskBoundingBox[0]), int(maskBoundingBox[2])),\
                             (int(maskBoundingBox[1]), int(maskBoundingBox[2])),\
                             (int(maskBoundingBox[1]), int(maskBoundingBox[3])),\
                             (int(maskBoundingBox[0]), int(maskBoundingBox[3]))]
    frameBoundingBoxPoints = [(int(frameBoundingBox[0]), int(frameBoundingBox[2])),\
                             (int(frameBoundingBox[1]), int(frameBoundingBox[2])),\
                             (int(frameBoundingBox[1]), int(frameBoundingBox[3])),\
                             (int(frameBoundingBox[0]), int(frameBoundingBox[3]))]
    
    def turnToArrayOfPoints(arg):
        return (int(arg[0]), int(arg[1]));

    frameKeypoints = np.array(frameKeypoints, dtype = np.int32)
    maskKeypoints = np.array(maskKeypoints, dtype = np.int32)
    
    listOfFrameKeypoints = list(map(\
        turnToArrayOfPoints, \
        np.concatenate([\
            frameKeypoints, \
            np.array(frameBoundingBoxPoints)\
        ], axis = 0)\
    ));
    listOfMaskKeypoints = list(map(\
        turnToArrayOfPoints,\
        np.concatenate([\
            maskKeypoints,\
            np.array(maskBoundingBoxPoints)\
            ], axis = 0)\
    ));
    
    #------------Convex Hull-----------#
    convexHull = cv2.convexHull(maskKeypoints)
    rect = (0, 0, maskFilter.shape[1], maskFilter.shape[0])
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(listOfMaskKeypoints)
    triangles = subdiv.getTriangleList()
    
    maskKeypointsToIndex = {point: index for index, point in enumerate(listOfMaskKeypoints)}
    trianglesIndexs = list(map(lambda points: [maskKeypointsToIndex[(points[0], points[1])], maskKeypointsToIndex[(points[2], points[3])], maskKeypointsToIndex[(points[4], points[5])]], triangles))

    #------------Affine Transform-----------#
    maskToFrame = np.zeros(frame.shape, dtype = np.uint8)
    for pts in trianglesIndexs:
        # Index of keypoints of triangle from delaunay triangulation
        p1 = np.array([listOfMaskKeypoints[pts[0]], listOfMaskKeypoints[pts[1]], listOfMaskKeypoints[pts[2]]])
        p2 = np.array([listOfFrameKeypoints[pts[0]], listOfFrameKeypoints[pts[1]], listOfFrameKeypoints[pts[2]]])
        
        # Receive triangle part from mask to transform
        rect1 = cv2.boundingRect(p1)
        triangleImg1 = maskFilter[rect1[1]: rect1[1] + rect1[3], rect1[0]: rect1[0] + rect1[2]]
        maskZerosImg1 = np.zeros((rect1[3], rect1[2]), dtype = np.int8)

        p1InTriangleImg1 = np.copy(p1)
        p1InTriangleImg1[:, 0] = p1InTriangleImg1[:, 0] - rect1[0]
        p1InTriangleImg1[:, 1] = p1InTriangleImg1[:, 1] - rect1[1] 

        cv2.fillConvexPoly(maskZerosImg1, p1InTriangleImg1, 255)
        triangleImg1 = cv2.bitwise_and(triangleImg1, triangleImg1, mask = maskZerosImg1)

        # Receive triangle part from frame to transform
        rect2 = cv2.boundingRect(p2)
        triangleImg2 = frame[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]]
        maskZerosImg2 = np.zeros((rect2[3], rect2[2]), dtype = np.int8)
        
        p2InTriangleImg2 = np.copy(p2)
        p2InTriangleImg2[:, 0] = p2InTriangleImg2[:, 0] - rect2[0]
        p2InTriangleImg2[:, 1] = p2InTriangleImg2[:, 1] - rect2[1] 
        
        cv2.fillConvexPoly(maskZerosImg2, p2InTriangleImg2, 255)
        triangleImg2 = cv2.bitwise_and(triangleImg2, triangleImg2, mask = maskZerosImg2)
        
        # Apply Affine Transform
        transformMatrix = cv2.getAffineTransform(np.float32(p1InTriangleImg1), np.float32(p2InTriangleImg2))
        transformedImage = cv2.warpAffine(triangleImg1, transformMatrix, (rect2[2], rect2[3]), flags=cv2.INTER_NEAREST)

        maskTransformedImage = cv2.bitwise_or(cv2.bitwise_or(transformedImage[:, :, 0], transformedImage[:, :, 1]), transformedImage[:, :, 2])
        _, maskTransformedImage = cv2.threshold(maskTransformedImage, 1, 255, cv2.THRESH_BINARY_INV)

        maskToFrame[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]] = cv2.bitwise_and(
            maskToFrame[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]],
            maskToFrame[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]],
            mask = maskTransformedImage
        )
        maskToFrame[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]] = cv2.add(maskToFrame[rect2[1]: rect2[1] + rect2[3], rect2[0]: rect2[0] + rect2[2]], transformedImage)

    maskOfFrame = cv2.bitwise_or(cv2.bitwise_or(maskToFrame[:, :, 0], maskToFrame[:, :, 1]), maskToFrame[:, :, 2])
    _, maskOfFrame = cv2.threshold(maskOfFrame, 1, 255, cv2.THRESH_BINARY_INV)
    frame = cv2.bitwise_and(frame, frame, mask = maskOfFrame)

    if switch == True:
        maskToFrame = 255 - maskToFrame
        maskToFrame = cv2.bitwise_and(maskToFrame, maskToFrame, mask = cv2.bitwise_not(maskOfFrame))

    result = cv2.add(frame, maskToFrame)
    # #------------Show Frame-----------#
    
    # for pts in triangles:
    #     p = []
    #     p.append((int(pts[0]), int(pts[1])))
    #     p.append((int(pts[2]), int(pts[3])))
    #     p.append((int(pts[4]), int(pts[5])))
    #     p = np.array(p)
    #     cv2.polylines(maskFilter, [p], True, (0, 0, 0), 2)
        
    # for pts in trianglesIndexs:
    #     p = []
    #     p.append(listOfFrameKeypoints[pts[0]])
    #     p.append(listOfFrameKeypoints[pts[1]])
    #     p.append(listOfFrameKeypoints[pts[2]])
    #     p = np.array(p)
    #     cv2.polylines(maskFilter, [p], True, (0, 0, 0), 2)
        
    # cv2.rectangle(frame, (rect2[0], rect2[1]), (rect2[0] + rect2[2], rect2[1] + rect2[3]), (255, 0, 0), 4)
    # cv2.rectangle(maskFilter, (rect1[0], rect1[1]), (rect1[0] + rect1[2], rect1[1] + rect1[3]), (255, 0, 0), 4)
    # cv2.polylines(maskFilter, [convexHull], True, (0, 0, 255), 3)
    # #cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (255, 0, 0), 4)
    # draw(frameBoundingBox, frameKeypoints, frame)
    # draw(maskBoundingBox, maskKeypoints, maskFilter)
    # cv2.imshow("frame", frame)
    # cv2.imshow("mask", maskFilter)
    return result;

# ##------------------Image Testing----------------###
# box_predict.init_model()
# image = cv2.imread("./image.png")
# image = cv2.copyMakeBorder(image, 200, 100, 50, 50, cv2.BORDER_CONSTANT, value = (255, 255, 255))
# net = loadModel();
# predictBoundingBoxs = box_predict.predict(image = image);
# #print(predictBoundingBoxs)
# predictPointsOfAllFaces = predictKeypoints(image = image, boundingBoxs = predictBoundingBoxs, net = net);

# # width = predictBoundingBoxs[0][1] - predictBoundingBoxs[0][0]
# # predictBoundingBoxs[0][0] = int(predictBoundingBoxs[0][0] - width / 2)
# # predictBoundingBoxs[0][1] = int(predictBoundingBoxs[0][1] + width / 2)
# # height = predictBoundingBoxs[0][3] - predictBoundingBoxs[0][2]
# # predictBoundingBoxs[0][2] = int(predictBoundingBoxs[0][2] - height * 2 / 3)
# # predictBoundingBoxs[0][3] = int(predictBoundingBoxs[0][3] + height / 3)


# ###------------------Mask Testing-----------------###
# mask = cv2.imread("./mask.png")
# keypoints = np.array(json.load(open("keypoints_mask.json")))
# width = mask.shape[1]
# height = mask.shape[0]
# boundingBox = np.array([1, width - 1, 1, height - 1])

# for predictPoints in predictPointsOfAllFaces:
#     predictPoints = predictPoints.type(torch.int);
#     xPoints = predictPoints[:, 0].clone();
#     yPoints = predictPoints[:, 1].clone();
#     xPoints[xPoints >= image.shape[1] - 1] = image.shape[1] - 2;
#     yPoints[yPoints >= image.shape[0] - 1] = image.shape[0] - 2;
#     xPoints[xPoints < 0] = 0;
#     yPoints[yPoints < 0] = 0;

#     image[yPoints, xPoints] = (0, 0, 255);
#     image[yPoints + 1, xPoints] = (0, 0, 255)
#     image[yPoints, xPoints + 1] = (0, 0, 255)
#     image[yPoints + 1, xPoints + 1] = (0, 0, 255)
# for boudingBox in predictBoundingBoxs:
#     points = [(boudingBox[0], boudingBox[2]), (boudingBox[1], boudingBox[3])];
#     image = cv2.rectangle(image, points[0], points[1], (255, 0, 0), 2);

# cv2.imshow("mask ori", mask)
# cv2.imshow("image ori", image)

# result = applyMaskFilter(image, predictBoundingBoxs[0], predictPointsOfAllFaces[0], mask, boundingBox, keypoints);

# cv2.imshow("res", result)
# cv2.waitKey(0);
# cv2.destroyAllWindows();
    