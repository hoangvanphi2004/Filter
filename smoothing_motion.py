import numpy as np
import torch
import cv2 as cv

kalman_fil_for_keypoints = []
kalman_fil_for_boxs = []

def create_kalman_filter(num_of_faces, argv):
    global kalman_fil_for_keypoints, kalman_fil_for_boxs
    kalman_fil_for_keypoints = [[None for i in range(68)] for i in range(num_of_faces)]
    kalman_fil_for_boxs = [[None for i in range(4)] for i in range(num_of_faces)]
    for i in range(num_of_faces):
        for j in range(68):
            kalman_fil_for_keypoints[i][j] = cv.KalmanFilter(4, 2)
            kalman_fil_for_keypoints[i][j].measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kalman_fil_for_keypoints[i][j].transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0.8 if "floating-mask" not in argv else 1, 0], [0, 0, 0, 0.8 if "floating-mask" not in argv else 1]], np.float32
            )
            kalman_fil_for_keypoints[i][j].processNoiseCov = (
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * (0.0001 if "floating-mask" in argv else 1)
            )
    for i in range(num_of_faces):
        for j in range(4):
            kalman_fil_for_boxs[i][j] = cv.KalmanFilter(4, 2)
            kalman_fil_for_boxs[i][j].measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
            kalman_fil_for_boxs[i][j].transitionMatrix = np.array(
                [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 0.8 if "floating-mask" not in argv else 1, 0], [0, 0, 0, 0.8 if "floating-mask" not in argv else 1]], np.float32
            )
            kalman_fil_for_boxs[i][j].processNoiseCov = (
                np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * (0.0001 if "floating-mask" in argv else 1)
            )
    
def kalman_filter_update(predictPointsOfAllFacesInNumpy, predictBoxs):
    result_key_points = [[]for i in range(len(kalman_fil_for_keypoints))]
    for i in range(len(kalman_fil_for_keypoints)):
        for j in range(68):
            kalman_fil_for_keypoints[i][j].correct(np.array(predictPointsOfAllFacesInNumpy[i, j].T, dtype = np.float32))
            tp = kalman_fil_for_keypoints[i][j].predict()
            result_key_points[i].append([int(tp[0]), int(tp[1])])

    result_boxs = []

    for i in range(len(kalman_fil_for_boxs)):
        predictBoxsInPoints = np.array([[predictBoxs[i, 0], predictBoxs[i, 2]],
                                        [predictBoxs[i, 1], predictBoxs[i, 2]],
                                        [predictBoxs[i, 0], predictBoxs[i, 3]],
                                        [predictBoxs[i, 1], predictBoxs[i, 3]]]);
        tp = []
        for j in range(4):
            kalman_fil_for_boxs[i][j].correct(np.array(predictBoxsInPoints[j].T, dtype = np.float32))
            tp.append(kalman_fil_for_boxs[i][j].predict())

        result_boxs.append([int(tp[0][0]), int(tp[1][0]), int(tp[0][1]), int(tp[2][1])])

    return torch.tensor(result_key_points), np.array(result_boxs)
    
#-------------------In experiment-------------------#    
def local_search(pointList1, pointList2, image, previousImage):
    padding = 1;
    pointList = np.stack([np.array(pointList1), np.array(pointList2)], axis = 1)
    pointList = np.array(pointList, dtype = np.int64)
    stabledPoints = []
    for point in pointList:
        previousPoint = point[0]
        local = image[point[1][1] - padding: point[1][1] + padding, point[1][0] - padding: point[1][0] + padding, :]
        previousColor = previousImage[previousPoint[1], previousPoint[0], :]
        
        # print(previousPoint)
        # print(local)
        # print(previousColor)
        
        # print(np.linalg.norm(np.absolute(local - previousColor), axis = 2))
        
        minDistance = np.min(np.linalg.norm(np.absolute(local - previousColor), axis = 2))
        index = np.where((np.linalg.norm(np.absolute(local - previousColor), axis = 2) == minDistance))
        #print(index)
        stabledPoints.append([int(index[1][0]) + point[1][0] - padding, int(index[0][0]) + point[1][1] - padding])
        #print(np.linalg.norm(np.absolute(local - previousColor), axis = 2))
    
    #print(stabledPoints)
    #print(pointList2)
    return torch.tensor([stabledPoints])