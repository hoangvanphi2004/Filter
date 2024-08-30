import numpy as np
import cv2 as cv
import json

class CreatePointsOnMask:
    def __init__(self, path) -> None:
        self.recentPoint = [0, 0]
        self.keypoints = [];
        self.image = cv.imread(path)

    def draws(self, image, keypoints):
        cnt = 0;
        for keypoint in keypoints:
            cnt += 1;
            cv.circle(image, (int(keypoint[0]), int(keypoint[1])), radius = 2, color = (0, 255, 0), thickness = 2)

    def onMouse(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            self.keypoints.append([x, y])
            print(len(self.keypoints))

    def run(self):
        cv.imshow('frame', self.image)
        while True:
            recent = np.copy(self.image)

            cv.setMouseCallback('frame', self.onMouse)
            self.draws(recent, self.keypoints)
            cv.imshow('frame', recent)

            k = cv.waitKey(1);
            if k == ord('q'):
                break
            if k == ord('r'):
                self.keypoints = []
            if k == ord('d'):
                self.keypoints = self.keypoints[:-1]

            if len(self.keypoints) == 68:
                with open("keypoints_mask.json", 'w') as f:
                    json.dump(self.keypoints, f)
                break
        cv.destroyAllWindows()

run = CreatePointsOnMask("mask.png")
run.run()