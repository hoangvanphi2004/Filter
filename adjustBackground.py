import numpy as np
import cv2
from sys import argv;


class AdjustBackground:
    def __init__(self, path, range) -> None:
        self.image = cv2.imread(path)
        self.image = np.array(self.image)
        self.image = self.image[:, :, :3]
        self.range = range
        ten_image = np.full(self.image.shape, 10, dtype = np.uint8)
        if((self.image[0, 0] == (255, 255, 255)).all()):
            self.image = cv2.add(self.image, ten_image)
        else:
            self.image = cv2.subtract(self.image, ten_image)

        self.render_image = np.copy(self.image)

    def onMouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            if((self.image[0, 0] == (255, 255, 255)).all()):
                cv2.floodFill(self.image, None, (x, y), (255, 255, 255), loDiff = (int(argv[1]), int(argv[1]), int(argv[1])), upDiff = (int(argv[1]), int(argv[1]), int(argv[1])), flags = cv2.FLOODFILL_FIXED_RANGE)
            else:
                cv2.floodFill(self.image, None, (x, y), (0, 0, 0), loDiff = (int(argv[1]), int(argv[1]), int(argv[1])), upDiff = (int(argv[1]), int(argv[1]), int(argv[1])), flags = cv2.FLOODFILL_FIXED_RANGE)
            cv2.floodFill(self.render_image, None, (x, y), (120, 120, 120), loDiff = (int(argv[1]), int(argv[1]), int(argv[1])), upDiff = (int(argv[1]), int(argv[1]), int(argv[1])), flags = cv2.FLOODFILL_FIXED_RANGE)
    def run(self):
        cv2.imshow('frame', self.render_image)
        while True:
            cv2.setMouseCallback('frame', self.onMouse)
            cv2.imshow('frame', self.render_image)

            k = cv2.waitKey(1);
            if k == ord('q'):
                break
        
        cv2.imwrite("mask.png", self.image)
        cv2.destroyAllWindows()

run = AdjustBackground(argv[2], int(argv[1]))
run.run()

# image = Image.open("./mask3.jpg")
# image = image.convert("RGB")
# image = np.array(image, dtype = np.uint8)
# if("remove-black" in argv):
#     ten_image = np.full(image.shape, 10, dtype = np.uint8)
#     image = cv2.add(image, ten_image)
#     if(len(argv) == 3):
#         cv2.floodFill(image, None, (0, 0), (0, 0, 0), loDiff = (int(argv[2]), int(argv[2]), int(argv[2])), upDiff = (int(argv[2]), int(argv[2]), int(argv[2])), flags = cv2.FLOODFILL_FIXED_RANGE)
# if("remove-white" in argv):
#     ten_image = np.full(image.shape, 10, dtype = np.uint8)
#     image = cv2.subtract(image, ten_image)
#     if(len(argv) == 3):
#         cv2.floodFill(image, None, (0, 0), (255, 255, 255), loDiff = (int(argv[2]), int(argv[2]), int(argv[2])), upDiff = (int(argv[2]), int(argv[2]), int(argv[2])), flags = cv2.FLOODFILL_FIXED_RANGE)
# image = Image.fromarray(image)
# image.save("mask2.jpg")