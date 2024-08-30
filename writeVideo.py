import cv2 as cv
from datetime import datetime

class VideoWriter:
    def __init__(self) -> None:
        now = datetime.now()
        date = now.strftime("%d_%m_%Y_%H_%M_%S")
        filename = "./output/video_" + date + ".avi"
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        self.video = cv.VideoWriter(filename, fourcc, 12.0, (1280,  720))
    def write(self, frame):
        self.video.write(frame);
    def release(self):
        self.video.release()