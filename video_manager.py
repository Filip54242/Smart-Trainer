import cv2
import matplotlib.pyplot as plt


class VideoManager:
    def __init__(self):
        self.frames = None

    def get_video(self, path):
        self.frames = []
        capture = cv2.VideoCapture(path)
        good, frame = capture.read()
        while good:
            self.frames.append(frame)
            good, frame = capture.read()
        capture.release()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, item):
        if item > len(self)-1:
            return None
        return self.frames[item]

    def export_video(self, path, framerate=30):
        resolution = (self.frames[0].shape[1], self.frames[0].shape[0])
        output = cv2.VideoWriter(path, -1, framerate, resolution)
        for frame in self.frames:
            output.write(frame)
        output.release()
