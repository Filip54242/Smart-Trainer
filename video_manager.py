import cv2


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

    def __getitem__(self, item):
        return self.frames[item]

    def export_video(self, path, framerate=30):
        resolution = (self.frames[0].shape[1], self.frames[0].shape[0])
        output = cv2.VideoWriter(path, -1, framerate, resolution)
        for frame in self.frames:
            output.write(frame)
        output.release()

    def show_frame(self, index=0, title=None):
        frame_title = 'Frame ' + str(index) if title is None else title
        cv2.namedWindow(frame_title)
        cv2.imshow(frame_title, self.frames[index])
        cv2.waitKey(0)

