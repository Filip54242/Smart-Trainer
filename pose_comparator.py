import numpy as np
import cv2
from common.utils import npy_to_poses
from video_manager import VideoManager
import matplotlib.pyplot as plt


def plot_frames(first_frame, second_frame):
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    second_frame = cv2.cvtColor(second_frame, cv2.COLOR_BGR2RGB)
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(first_frame)
    f.add_subplot(1, 2, 2)
    plt.imshow(second_frame)
    plt.show(block=True)


class Comparator:
    DEFAULT_WEIGHTS = {"BOTTOM SPINE": 0.1,
                       "RIGHT HIP": 1,
                       "RIGHT KNEE": 1,
                       "RIGHT ANKLE": 1,
                       "LEFT HIP": 1,
                       "LEFT KNEE": 1,
                       "LEFT ANKLE": 1,
                       "MIDDLE SPINE": 0.1,
                       "TOP SPINE": 0.1,
                       "MIDDLE HEAD": 0.1,
                       "TOP HEAD": 0.1,
                       "LEFT SHOULDER": 1,
                       "LEFT ELBOW": 1,
                       "LEFT HAND": 1,
                       "RIGHT SHOULDER": 1,
                       "RIGHT ELBOW": 1,
                       "RIGHT HAND": 1}

    def __init__(self, good_pose, bad_pose, good_pose_video=None, bad_pose_video=None, weights=None):
        self.good_poses = npy_to_poses(good_pose)
        self.bad_poses = npy_to_poses(bad_pose)
        self.weights = list(self.DEFAULT_WEIGHTS.values()) if weights is None else weights
        self.good_poses_video = None
        self.bad_poses_video = None
        if good_pose_video is not None:
            self.good_poses_video = VideoManager()
            self.good_poses_video.get_video(good_pose_video)
        if bad_pose_video is not None:
            self.bad_poses_video = VideoManager()
            self.bad_poses_video.get_video(bad_pose_video)

    def compute_pose_distance(self, pose_1, pose_2):
        if len(pose_1) != len(pose_2):
            return float("inf")
        distance = 0
        for index in range(len(pose_1)):
            distance += pose_1[index].euclidian_distance(pose_2[index]) * self.weights[index]
        return distance

    def compare_poses(self, treshold=4, frameskip=10):
        good_pose_indexes = list(range(len(self.good_poses)))
        for index_1 in range(0, len(self.bad_poses), frameskip):
            min_value = treshold + 1
            min_index = 0
            for index_2 in good_pose_indexes:
                distance = self.compute_pose_distance(self.bad_poses[index_1], self.good_poses[index_2])
                min_value, min_index = (distance, index_2) if distance < min_value else (min_value, min_index)
            if min_value > treshold:
                continue
            good_pose_indexes.remove(min_index)
            self.bad_poses[index_1].compute_corrections(self.good_poses[min_index])
            if self.good_poses_video is not None and self.bad_poses_video is not None:
                print(str(index_1), "+", str(min_index))
                plot_frames(self.good_poses_video[min_index], self.bad_poses_video[index_1])
