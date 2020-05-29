import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from common.utils import npy_to_poses
from video_manager import VideoManager
import matplotlib.pyplot as plt
from pose import *


def plot_frames(first_frame, second_frame):
    first_frame = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
    second_frame = cv2.cvtColor(second_frame, cv2.COLOR_BGR2RGB)
    f = plt.figure()
    f.add_subplot(1, 2, 1)
    plt.imshow(first_frame)
    f.add_subplot(1, 2, 2)
    plt.imshow(second_frame)
    plt.show(block=True)


def normalize_to_interval(array, x, y):
    range = max(array) - min(array)
    array = [(element - min(array)) / range for element in array]
    range = y - x
    array = [(element * range) + x for element in array]
    return array


class Comparator:
    DEFAULT_WEIGHTS = {"BOTTOM SPINE": 0,
                       "RIGHT HIP": 1,
                       "RIGHT KNEE": 1,
                       "RIGHT ANKLE": 1,
                       "LEFT HIP": 1,
                       "LEFT KNEE": 1,
                       "LEFT ANKLE": 1,
                       "MIDDLE SPINE": 0,
                       "TOP SPINE": 0,
                       "MIDDLE HEAD": 0,
                       "TOP HEAD": 0,
                       "LEFT SHOULDER": 1,
                       "LEFT ELBOW": 1,
                       "LEFT HAND": 1,
                       "RIGHT SHOULDER": 1,
                       "RIGHT ELBOW": 1,
                       "RIGHT HAND": 1}

    def __init__(self, good_pose, bad_pose, good_bboxes, bad_bboxes, good_pose_video=None, bad_pose_video=None,
                 weights=None):
        self.good_poses = npy_to_poses(good_pose)
        self.bad_poses = npy_to_poses(bad_pose)
        self.good_bboxes = (np.load(good_bboxes, allow_pickle=True))[0]
        self.bad_bboxes = np.load(bad_bboxes, allow_pickle=True)[0]
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

    def compare_poses(self, treshold=168, frameskip=1):
        sets=[]
        good_pose_indexes = list(range(len(self.good_poses)))
        # print(good_pose_indexes)
        for index_1 in range(0, len(self.bad_poses), frameskip):
            min_value = treshold + 1
            min_index = 0
            for index_2 in good_pose_indexes:
                distance = self.bad_poses[index_1].angle_similarity(self.good_poses[index_2])
                min_value, min_index = (distance, index_2) if distance < min_value else (min_value, min_index)
            if min_value > treshold:
                continue
            good_pose_indexes.remove(min_index)
            self.bad_poses[index_1].compute_corrections(self.good_poses[min_index])
            if self.good_poses_video is not None and self.bad_poses_video is not None:
                # print(str(index_1), "+", str(min_index), "=", str(min_value))
                if self.good_poses_video[min_index] is not None and self.bad_poses_video[index_1] is not None:
                    frame_1 = self.good_poses[min_index].put_on_image(self.good_poses_video[min_index],
                                                                      self.good_bboxes[min_index])
                    frame_2 = self.bad_poses[index_1].put_on_image(self.bad_poses_video[index_1],
                                                                   self.bad_bboxes[index_1])
                    #plot_frames(frame_1, frame_2)
                    sets.append([[frame_1, frame_2]])
        return sets

