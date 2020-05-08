from math import sqrt, acos, degrees

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import ImageDraw, Image
from mpl_toolkits.mplot3d import Axes3D


def joints_to_lines(joint_1, joint_2):
    line_1 = [[joint_1.x, joint_1.y], (joint_2.x, joint_2.y)]
    line_2 = [[joint_1.x, joint_1.z], (joint_2.x, joint_2.z)]
    line_3 = [[joint_1.y, joint_1.z], (joint_2.y, joint_2.z)]
    return line_1, line_2, line_3


def dot(first, second):
    return first[0] * second[0] + first[1] * second[1]


def make_pose(keypoints):
    return Pose(keypoints)


def normalize_to_interval(array, x, y):
    range = max(array) - min(array)
    array = [(element - min(array)) / range for element in array]
    range = y - x
    array = [(element * range) + x for element in array]
    return array


def angle(first_line, second_line):
    first_line = [(first_line[0][0] - first_line[1][0]), (first_line[0][1] - first_line[1][1])]
    second_line = [(second_line[0][0] - second_line[1][0]), (second_line[0][1] - second_line[1][1])]
    dot_prod = dot(first_line, second_line)
    dot_first = dot(first_line, first_line) ** 0.5
    dot_second = dot(second_line, second_line) ** 0.5
    angle = acos(dot_prod / dot_second / dot_first)
    ang_deg = degrees(angle) % 360

    if ang_deg - 180 >= 0:
        return 360 - ang_deg
    else:
        return ang_deg


class Joint:
    JOINTS_NAME = {0: "BOTTOM SPINE",
                   1: "RIGHT HIP",
                   2: "RIGHT KNEE",
                   3: "RIGHT ANKLE",
                   4: "LEFT HIP",
                   5: "LEFT KNEE",
                   6: "LEFT ANKLE",
                   7: "MIDDLE SPINE",
                   8: "TOP SPINE",
                   9: "MIDDLE HEAD",
                   10: "TOP HEAD",
                   11: "LEFT SHOULDER",
                   12: "LEFT ELBOW",
                   13: "LEFT HAND",
                   14: "RIGHT SHOULDER",
                   15: "RIGHT ELBOW",
                   16: "RIGHT HAND"}

    def __init__(self, x, y, z, name):
        self.x = x
        self.y = y
        self.z = z
        self.type = name

    def __str__(self):
        return self.JOINTS_NAME[self.type]

    def __sub__(self, other):
        return float(self.x - other.x), float(self.y - other.y), float(self.z - other.z)

    def euclidian_distance(self, other):
        return sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    def plot(self, axis):
        axis.scatter(self.x, self.y, self.z)
        axis.text(self.x, self.y, self.z, str(self.type), 'z')

    def plot_2d(self, axis):
        axis.scatter(self.x, self.z)


class Pose:
    JOINT_GROUPS = [[[10, 9], [9, 8]], [[8, 7], [7, 0]], [[0, 4], [4, 5], [5, 6]], [[0, 1], [0, 2], [0, 3]],
                    [[14, 15], [15, 16]], [[11, 12], [12, 13]]]
    SPINE_GROUP = [7, 8, 0]
    HEAD_GROUP = [9, 10]
    LIMB_GROUP = [1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 16]
    CORRECTIONS = [["UP", "DOWN"], ["FORWARD", "BACKWARD"], ["RIGHT", "LEFT"]]
    SKELETON = {0: [1, 4, 7],
                1: [2],
                2: [3],
                3: None,
                4: [5],
                5: [6],
                6: None,
                7: [8, 11, 14],
                8: [9],
                9: [10],
                10: None,
                11: [12],
                12: [13],
                13: None,
                14: [15],
                15: [16],
                16: None}

    def __init__(self, joints):
        self.joints = []
        for index in range(len(joints)):
            x, y, z = joints[index]
            self.joints.append(Joint(x, y, z, index))
        self.num_joints = len(self.joints)

    def __len__(self):
        return len(self.joints)

    def __sub__(self, other):
        differences = []
        for index in range(len(self.joints)):
            differences.append(self.joints[index] - other.joints[index])
        return differences

    def __getitem__(self, item):
        return self.joints[item]

    def to_array(self):
        return [[element.x for element in self], [element.y for element in self], [element.z for element in self]]

    def put_pose_on_image(self, image, bbox):
        arr = self.to_array()
        x = arr[0]
        y = arr[2]
        x = normalize_to_interval(x, bbox[2], bbox[0])
        y = normalize_to_interval(y, bbox[3], bbox[1])
        for key, value in self.SKELETON.items():
            if value is not None:
                for item in list(value):
                    cv2.line(image, (int(x[key]), int(y[key])), (int(x[item]), int(y[item])), (0, 0, 255), 15)
        return image

    def angle_similarity(self, other):
        similarity = 0
        for group in self.JOINT_GROUPS:
            for index in range(len(group) - 1):
                average = 0
                pose_1_line_1 = joints_to_lines(self.joints[group[index][0]], self.joints[group[index][1]])
                pose_1_line_2 = joints_to_lines(self.joints[group[index + 1][0]], self.joints[group[index + 1][1]])
                pose_2_line_1 = joints_to_lines(other.joints[group[index][0]], other.joints[group[index][1]])
                pose_2_line_2 = joints_to_lines(other.joints[group[index + 1][0]], other.joints[group[index + 1][1]])
                for index_2 in range(3):
                    average += abs(angle(pose_1_line_1[index_2], pose_1_line_2[index_2]) - angle(pose_2_line_1[index_2],
                                                                                                 pose_2_line_2[
                                                                                                     index_2]))
                similarity += average / 3
        return similarity

    def limbs_group_corrections(self, diff, treshold):
        for index in self.LIMB_GROUP:
            indexes = [None, None, None]
            for index_2 in range(len(diff[index])):
                if int(diff[index][index_2] * treshold) > 0:
                    indexes[index_2] = 0
                if int(diff[index][index_2] * treshold) < 0:
                    indexes[index_2] = 1
            directions = ''
            directions += self.CORRECTIONS[0][indexes[0]] + " " if indexes[0] is not None else ''
            directions += self.CORRECTIONS[1][indexes[1]] + " " if indexes[1] is not None else ''
            directions += self.CORRECTIONS[2][indexes[2]] + " " if indexes[2] is not None else ''
            if directions != '':
                print("MOVE YOUR  " + str(self.joints[index]) + " " + directions + '\n')

    def group_corrections(self, diff, group, treshold):
        if group == 'HEAD':
            iterate_group = self.HEAD_GROUP
        elif group == 'BACK':
            iterate_group = self.SPINE_GROUP
        else:
            return
        texts = set()
        for index in iterate_group:
            indexes = [None, None, None]
            for index_2 in range(len(diff[index])):
                if int(diff[index][index_2] * treshold) > 0:
                    indexes[index_2] = 0
                if int(diff[index][index_2] * treshold) < 0:
                    indexes[index_2] = 1
            texts.add(self.CORRECTIONS[0][indexes[0]] + " " if indexes[0] is not None else '')
            texts.add(self.CORRECTIONS[1][indexes[1]] + " " if indexes[1] is not None else '')
            texts.add(self.CORRECTIONS[2][indexes[2]] + " " if indexes[2] is not None else '')
        directions = ''
        for text in texts:
            directions += text
        if directions != '':
            print("MOVE YOUR " + group + ' ' + directions + '\n')

    def compute_corrections(self, other, treshold=50):
        diff = self - other
        self.limbs_group_corrections(diff, treshold)
        # self.group_corrections(diff, 'HEAD', treshold)
        # self.group_corrections(diff, 'BACK', treshold)

    def prepare_plot(self, axis=None):
        if axis is None:
            fig = plt.figure()
            axis = Axes3D(fig)
        for joint in self.joints:
            joint.plot(axis)

        for key, value in self.SKELETON.items():
            if value is not None:
                for index in value:
                    axis.plot([self.joints[key].x, self.joints[index].x], [self.joints[key].y, self.joints[index].y],
                              [self.joints[key].z, self.joints[index].z])

    def prepare_2d_plot(self, axis=None):
        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot()
        for joint in self.joints:
            joint.plot_2d(axis)

        for key, value in self.SKELETON.items():
            if value is not None:
                for index in value:
                    axis.plot([self.joints[key].x, self.joints[index].x], [self.joints[key].z, self.joints[index].z])

    def plot(self):
        plt.show()
