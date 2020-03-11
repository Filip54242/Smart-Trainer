from math import sqrt

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def make_pose(keypoints):
    joints = []
    for index in range(len(keypoints)):
        x, y, z = keypoints[index]
        joints.append(Joint(x, y, z, index))
    return Pose(joints)


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


class Pose:
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

    def __init__(self, joints=None):
        self.joints = joints
        self.num_joints = len(self.joints)

    def __sub__(self, other):
        differences = []
        for index in range(len(self.joints)):
            differences.append(self.joints[index] - other.joints[index])
        return differences

    def pose_distance(self, other):
        distance = 0
        for index in range(1, self.num_joints):
            distance += self.joints[index].euclidian_distance(other.joints[index])
        return distance

    def compute_corrections(self, diff, treshold=50):
        for index in range(1, self.num_joints):
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

    def plot(self):
        plt.show()
