import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from video_manager import VideoManager
from pose_comparator import Comparator

comp = Comparator(good_pose='./predictions/baseball_1.npy',
                  bad_pose='./predictions/baseball_george_1.npy',
                  good_pose_video='./inputs/baseball_1.mp4',
                  bad_pose_video='./inputs/baseball_george_1.mp4')
comp.compare_poses()

