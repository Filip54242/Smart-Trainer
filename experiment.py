import numpy as np
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from video_manager import VideoManager
from pose_comparator import Comparator

comp = Comparator(good_pose='./predictions/baseball_1.npy',
                  bad_pose='./predictions/baseball_me_2.npy',
                  good_pose_video='./inputs/baseball_1.mp4',
                  bad_pose_video='./inputs/me_2.mp4',
                  good_bboxes='/home/filip/Documents/Repos/Smart-Trainer/predictions/baseball_1_metadata.npy',
                  bad_bboxes='/home/filip/Documents/Repos/Smart-Trainer/predictions/baseball_me_2_metadata.npy')
comp.compare_poses()

