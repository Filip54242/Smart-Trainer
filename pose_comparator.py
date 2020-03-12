from common.utils import npy_to_poses
from video_manager import VideoManager


class Comparator:
    def __init__(self, good_pose, bad_pose, good_pose_video=None, bad_pose_video=None):
        self.good_poses = npy_to_poses(good_pose)
        self.bad_poses = npy_to_poses(bad_pose)
        self.good_poses_video = None
        self.bad_poses_video = None
        if good_pose_video is not None:
            self.good_poses_video = VideoManager()
            self.good_poses_video.get_video(good_pose_video)
        if bad_pose_video is not None:
            self.bad_poses_video = VideoManager()
            self.bad_poses_video.get_video(bad_pose_video)

    def compare_poses(self, treshold=1.5):
        for index_1 in range(len(self.bad_poses)):
            min_value = 10
            min_index = 0
            for index_2 in range(len(self.good_poses)):
                distance = self.bad_poses[index_1].pose_distance(self.good_poses[index_2])
                min_value, min_index = (distance, index_2) if distance < min_value else (min_value, min_index)
            if min_value > treshold:
                continue
            self.bad_poses[index_1].compute_corrections(self.good_poses[min_index])
            if self.good_poses_video is not None:
                self.good_poses_video.show_frame(min_index)
            if self.bad_poses_video is not None:
                self.bad_poses_video.show_frame(index_1)
