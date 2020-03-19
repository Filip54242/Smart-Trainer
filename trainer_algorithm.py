import numpy as np
import torch

from common.camera import normalize_screen_coordinates, camera_to_world
from common.custom_dataset import CustomDataset
from common.generators import UnchunkedGenerator
from common.model import TemporalModel
from common.utils import npy_to_poses
from pose import make_pose


class Predictor:
    def __init__(self, dataset_path, checkpoint_path, input_video_path=None, export_path=None, output_path=None):
        self.dataset_path = dataset_path
        self.export_path = export_path
        self.output_path = output_path
        self.input_video_path = input_video_path
        self.dataset = CustomDataset(self.dataset_path)
        self.keypoints = None
        self.keypoints_left = None
        self.keypoints_right = None
        self.joints_left = None
        self.joints_right = None
        self.checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.model = None
        self.init_keypoints()
        self.valid_poses = self.keypoints["detectron2"]["custom"]
        self.init_model()
        self.test_generator = None
        self.init_generator()
        self.prediction = None
        self.make_prediction()

    def export_prediction(self):
        if self.export_path is not None:
            np.save(self.export_path, self.prediction)

    def init_model(self):
        self.model = TemporalModel(self.valid_poses[0].shape[-2], self.valid_poses[0].shape[-1],
                                   self.dataset.skeleton().num_joints(),
                                   filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25,
                                   channels=1024,
                                   dense=False)
        self.model.load_state_dict(self.checkpoint['model_pos'])

    def init_keypoints(self):
        self.keypoints = np.load(self.dataset_path, allow_pickle=True)
        keypoints_metadata = self.keypoints['metadata'].item()
        keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
        self.keypoints_left, self.keypoints_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
        self.joints_left, self.joints_right = list(self.dataset.skeleton().joints_left()), list(
            self.dataset.skeleton().joints_right())
        self.keypoints = self.keypoints['positions_2d'].item()

        for subject in self.keypoints.keys():
            for action in self.keypoints[subject]:
                for cam_idx, kps in enumerate(self.keypoints[subject][action]):
                    # Normalize camera frame
                    cam = self.dataset.cameras()[subject][cam_idx]
                    kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
                    self.keypoints[subject][action][cam_idx] = kps

    def init_generator(self):
        receptive_field = self.model.receptive_field()
        pad = (receptive_field - 1) // 2
        causal_shift = 0
        self.test_generator = UnchunkedGenerator(None, None, self.valid_poses,
                                                 pad=pad, causal_shift=causal_shift, augment=False,
                                                 kps_left=self.keypoints_left, kps_right=self.keypoints_right,
                                                 joints_left=self.joints_left,
                                                 joints_right=self.joints_right)

    def make_prediction(self):
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        with torch.no_grad():
            self.model.eval()
            for _, batch, batch_2d in self.test_generator.next_epoch():
                inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
                if torch.cuda.is_available():
                    inputs_2d = inputs_2d.cuda()

            predicted_3d_pos = self.model(inputs_2d)

            if self.test_generator.augment_enabled():
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, self.joints_left + self.joints_right] = predicted_3d_pos[1, :,
                                                                               self.joints_right + self.joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            predicted_3d_pos = predicted_3d_pos.squeeze(0).cpu().numpy()
            rot = self.dataset.cameras()['detectron2'][0]['orientation']
            predicted_3d_pos = camera_to_world(predicted_3d_pos, R=rot, t=0)
            predicted_3d_pos[:, :, 2] -= np.min(predicted_3d_pos[:, :, 2])
            self.prediction = predicted_3d_pos

    def plot_pose(self, pose_index=0):
        pose = make_pose(self.prediction.tolist()[pose_index])
        pose.prepare_plot()
        pose.plot()


# poses=npy_to_poses("./predictions/baseball_1.npy")

pred = Predictor('./data/data_2d_custom_baseball_me.npz',
                 './checkpoint/Model_3D.bin',
                 export_path="/home/filip/Documents/Repos/Smart-Trainer/predictions/baseball_me")
pred.export_prediction()

pred.plot_pose()
