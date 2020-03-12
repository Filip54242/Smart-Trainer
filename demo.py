from common.camera import *
from common.custom_dataset import CustomDataset
from common.generators import UnchunkedGenerator
from common.loss import *
from common.model import TemporalModel
from common.utils import deterministic_random
from common.visualization import render_animation

custom_dataset = '/home/filip/Documents/VideoPose3D-master/data/data_2d_custom_baseball_george_2.npz'
output_path = "/home/filip/Documents/VideoPose3D-master/baseball_george_1.mp4"
chk_filename = '/home/filip/Documents/VideoPose3D-master/checkpoint/Model_3D.bin'
input_video_path = '/home/filip/Documents/VideoPose3D-master/inputs/baseball_george_2.mp4'
export_path = None

dataset = CustomDataset(custom_dataset)

print('Loading 2D detections...')
keypoints = np.load(custom_dataset, allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps


def fetch(subjects, action_filter=None, subset=1, parse_3d_poses=True):
    out_poses_3d = []
    out_poses_2d = []
    out_camera_params = []
    for subject in subjects:
        for action in keypoints[subject].keys():
            if action_filter is not None:
                found = False
                for a in action_filter:
                    if action.startswith(a):
                        found = True
                        break
                if not found:
                    continue

            poses_2d = keypoints[subject][action]
            for i in range(len(poses_2d)):  # Iterate across cameras
                out_poses_2d.append(poses_2d[i])

            if subject in dataset.cameras():
                cams = dataset.cameras()[subject]
                assert len(cams) == len(poses_2d), 'Camera count mismatch'
                for cam in cams:
                    if 'intrinsic' in cam:
                        out_camera_params.append(cam['intrinsic'])

            if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
                poses_3d = dataset[subject][action]['positions_3d']
                assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
                for i in range(len(poses_3d)):  # Iterate across cameras
                    out_poses_3d.append(poses_3d[i])

    if len(out_camera_params) == 0:
        out_camera_params = None
    if len(out_poses_3d) == 0:
        out_poses_3d = None

    stride = 1
    if subset < 1:
        for i in range(len(out_poses_2d)):
            n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
            start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
            out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    elif stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_camera_params, out_poses_3d, out_poses_2d


cameras_valid, poses_valid, poses_valid_2d = fetch(['detectron2'], None)

model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                          filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25,
                          channels=1024,
                          dense=False)

receptive_field = model_pos.receptive_field()
pad = (receptive_field - 1) // 2
causal_shift = 0
if torch.cuda.is_available():
    model_pos = model_pos.cuda()

checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
model_pos.load_state_dict(checkpoint['model_pos'])

test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=pad, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                    joints_right=joints_right)


def evaluate(test_generator, action=None, return_predictions=False):
    epoch_loss_3d_pos = 0
    epoch_loss_3d_pos_procrustes = 0
    epoch_loss_3d_pos_scale = 0
    epoch_loss_3d_vel = 0
    with torch.no_grad():
        model_pos.eval()
        N = 0
        for _, batch, batch_2d in test_generator.next_epoch():
            inputs_2d = torch.from_numpy(batch_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            # Positional model
            predicted_3d_pos = model_pos(inputs_2d)

            # Test-time augmentation (if enabled)
            if test_generator.augment_enabled():
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

            if return_predictions:
                return predicted_3d_pos.squeeze(0).cpu().numpy()

            inputs_3d = torch.from_numpy(batch.astype('float32'))
            if torch.cuda.is_available():
                inputs_3d = inputs_3d.cuda()
            inputs_3d[:, :, 0] = 0
            if test_generator.augment_enabled():
                inputs_3d = inputs_3d[:1]

            error = mpjpe(predicted_3d_pos, inputs_3d)
            epoch_loss_3d_pos_scale += inputs_3d.shape[0] * inputs_3d.shape[1] * n_mpjpe(predicted_3d_pos,
                                                                                         inputs_3d).item()

            epoch_loss_3d_pos += inputs_3d.shape[0] * inputs_3d.shape[1] * error.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])

            epoch_loss_3d_pos_procrustes += inputs_3d.shape[0] * inputs_3d.shape[1] * p_mpjpe(predicted_3d_pos, inputs)

            # Compute velocity error
            epoch_loss_3d_vel += inputs_3d.shape[0] * inputs_3d.shape[1] * mean_velocity_error(predicted_3d_pos, inputs)

    if action is None:
        print('----------')
    else:
        print('----' + action + '----')
    e1 = (epoch_loss_3d_pos / N) * 1000
    e2 = (epoch_loss_3d_pos_procrustes / N) * 1000
    e3 = (epoch_loss_3d_pos_scale / N) * 1000
    ev = (epoch_loss_3d_vel / N) * 1000
    print('Test time augmentation:', test_generator.augment_enabled())
    print('Protocol #1 Error (MPJPE):', e1, 'mm')
    print('Protocol #2 Error (P-MPJPE):', e2, 'mm')
    print('Protocol #3 Error (N-MPJPE):', e3, 'mm')
    print('Velocity Error (MPJVE):', ev, 'mm')
    print('----------')

    return e1, e2, e3, ev


print('Rendering...')

input_keypoints = keypoints['detectron2']['custom'][0].copy()
ground_truth = None
gen = UnchunkedGenerator(None, None, [input_keypoints],
                         pad=pad, causal_shift=causal_shift, augment=True,
                         kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
prediction = evaluate(gen, return_predictions=True)

if export_path is not None:
    print('Exporting joint positions to', export_path)
    # Predictions are in camera space
    np.save(export_path, prediction)

    # Invert camera transformation
cam = dataset.cameras()['detectron2'][0]

# If the ground truth is not available, take the camera extrinsic params from a random subject.
# They are almost the same, and anyway, we only need this for visualization purposes.
for subject in dataset.cameras():
    if 'orientation' in dataset.cameras()[subject][0]:
        rot = dataset.cameras()[subject][0]['orientation']
        break
prediction = camera_to_world(prediction, R=rot, t=0)
# We don't have the trajectory, but at least we can rebase the height
prediction[:, :, 2] -= np.min(prediction[:, :, 2])

anim_output = {'Reconstruction': prediction}

input_keypoints = image_coordinates(input_keypoints[..., :2], w=cam['res_w'], h=cam['res_h'])

render_animation(input_keypoints, keypoints_metadata, anim_output,
                 dataset.skeleton(), dataset.fps(), 300, cam['azimuth'], output_path,
                 limit=-1, downsample=1, size=6,
                 input_video_path=input_video_path, viewport=(cam['res_w'], cam['res_h']),
                 input_video_skip=0)
