import os

import numpy as np
import torch

from common.camera import world_to_camera, normalize_screen_coordinates
from common.generators import UnchunkedGenerator
from common.loss import mpjpe, mean_velocity_error, p_mpjpe, n_mpjpe
from common.model import TemporalModelOptimized1f, TemporalModel
from common.utils import deterministic_random

args = {'actions': '*',
        'architecture': '3,3,3,3,3',
        'batch_size': 1024,
        'bone_length_term': True,
        'by_subject': False,
        'causal': False,
        'channels': 1024,
        'checkpoint': 'checkpoint',
        'checkpoint_frequency': 10,
        'data_augmentation': True,
        'dataset': 'h36m',
        'dense': False,
        'disable_optimizations': False,
        'downsample': 1,
        'dropout': 0.25,
        'epochs': 90,
        'evaluate': '/home/filip/Documents/VideoPose3D-master/checkpoint/epoch_110.bin',
        'export_training_curves': False,
        'keypoints': 'cpn_ft_h36m_dbb',
        'learning_rate': 0.001,
        'linear_projection': False,
        'lr_decay': 0.95,
        'no_eval': False,
        'no_proj': False,
        'render': False,
        'resume': '',
        'stride': 1,
        'subjects_test': 'S9,S11',
        'subjects_train': 'S1,S5,S6,S7,S8',
        'subjects_unlabeled': '',
        'subset': 1,
        'test_time_augmentation': True,
        'viz_action': None,
        'viz_bitrate': 3000,
        'viz_camera': 0,
        'viz_downsample': 1,
        'viz_export': None,
        'viz_limit': -1,
        'viz_no_ground_truth': False,
        'viz_output': None,
        'viz_size': 5,
        'viz_skip': 0,
        'viz_subject': None,
        'viz_video': None,
        'warmup': 1}

dataset_path = 'data/data_3d_' + args['dataset'] + '.npz'
if args['dataset'] == 'h36m':
    from common.h36m_dataset import Human36mDataset

    dataset = Human36mDataset(dataset_path)
elif args['dataset'].startswith('humaneva'):
    from common.humaneva_dataset import HumanEvaDataset

    dataset = HumanEvaDataset(dataset_path)

print('Preparing data...')
for subject in dataset.subjects():
    for action in dataset[subject].keys():
        anim = dataset[subject][action]

        if 'positions' in anim:
            positions_3d = []
            for cam in anim['cameras']:
                pos_3d = world_to_camera(anim['positions'], R=cam['orientation'], t=cam['translation'])
                pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
                positions_3d.append(pos_3d)
            anim['positions_3d'] = positions_3d

print('Loading 2D detections...')
# keypoints = np.load('data/data_2d_' + args['dataset'] + '_' + args.keypoints + '.npz', allow_pickle=True)
keypoints = np.load('/home/filip/Documents/VideoPose3D-master/data/data_2d_h36m_gt.npz', allow_pickle=True)
keypoints_metadata = keypoints['metadata'].item()
keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
keypoints = keypoints['positions_2d'].item()

for subject in dataset.subjects():
    assert subject in keypoints, 'Subject {} is missing from the 2D detections dataset'.format(subject)
    for action in dataset[subject].keys():
        assert action in keypoints[subject], 'Action {} of subject {} is missing from the 2D detections dataset'.format(
            action, subject)
        if 'positions_3d' not in dataset[subject][action]:
            continue

        for cam_idx in range(len(keypoints[subject][action])):

            # We check for >= instead of == because some videos in H3.6M contain extra frames
            mocap_length = dataset[subject][action]['positions_3d'][cam_idx].shape[0]
            assert keypoints[subject][action][cam_idx].shape[0] >= mocap_length

            if keypoints[subject][action][cam_idx].shape[0] > mocap_length:
                # Shorten sequence
                keypoints[subject][action][cam_idx] = keypoints[subject][action][cam_idx][:mocap_length]

        assert len(keypoints[subject][action]) == len(dataset[subject][action]['positions_3d'])

for subject in keypoints.keys():
    for action in keypoints[subject]:
        for cam_idx, kps in enumerate(keypoints[subject][action]):
            # Normalize camera frame
            cam = dataset.cameras()[subject][cam_idx]
            kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
            keypoints[subject][action][cam_idx] = kps

subjects_train = args['subjects_train'].split(',')
subjects_semi = [] if not args['subjects_unlabeled'] else args['subjects_unlabeled'].split(',')
subjects_test = args['subjects_test'].split(',')


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


cameras_valid, poses_valid, poses_valid_2d = fetch(subjects_test, None)

filter_widths = [int(x) for x in args['architecture'].split(',')]
model_pos_train = TemporalModelOptimized1f(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1],
                                           dataset.skeleton().num_joints(),
                                           filter_widths=filter_widths, causal=args['causal'], dropout=args['dropout'],
                                           channels=args['channels'])

model_pos = TemporalModel(poses_valid_2d[0].shape[-2], poses_valid_2d[0].shape[-1], dataset.skeleton().num_joints(),
                          filter_widths=filter_widths, causal=args['causal'], dropout=args['dropout'],
                          channels=args['channels'], dense=args['dense'])
causal_shift = 0

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

if args['resume'] or args['evaluate']:
    chk_filename = os.path.join(args['checkpoint'], args['resume'] if args['resume'] else args['evaluate'])
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])

test_generator = UnchunkedGenerator(cameras_valid, poses_valid, poses_valid_2d,
                                    pad=121, causal_shift=causal_shift, augment=False,
                                    kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                    joints_right=joints_right)
print('Evaluating...')
all_actions = {}
all_actions_by_subject = {}
for subject in subjects_test:
    if subject not in all_actions_by_subject:
        all_actions_by_subject[subject] = {}

    for action in dataset[subject].keys():
        action_name = action.split(' ')[0]
        if action_name not in all_actions:
            all_actions[action_name] = []
        if action_name not in all_actions_by_subject[subject]:
            all_actions_by_subject[subject][action_name] = []
        all_actions[action_name].append((subject, action))
        all_actions_by_subject[subject][action_name].append((subject, action))


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


def fetch_actions(actions):
    out_poses_3d = []
    out_poses_2d = []

    for subject, action in actions:
        poses_2d = keypoints[subject][action]
        for i in range(len(poses_2d)):  # Iterate across cameras
            out_poses_2d.append(poses_2d[i])

        poses_3d = dataset[subject][action]['positions_3d']
        assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
        for i in range(len(poses_3d)):  # Iterate across cameras
            out_poses_3d.append(poses_3d[i])

    stride = args['downsample']
    if stride > 1:
        # Downsample as requested
        for i in range(len(out_poses_2d)):
            out_poses_2d[i] = out_poses_2d[i][::stride]
            if out_poses_3d is not None:
                out_poses_3d[i] = out_poses_3d[i][::stride]

    return out_poses_3d, out_poses_2d


def run_evaluation(actions, action_filter=None):
    errors_p1 = []
    errors_p2 = []
    errors_p3 = []
    errors_vel = []

    for action_key in actions.keys():
        if action_filter is not None:
            found = False
            for a in action_filter:
                if action_key.startswith(a):
                    found = True
                    break
            if not found:
                continue

        poses_act, poses_2d_act = fetch_actions(actions[action_key])
        gen = UnchunkedGenerator(None, poses_act, poses_2d_act,
                                 pad=121, causal_shift=causal_shift, augment=args['test_time_augmentation'],
                                 kps_left=kps_left, kps_right=kps_right, joints_left=joints_left,
                                 joints_right=joints_right)
        e1, e2, e3, ev = evaluate(gen, action_key)
        errors_p1.append(e1)
        errors_p2.append(e2)
        errors_p3.append(e3)
        errors_vel.append(ev)
    print('Protocol #1   (MPJPE) action-wise average:', round(np.mean(errors_p1), 1), 'mm')
    print('Protocol #2 (P-MPJPE) action-wise average:', round(np.mean(errors_p2), 1), 'mm')
    print('Protocol #3 (N-MPJPE) action-wise average:', round(np.mean(errors_p3), 1), 'mm')
    print('Velocity      (MPJVE) action-wise average:', round(np.mean(errors_vel), 2), 'mm')


run_evaluation(all_actions)
