from common.camera import *
from common.custom_dataset import CustomDataset
from common.generators import UnchunkedGenerator
from common.loss import *
from common.model import TemporalModel
from common.utils import deterministic_random
from pose import *

custom_dataset = './data/data_2d_custom_baseball_3.npz'
output_path = None
chk_filename = './checkpoint/Model_3D.bin'
input_video_path = './inputs/baseball_1.mp4'
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

cameras_valid, poses_valid, poses_valid_2d = None, None, keypoints["detectron2"]["custom"]

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


def evaluate(test_generator):
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

            return predicted_3d_pos.squeeze(0).cpu().numpy()


print('Rendering...')

input_keypoints = keypoints['detectron2']['custom'][0].copy()
ground_truth = None
gen = UnchunkedGenerator(None, None, [input_keypoints],
                         pad=pad, causal_shift=causal_shift, augment=True,
                         kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)
prediction = evaluate(gen)

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

plot_list = prediction.tolist()[100]
prediction_one = prediction.tolist()[100]
prediction_two = prediction.tolist()[200]
pose1 = make_pose(prediction_one)
pose2 = make_pose(prediction_two)
print(str(pose1.pose_distance(pose2)))
pose1.compute_corrections(pose1 - pose2)

skeleton = dataset.skeleton()._children
lines = []

for index in range(17):
    for joint in skeleton[index]:
        lines.append([plot_list[index], plot_list[joint]])

plot_x, plot_y, plot_z = [element[0] for element in plot_list], [element[1] for element in plot_list], [
    element[2] for element in plot_list]

fig = plt.figure()
ax = Axes3D(fig)
#pose1.prepare_plot(ax)
pose2.prepare_plot(ax)
pose2.plot()

ax.scatter(plot_x, plot_y, plot_z)
for index in range(17):
    ax.text(plot_x[index], plot_y[index], plot_z[index], str(index), 'z')
for line in lines:
    ax.plot([line[0][0], line[1][0]], [line[0][1], line[1][1]], [line[0][2], line[1][2]])
plt.show()
