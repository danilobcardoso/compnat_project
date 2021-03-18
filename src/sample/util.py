import numpy as np
from scipy.spatial.transform import Rotation as R

from sample.pose import Pose
from sample.pose_sequence import PoseSequence


def rotate_poses(poses, degree):
    new_poses = []
    for skeleton in poses:
        new_poses.append(rotate_skeleton(skeleton, degree))
    return new_poses


def rotate_skeleton(skeleton, degree):
    r = R.from_rotvec(np.deg2rad(degree) * np.array([0, 1, 0]))
    new_skeleton = Pose(skeleton.skeleton_model)
    new_skeleton.add_data('position_xyz', r.apply(skeleton.data['position_xyz']))
    return new_skeleton


def gaussian_noise_poses(poses, sigma):
    new_poses = []
    for skeleton in poses:
        new_poses.append(gaussian_noise_skeleton(skeleton, sigma))
    return new_poses


def gaussian_noise_skeleton(skeleton, sigma):
    mu = 0
    noise = np.random.normal(mu, sigma, skeleton.data['position_xyz'].shape)
    new_skeleton = Pose(skeleton.skeleton_model)
    new_skeleton.add_data('position_xyz', skeleton.data['position_xyz'] + noise)
    return new_skeleton


def scale_noise_poses(poses, noise_intensity):
    new_poses = []
    for skeleton in poses:
        new_poses.append(scale_noise_skeleton(skeleton, noise_intensity))
    return new_poses


def scale_noise_skeleton(skeleton, noise_intensity):
    new_skeleton = Pose(skeleton.skeleton_model)
    new_skeleton.add_data('position_xyz', skeleton.data['position_xyz'] * (1 + noise_intensity))
    return new_skeleton


def data_augmentation_by_action_sequence(action_sequence, rotate=True, noise=True, rotation_angle=50, noise_intensity=0.1, group='N/A'):
    angle = np.random.randint(rotation_angle) - (rotation_angle / 2)
    scale_noise = (2 * np.random.random() - 1) * noise_intensity
    base_poses = action_sequence.poses
    if rotate:
        base_poses = rotate_poses(base_poses, angle)
    if noise:
        base_poses = gaussian_noise_poses(base_poses, noise_intensity)
        base_poses = scale_noise_poses(base_poses, scale_noise)

    new_metadata = dict(action_sequence.metadata)
    new_metadata['group'] = group
    new_action = PoseSequence(metadata=new_metadata)
    new_action.poses = base_poses
    new_action.labels = action_sequence.labels

    return new_action


def generate_n_action_sequence(action_sequence, n, rotate=True, noise=True, rotation_angle=50, noise_intensity=0.1):
    new_action_sequences = []
    for i in range(n):
        group_name = 'Copy ' + str(i)
        new_action_sequence = data_augmentation_by_action_sequence(action_sequence, group=group_name)
        new_action_sequences.append(new_action_sequence)
    return new_action_sequences



def merge_metadata(actions):
    merged_metadata = {}
    for action in actions:
        for key in action.metadata.keys():
            merged_metadata[key] = action.metadata[key]
    return merged_metadata


def merge_actions(actions):
    metadata = merge_metadata(actions)
    merged_action = PoseSequence(metadata=metadata)
    has_labels = True
    for action in actions:
        merged_action.poses = merged_action.poses + action.poses
        if action.labels:
            if not has_labels:
                raise ValueError
            merged_action.labels = merged_action.labels + action.labels
        else:
            has_labels = False

    if has_labels & (len(merged_action.poses) != len(merged_action.labels)):
        raise ValueError

    return merged_action


def list_values_by_field_name(actions, field_name):
    values = []
    for action in actions:
        if action.metadata[field_name] not in values:
            values.append(action.metadata[field_name])
    return values


def filter_by_metadata(actions, field_name, value):
    if isinstance(value, list):
        return_list = []
        for s in value:
            return_list = return_list + list(filter(lambda action: action.metadata[field_name] == s, actions))
        return return_list
    else:
        return list(filter(lambda action: action.metadata[field_name] == value, actions))