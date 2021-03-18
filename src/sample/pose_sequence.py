import numpy as np


class PoseSequence:
    def __init__(self, metadata=None):
        self.poses = []
        self.labels = []
        self.metadata = metadata

    def add_pose(self, pose, label=None):
        self.poses.append(pose)
        self.labels.append(label)

    def calculate_velocity_xyz(self):
        last_pose = None
        for pose in self.poses:
            if not last_pose:
                pose.add_data('velocity_xyz', np.zeros(pose.data['position_xyz'].shape))
                last_pose = pose
            else:
                velocity = pose.data['position_xyz'] - last_pose.data['position_xyz']
                pose.add_data('velocity_xyz', velocity)
                last_pose = pose

    def to_numpy(self, data_types):
        action_data = []
        for pose in self.poses:
            pose_data = None
            for data_type in data_types:
                if pose_data is None:
                    pose_data = pose.data[data_type]
                else:
                    pose_data = np.concatenate((pose_data, pose.data[data_type]), axis=1)
            action_data.append(pose_data)
        return np.array(action_data)

    def __str__(self):
       return '{}  \t {} poses'.format(self.metadata, len(self.poses))

    def __repr__(self):
        return self.__str__()

