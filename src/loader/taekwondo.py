import os
import numpy as np

from loader.base import BaseLoader
from sample.pose_sequence import PoseSequence
from sample.pose import Pose


class TaekwondoLoader(BaseLoader):
    def __init__(self, data_path='./datasets/taekwondo'):
        super().__init__(data_path)
        self.prefix = 'Athlete'
        self.data_file = 'Filtered Joints.txt'

    def load(self):
        actions = []
        labels = set()
        skeleton_model = self.skeleton_model()
        for folder in os.listdir(self.data_path):
            if self.prefix in folder:
                athlete_id = folder.replace(self.prefix, '')
                athlete_path = '{}/{}'.format(self.data_path, folder)
                for label_folder in os.listdir(athlete_path):
                    label = label_folder
                    labels.add(label)
                    data_path = '{}/{}/{}'.format(athlete_path, label_folder, self.data_file)
                    data = self.read_activity_file(data_path)

                    action = PoseSequence(metadata={'source': athlete_id})
                    for frame_idx in range(data.shape[0]):
                        skeleton = Pose(skeleton_model)
                        skeleton.add_data('position_xyz', data[frame_idx])
                        action.add_pose(skeleton, label=label)

                    actions.append(action)

        label_idx = {key: idx for (idx, key) in enumerate(labels)}
        return actions, labels, label_idx, skeleton_model


    def read_activity_file(self, data_path):
        # print(data_path)
        invalid_nodes = [11, 12, 17, 18]
        with open(data_path, "r") as data_file:
            param = data_file.readline().split(',')
            lines = int(param[0].rstrip()) - 2  # Duas primeiras linhas são parametros
            frames = int(param[1].rstrip())
            dunno = data_file.readline()
            data = [[[0, 0, 0] for j in range(20 - len(invalid_nodes))] for i in range(frames)]

            jumps = 0
            for i in range(lines):
                line = data_file.readline().split(',')
                if int(i / 3) in invalid_nodes:
                    if int(i % 3) == 0:
                        jumps = jumps + 1;
                    continue;

                node = int(i / 3) - jumps
                axis = int(i % 3)

                # print("node: {} / axis: {}".format(node, axis))
                for frame in range(frames):
                    data[frame][node][axis] = float(line[frame])
            return np.array(data)

    def skeleton_model(self):
        return {
            0: [3, 7, 15, 6],
            1: [3, 8, 15, 6],
            2: [15, 6],
            3: [0, 1, 4, 5, 15, 6],
            4: [3, 5, 12, 6],
            5: [3, 4, 11, 6],
            6: [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            7: [0, 9, 6],
            8: [1, 10, 6],
            9: [7, 6],
            10: [8, 6],
            11: [5, 13, 6],
            12: [4, 14, 6],
            13: [11, 6],
            14: [12, 6],
            15: [0, 1, 2, 3, 6],
            'num_nodes': 16,
            'name': {
                2: 'head',
                15: 'neck',
                3: 'chest',
                6: 'aggregator',
                0: 'left_shoulder',
                7: 'left_elbow',
                9: 'left_hand',
                1: 'right_shoulder',
                8: 'right_elbow',
                10: 'right_hand',
                4: 'right_rib',
                12: 'right_knee',
                14: 'right_foot',
                5: 'left_rib',
                11: 'left_knee',
                13: 'left_foot'
            },
            'colors': {
                2: '#666666',
                15: '#FF00FF',
                3: '#FF00FF',
                6: '#666666',
                0: '#FFFF00',
                7: '#FF9802',
                9: '#FF0000',
                1: '#02FF00',
                8: '#02FFFF',
                10: '#0600FF',
                4: '#02FF00',
                12: '#02FFFF',
                14: '#0600FF',
                5: '#FFFF00',
                11: '#FF9802',
                13: '#FF0000'
            }
        }
