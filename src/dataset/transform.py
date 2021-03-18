import torch
import random
import numpy as np


class SplitFramesAndLabels(object):
    def __init__(self, label_idx, data_types, max_lenght=None):
        self.label_idx = label_idx
        self.max_lenght = max_lenght
        self.data_types = data_types

    def __call__(self, action_sequence):

        sequence_frames = action_sequence.to_numpy(self.data_types)
        frames = sequence_frames.shape[0]
        nodes = sequence_frames.shape[1]

        sequence_labels = np.ones((frames, nodes))
        for frame_idx in range(frames):
            sequence_labels[frame_idx, :] = np.ones((1, nodes)) * self.label_idx[action_sequence.labels[frame_idx]]

        if self.max_lenght:
            sequence_frames = sequence_frames[:self.max_lenght]
            sequence_labels = sequence_labels[:self.max_lenght]

        return {'frames': sequence_frames, 'labels': sequence_labels}


class UnfoldFeatures(object):

    def __init__(self, skeleton_model):
        self.skeleton_model = skeleton_model

    def __call__(self, sample):
        frames = sample['frames']
        num_frames = frames.shape[0]
        num_nodes = frames.shape[1]
        num_dim = frames.shape[2]
        res = np.zeros((num_frames, num_nodes, num_nodes * num_dim))
        for frame_idx in range(num_frames):
            for node_idx in range(num_nodes):
                res[frame_idx, node_idx, num_dim * node_idx:num_dim * (node_idx + 1)] = frames[frame_idx, node_idx]

        return {'frames': res, 'labels': sample['labels'] }


class ToSTGcn(object):

    def __call__(self, sample):
        frames = sample['frames']
        return {'frames': frames.transpose((2, 0, 1)), 'labels': sample['labels']}
