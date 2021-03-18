from torch.utils.data import Dataset
import torch
import random
import numpy as np


class PoseSequenceDataset(Dataset):
    def __init__(self, action_sequences, shuffle_sequence=False, transform=None):
        self.transform = transform
        self.action_sequences = action_sequences
        self.shuffle_sequence = shuffle_sequence

    def __len__(self):
        return len(self.action_sequences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        action_sequence = self.action_sequences[idx]
        if self.shuffle_sequence:
            raise NotImplemented
            # random.shuffle(action_sequence.actions)

        if self.transform:
            action_sequence = self.transform(action_sequence)
        return action_sequence


