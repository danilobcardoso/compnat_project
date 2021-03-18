import numpy as np
import os
import pdb
from loader.base import BaseLoader


def sort_files(string):
    return int(string.split('_')[1])


def get_values(line):
    try:
        values = line.split(',')
    except AttributeError:
        return [None]
    except Exception as e:
        print(e)
        pdb.set_trace()
    values = [float(value) for value in values]
    return values


def load_csv(csv_path):
    with open(csv_path, 'r') as f:
        lines = [line.rstrip() for line in f]
    lines = [line for line in lines if line != '']
    return lines


def split_data(data):
    inertial = [sample[:, -6:] for sample in data]
    camera = [sample[:, 75:150].reshape(-1, 25, 3) for sample in data]
    test1 = [sample[:, :75] for sample in data]
    return camera, inertial, test1


def integrate_data(data):
    for sample_idx, sample in enumerate(data):
        n_frames = len(sample)
        for frame_idx in range(1, n_frames):
            data[sample_idx][frame_idx, :, :] = sample[frame_idx - 1, :, :] + sample[frame_idx, :, :]
    return data


def get_min_seq_size(data):
    min_seq = float('inf')
    for sample_idx, sample in enumerate(data):
        if sample.shape[0] < min_seq:
            min_seq = sample.shape[0]
    return min_seq


def load_data(csv_path):
    lines = load_csv(csv_path)
    data = []
    n_line = 0
    while len(lines) > 1:
        header = get_values(lines[0])
        n_lines = int(header[0])
        n_frames = int(header[1])
        lines = lines[1:]
        frame_data = np.empty((n_frames, n_lines))
        for i in range(n_lines):
            n_line = n_line + 1
            line = lines[i].split(',')
            line_values = [float(value) for value in line]
            frame_data[:, i] = line_values
        lines = lines[n_lines:]
        data.append(frame_data)
    return data


class LGLoader(BaseLoader):

    def __init__(self, data_path='./datasets/lg_activitydetection/dataset_csv/body_inertial/'):
        super().__init__(data_path)


    def load(self):
        ## LOAD DATA
        dataset_files = os.listdir(self.data_path)

        data_files = sorted([f for f in dataset_files if "labels" not in f], key=sort_files)
        data_files = [self.data_path + data_file for data_file in data_files]

        labels_files = sorted([f for f in dataset_files if "labels" in f], key=sort_files)
        labels_files = [self.data_path + label_file for label_file in labels_files]

        data = []
        labels = []

        data_files = [data_files[0]]
        labels_files = [labels_files[0]]

        for idx, data_file in enumerate(data_files):
            ## DATA LOADING
            file_data = load_data(data_file)
            camera, inertial, test1 = split_data(file_data)
            camera = integrate_data(camera)
            file_labels = load_data(labels_files[idx])
            camera, file_labels = zip(
                *((sample, label) for sample, label in zip(camera, file_labels) if len(sample) > 1))
            camera, file_labels = zip(
                *((sample, label) for sample, label in zip(camera, file_labels) if sample.sum != 0))
            camera = list(camera)
            file_labels = list(file_labels)
            data.extend(camera)
            labels.extend(file_labels)
        return data, labels



    def skeleton_model(self):
        pass