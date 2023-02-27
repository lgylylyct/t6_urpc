import os
import pickle
import numpy as np
import torchio as tio

from Core.config_seg import config


def get_validset(fold):
    with open(os.path.join(config.DATASET.ROOT, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)[fold]
    valids = splits['val']

    subjects = []
    for name in valids:
        data = np.load(os.path.join(config.DATASET.ROOT, name+'.npz'))['data']
        subject = tio.Subject(
            data = tio.ScalarImage(tensor=data[:-1]),
            label = tio.LabelMap(tensor=data[-1:]),
            name = name
        )
        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects)
    return dataset

def get_validset_one_fold(dataset_root, dataset_root_2):
    with open(os.path.join(dataset_root, 'spilt.pkl'), 'rb') as f:
        splits = pickle.load(f)
    valids = splits

    subjects = []
    for name in valids:
        data = np.load(os.path.join(dataset_root_2, name+'.npz'))['data']
        subject = tio.Subject(
            data = tio.ScalarImage(tensor=data[:-1]),
            label = tio.LabelMap(tensor=data[-1:]),
            name = name
        )
        subjects.append(subject)
    dataset = tio.SubjectsDataset(subjects)
    return dataset

