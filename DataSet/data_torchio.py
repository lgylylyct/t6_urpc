from glob import glob
from os.path import dirname, join, basename, isfile
import sys

sys.path.append('./')
import csv
import torch
import numpy as np
from PIL import Image
from torch import nn
import torch.nn.functional as F
import random
import torchio as tio
import nibabel as nib
import numpy as np
import torchio as tio
from torchio import AFFINE, DATA
import torchio
from torchio import ScalarImage, LabelMap, Subject, SubjectsDataset, Queue
from torchio.data import UniformSampler
from torch.utils.data import DataLoader
from torchio.transforms import (
    RandomFlip,
    RandomAffine,
    RandomElasticDeformation,
    RandomNoise,
    RandomMotion,
    RandomBiasField,
    RescaleIntensity,
    Resample,
    ToCanonical,
    ZNormalization,
    CropOrPad,
    HistogramStandardization,
    OneOf,
    Compose,
)
from pathlib import Path

test_path = r"E:\NPCICdataset\Patient_Image\test_nii"

source_train_0 = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\0'
source_train_1 = r'C:\Users\Admin\SIAT\model\Pytorch-Medical-Classification-main\1'

class MedData_train(torch.utils.data.Dataset):
    def __init__(self, images_dir_0, images_dir_1):
        self.subjects = []
        images_dir_0 = Path(images_dir_0)
        self.image_paths_0 = sorted(images_dir_0.glob(hp.fold_arch))
        images_dir_1 = Path(images_dir_1)
        self.image_paths_1 = sorted(images_dir_1.glob(hp.fold_arch))

        for (image_path) in zip(self.image_paths_0):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label = 0   ###需要给对应的标签进行数据的操作
            )
            self.subjects.append(subject)

        for (image_path) in zip(self.image_paths_1):
            subject = tio.Subject(
                source=tio.ScalarImage(image_path),
                label= 1
            )
            self.subjects.append(subject)
        self.transforms = self.transform()
        self.training_set = tio.SubjectsDataset(self.subjects, transform=self.transforms)
        # one_subject = self.training_set[0]
        # one_subject.plot()

def get_load():
    # dataset_ = PET_dataset(dir)
    # data_loader = DataLoader(dataset=dataset_,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers)
    train_dataset = MedData_train(source_train_0, source_train_1)
    train_loader = DataLoader(train_dataset.training_set,
                              batch_size=1,
                              shuffle=True,
                              pin_memory=True,
                              drop_last=True)
    return train_loader


train_loader = get_load()

for i, batch in enumerate(train_loader):
    x = batch['source']['data']
    # imag = x.reshape([64,64,64,1]) #将之转化为数组查看
    # imag = imag.numpy()
    y = batch['label']

    x = x.type(torch.FloatTensor).cuda()
    y = y.type(torch.LongTensor).cuda()
    print(x.shape)
    print(y)

    # img_t1 = nib.Nifti1Image(imag, np.eye(4))
    # nib.save(img_t1, 'output.nii.gz')#将之保存为nii查看
    # print(imag.shape)