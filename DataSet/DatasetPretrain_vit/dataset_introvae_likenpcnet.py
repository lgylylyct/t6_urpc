import math
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from DataSet.data_transform import HorizontalFlip, VerticalFlip, Rotate, ToTensor, Resize, Resize_2, ToTensor_2d
import os
import nibabel as nib
import torchio as tio
import SimpleITK as sitk
from torch.utils.data import DataLoader
from Untils.debug import checkImageMatrix
from Untils.show_dataloader import show_pic,show_pic_torchio,test_dataloader,show_pic_torchio_pic
from Core.config import config
from matplotlib import pyplot as plt


class NPCVAEDataset(Dataset):
    def __init__(self, img_dir, mask_dir, mode="test", seq_mode="T1C.nii.gz", num=16, smooth = False):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.mode = mode
        self.labels = [0, 1, 2]
        self.num = num
        self.seq_mode = seq_mode
        self.image = list(sorted(os.listdir(img_dir)))
        self.mask = list(sorted(os.listdir(mask_dir)))
        self.hf = HorizontalFlip(p=1)
        self.vf = VerticalFlip(p=1)
        self.rt = Rotate(degrees=(90, 180, 270))                                                #这里我还是选择对其进行旋转的对应增强,关于解决这个问题
        self.rs = Resize(scales=[(320, 320), (192, 192), (384, 384), (128, 128)], p=0.5)         #随机进行缩放
        self.rs2 = Resize_2(scales=(256, 256))
        self.tt = ToTensor()
        self.tt2 = ToTensor_2d()

    def __len__(self):
        return len(os.listdir(self.img_dir))*self.num

    def __getitem__(self, item):
        if torch.is_tensor(item):
            item = item.tolist()

        image_3d = nib.load(os.path.join(self.img_dir, self.image[math.floor(item / self.num)], self.seq_mode))
        mask_3d = nib.load(os.path.join(self.img_dir, self.image[math.floor(item / self.num)], self.seq_mode))

        if item % self.num == 0:
            image_3d = nib.load(os.path.join(self.img_dir, self.image[math.floor(item / self.num)], self.seq_mode))
            mask_3d = nib.load(os.path.join(self.img_dir, self.image[math.floor(item / self.num)], self.seq_mode))

        image = image_3d.dataobj[:, :, item % self.num]
        mask = mask_3d.dataobj[:, :, item % self.num]

        if self.mode == "train":
            seed = np.random.randint(0, 4, 1)
            if seed == 0:
                pass
            elif seed == 1:
                image, mask = self.hf(image, mask)
            elif seed == 2:
                image, mask = self.vf(image, mask)
            elif seed == 3:
                image, mask = self.rt(image, mask)

            image, mask = self.rs2(image,mask)
            image, mask = self.tt2(image, mask)

        elif self.mode == 'val':

            #image, mask = self.tt(image, mask, labels=self.labels)
            image, mask = self.tt(image, mask)

        elif self.mode == 'test':

            #return image, mask, self.images[item]
            return image, mask, self.images[item]

        else:
            print("invalid transform mode")

        return image, mask


def get_dataloader(img_dir, mask_dir, batch_size, num_workers, mode="train", seq_mode="T1C.nii.gz", num=16, smooth=False):

    if mode == "train":
        train_dataset = NPCVAEDataset(img_dir=img_dir, mask_dir=mask_dir, mode="train", seq_mode=seq_mode, num=num, smooth=False)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        return train_dataloader
    elif mode == "test":
        test_dataset = NPCVAEDataset(img_dir, mask_dir, mode='test', smooth=False)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return test_dataloader
    else:
        val_dataset = NPCVAEDataset(img_dir, mask_dir, mode='val', smooth=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        return val_dataloader

if __name__ == "__main__":
    # mod = config.DATASET.MOD
    #
    # train_image_dir = config.DATASET.ROOT
    # train_label_dir = config.DATASET.ROOT
    # val_image_dir = "../data/Multi_V1/val/image"
    # val_label_dir = "../data/Multi_V1/val/label"
    #
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # train_dataset = NPCVAEDataset(img_dir=train_image_dir, mask_dir=train_label_dir, mode="train", seq_mode=config.DATASET.MOD, num=16,
    #                               smooth=False)

    train_loader = get_dataloader(img_dir = config.DATASET.ROOT, mask_dir = config.DATASET.ROOT, batch_size=config.TRAIN.BATCH_SIZE,num_workers=config.NUM_WORKERS, mode="train", seq_mode= config.DATASET.MOD, num=config.num, smooth=False)

    # NPCVAEDataset(img_dir=img_dir, mask_dir=mask_dir, mode="train", seq_mode=seq_mode, num=num, smooth=False)

