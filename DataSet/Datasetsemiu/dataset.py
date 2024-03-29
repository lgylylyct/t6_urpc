import torch
import random
import numpy as np
import os
from torch.utils.data import Dataset
import h5py
import torchio as tio
import nibabel as nib
from scipy.ndimage.interpolation import zoom
from torchvision import transforms
import itertools
from scipy import ndimage
from torch.utils.data.sampler import Sampler
import augmentations
from augmentations.ctaugment import OPS
import matplotlib.pyplot as plt


class BaseDataSets(Dataset):
    def __init__(
        self,
        base_img_dir=None,
        base_mask_dir=None,
        mod="T1C",
        split="train",
        label_mod_type="T1C_mask",
        train_num=None,
        transform=None,
        ops_weak=None,
        ops_strong=None,
    ):
        self.base_img_dir = base_img_dir
        self.base_mask_dir = base_mask_dir
        self.mod = mod
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.ops_weak = ops_weak
        self.ops_strong = ops_strong
        self.label_mod_type = label_mod_type
        self.train_num = train_num

        assert bool(ops_weak) == bool(
            ops_strong
        ), "For using CTAugment learned policies, provide both weak and strong batch augmentation policy"

        self.sample_list = [[], []]
        for patient_id in os.listdir(self.base_img_dir):
            self.sample_list[0].append(
                os.path.join(self.base_img_dir, patient_id, self.mod + ".npy")
            )
            self.sample_list[1].append(
                os.path.join(
                    self.base_mask_dir, patient_id, self.mod, self.label_mod_type + ".npy"
                )
            )

        # if self.split == "train":
        #     with open(self._base_dir + "/train.txt", "r") as f1:
        #         self.sample_list = f1.readlines()
        #     self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        # elif self.split == "val":
        #     with open(self._base_dir + "/val.txt", "r") as f:
        #         self.sample_list = f.readlines()
        #     self.sample_list = [item.replace("\n", "") for item in self.sample_list]

        if self.train_num is None:
            self.train_num = int(len(self.sample_list[0]) * 0.8)

        if self.split == "train":
            self.sample_list[0] = self.sample_list[0][: self.train_num]
            self.sample_list[1] = self.sample_list[1][: self.train_num]
        elif self.split == "val":
            self.sample_list[0] = self.sample_list[0][self.train_num :]
            self.sample_list[1] = self.sample_list[1][self.train_num :]

        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list[0])

    def __getitem__(self, idx):
        image = np.load(self.sample_list[0][idx])
        label = np.load(self.sample_list[1][idx])
        image_3d = torch.from_numpy(image)
        label_3d = torch.from_numpy(label)

        # select slice
        if self.split == "train":
            temp = 0
            while True:
                z = np.random.randint(image_3d.shape[0])
                if label_3d[z, :, :].sum() != 0:
                    break
                temp += 1
                if temp > 20:
                    break
            image = image_3d[z, :, :]
            label = label_3d[z, :, :]
        else:
            image = image_3d
            label = label_3d

            temp = label.sum(axis=(1, 2))
            d1 = -1
            d2 = -1
            for dd in range(len(temp)):
                if temp[dd] != 0:
                    if d1 == -1:
                        d1 = dd
                    d2 = dd
            
            image = image[d1:d2+1]
            label = label[d1:d2+1]

        sample = {"image": image, "label": label, "idx": idx}

        if self.split == "train":
            if None not in (self.ops_weak, self.ops_strong):
                sample = self.transform(sample, self.ops_weak, self.ops_strong)
            else:
                sample = self.transform(sample)

        return sample


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.rot90(label, k)
        label = np.flip(label, axis=axis).copy()
        return image, label
    else:
        return image


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=3, reshape=False, mode="nearest")
    label = ndimage.rotate(label, angle, order=3, reshape=False, mode="nearest")
    return image, label


def color_jitter(image):
    if not torch.is_tensor(image):
        np_to_tensor = transforms.ToTensor()
        image = np_to_tensor(image)

    # s is the strength of color distortion.
    s = 1.0
    jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
    return jitter(image)


class CTATransform(object):
    def __init__(self, output_size, cta):
        self.output_size = output_size
        self.cta = cta

    def __call__(self, sample, ops_weak, ops_strong):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        to_tensor = transforms.ToTensor()

        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        # apply augmentations
        image_weak = augmentations.cta_apply(transforms.ToPILImage()(image), ops_weak)
        image_strong = augmentations.cta_apply(image_weak, ops_strong)
        label_aug = augmentations.cta_apply(transforms.ToPILImage()(label), ops_weak)
        label_aug = to_tensor(label_aug).squeeze(0)
        label_aug = torch.round(255 * label_aug).int()

        sample = {
            "image_weak": to_tensor(image_weak),
            "image_strong": to_tensor(image_strong),
            "label_aug": label_aug,
        }
        return sample

    def cta_apply(self, pil_img, ops):
        if ops is None:
            return pil_img
        for op, args in ops:
            pil_img = OPS[op].f(pil_img, *args)
        return pil_img

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        # ind = random.randrange(0, img.shape[0])
        # image = img[ind, ...]
        # label = lab[ind, ...]
        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y = image.shape
        image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))
        sample = {"image": image, "label": label}
        # plt.imshow(image, cmap='gray')
        return sample


class WeakStrongAugment(object):
    """returns weakly and strongly augmented images

    Args:
        object (tuple): output size of network
    """

    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample["image"], sample["label"]
        image = self.resize(image)
        label = self.resize(label)
        # weak augmentation is rotation / flip
        image_weak, label = random_rot_flip(image, label)
        # strong augmentation is color jitter
        image_strong = color_jitter(image_weak).type("torch.FloatTensor")
        # fix dimensions
        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        image_weak = torch.from_numpy(image_weak.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.uint8))

        sample = {
            "image": image,
            "image_weak": image_weak,
            "image_strong": image_strong,
            "label_aug": label,
        }
        return sample

    def resize(self, image):
        x, y = image.shape
        return zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)


class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices

    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """

    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch) in zip(
                grouper(primary_iter, self.primary_batch_size),
                grouper(secondary_iter, self.secondary_batch_size),
            )
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size


def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)

    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)
