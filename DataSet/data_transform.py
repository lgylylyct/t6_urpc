import torchvision.transforms.functional as F
import random
import torch
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from torchvision import transforms
import PIL
#import cv2
from DataSet.util import mask_to_semantic


class HorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:
            image = np.flip(image, axis=1)
            mask = np.flip(mask, axis=1)

        return image, mask


class VerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:
            image = np.flip(image, axis=0)
            mask = np.flip(mask, axis=0)

        return image, mask


class Rotate(object):

    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, image, mask):

        angle = random.choice(self.degrees)

        if angle == 90:
            image = np.rot90(image, 1, (1, 0))
            mask = np.rot90(mask, 1, (1, 0))
        elif angle == 180:
            image = np.rot90(image, 2, (1, 0))
            mask = np.rot90(mask, 2, (1, 0))
        elif angle == 270:
            image = np.rot90(image, 3, (1, 0))
            mask = np.rot90(mask, 3, (1, 0))

        return image, mask


class Resize(object):
    """进行随机重采样"""
    def __init__(self, p=0.5, scales=None):
        if scales is None:
            scales = [(320, 320), (192, 192), (384, 384), (128, 128)]
        self.scales = scales
        self.p = p

    def __call__(self, image, mask):

        if random.random() < self.p:
            scale = random.choice(self.scales)
            image = image.resize(scale, resample=PIL.Image.BILINEAR)
            mask = mask.resize(scale, resample=PIL.Image.BILINEAR)

        return image, mask

class Resize_2(object):
    def __init__(self, scales=None):
        if scales is None:
            scales = (256,256)
        self.scales = scales

    def __call__(self, image, mask):
        image = cv2.resize(image, dsize=self.scales, interpolation=cv2.INTER_CUBIC)
        mask = cv2.resize(image, dsize=self.scales, interpolation=cv2.INTER_CUBIC)
        ## 但是请注意这个出来的情况后肯定是反过来的
        # image = image.resize(self.scales, resample=PIL.Image.BILINEAR)
        # mask = mask.resize(self.scales, resample=PIL.Image.BILINEAR)

        return image, mask


class ToTensor(object):

    def __call__(self, image, mask, labels=None, mode="train", smooth=False):
        # image transform

        if labels is None:
            labels = [0, 1, 2]
        for i in range(image.shape[2]):
            image[:, :, i] = (image[:, :, i] - np.min(image[:, :, i])) / (np.max(image[:, :, i]) - np.min(image[:, :, i]))

        # print(image.shape, image.dtype)
        image = torch.from_numpy(image.transpose((2, 0, 1)).copy())

        # mask transform to semantic
        mask = torch.from_numpy(mask_to_semantic(mask, labels, smooth=smooth))
        return image, mask

class ToTensor_2d(object):

    def __call__(self, image, mask, mode="train", smooth=False):
        # image transform

        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # print(image.shape, image.dtype)
        #image = torch.from_numpy(image.transpose((2, 0, 1)).copy())
        #mask is same with the image
        mask = image
        return image, mask