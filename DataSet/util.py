from sklearn.metrics import confusion_matrix, classification_report
import torch
import torch.nn.functional as F
from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from PIL import Image
import os
import os.path
import SimpleITK as sitk

#import pandas as pd
import glob


def get_classification_report(cm):
    precision = []
    recall = []
    for i in range(cm.shape[0]):
        precision.append(cm[i, i] / cm[:, i].sum())
        recall.append(cm[i, i] / cm[i, :].sum())
    return precision, recall


def get_confusion_matrix(true, pred, labels):
    true = true.flatten()
    pred = pred.flatten()

    return confusion_matrix(true, pred, labels)


def get_miou(cm):
    return np.diag(cm) / (cm.sum(1) + cm.sum(0) - np.diag(cm) + 1e-6)


def label_smooth(mask, semantic_map, labels, alpha, radius):
    """
    :param
        mask: mask of shape [H, W]
        semantic_map: one-hot mask of shape [C, H, W]
    """
    pad_mask = np.pad(mask, ((radius, radius), (radius, radius)), 'edge')

    for i in range(radius, len(pad_mask[0]) - radius):
        for j in range(radius, len(pad_mask[0]) - radius):
            if not (pad_mask[i][j] == pad_mask[i - radius: i + radius + 2, j - radius: j + radius + 2]).all():
                pos = 1. - alpha
                neg = alpha / len(labels)
                idx = semantic_map[:, i - radius, j - radius].argmax()
                semantic_map[:, i - radius, j - radius].fill(neg)
                semantic_map[:, i - radius, j - radius][idx] = pos + neg

    return semantic_map


def mask_to_semantic(mask, labels=[0, 1, 2], smooth=False, alpha=0.2, radius=8):
    # 标签图转语义图  labels代表所有的标签 [100, 200, 300, 400, 500, 600, 700, 800]
    # return shape [C, H, W]

    semantic_map = []

    if smooth == "edge":
        semantic_map = []
        filter = torch.randint(1, 10, (2 * radius + 1, 2 * radius + 1), requires_grad=False, device='cpu')
        filter[radius, radius] = - filter.sum() + filter[radius, radius]
        # print(np.unique(filter), filter)
        # print(mask.shape, filter.device)
        pad = np.pad(mask.astype(np.float32), ((radius, radius), (radius, radius)), 'edge')
        ipt = torch.from_numpy(pad).unsqueeze(dim=0).unsqueeze(dim=0)
        kernel = filter.unsqueeze(dim=0).unsqueeze(dim=0).float()
        # print(ipt.shape, kernel.shape)
        # print(ipt.shape, kernel.shape)
        # temp = F.conv2d(ipt, kernel, padding=radius, stride=1).numpy()

        smooth_map = (F.conv2d(ipt, kernel, stride=1).numpy() != 0).astype(np.float16).squeeze() * 0.5
        # print(np.unique(smooth_map), smooth_map.shape, smooth_map[255, 8])
        pos, neg = 1. - alpha + alpha / len(labels), alpha / len(labels)
        for label in labels:
            equality = (mask == label).astype(np.float16)
            # print(np.unique(equality))
            equality += smooth_map
            # print(np.unique(smooth_map))
            equality[equality == 1.5] = pos
            equality[equality == 1.0] = 1.0
            equality[equality == 0.5] = neg
            # print(np.unique(mask), np.unique(equality))
            semantic_map.append(equality)
        semantic_map = np.array(semantic_map).astype(np.float16)
        return semantic_map
    else:
        for label in labels:
            equality = np.equal(mask, label)
            semantic_map.append(equality)
        semantic_map = np.array(semantic_map).astype(np.float16)

    return semantic_map


def semantic_to_mask(mask, labels):
    # 语义图转标签图  labels代表所有的标签 [100, 200, 300, 400, 500, 600, 700, 800]
    x = np.argmax(mask, axis=1)
    label_codes = np.array(labels)
    x = np.uint8(label_codes[x.astype(np.uint8)])
    return x


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-12] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!",
                          dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)


def generate_dataset_pkl(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                         labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!",
                         dataset_description: str = "",
                         dataset_reference="", dataset_release='0.0'):
    pass


def gain_all_file_name(root_dir: str, image_type: str, suffix: str):
    all_files_path = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
        if len(files) > 0:
            each_foder_files = [os.path.join(root, x) for x in files]
            all_files_path.extend(each_foder_files)
    return all_files_path


def gain_set_file_name(root_dir: str, set_type='T1.nii.gz'):
    set_files_path = []
    for root, _, _ in os.walk(root_dir, topdown=False):
        each_foder_files = os.path.join(root, set_type)
        set_files_path.append(each_foder_files)
    return set_files_path[:-1]

import torch.utils.data as data

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def get_filelist(dir, Filelist):
    if os.path.isdir(dir):

        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            Filelist.append(newDir)

    return Filelist

def default_loader(path):
    return Image.open(path).convert('RGB')
## 读取当前文件夹下的所有文件
## 获取文件路径


def nii2npz(input_list, outputroot):
    """
    suf : T1.nii.gz/npz
    pid: wuzhou1_80977
    """
    for e in input_list:
        Refimg = sitk.ReadImage(e)
        RefimgArray = sitk.GetArrayFromImage(Refimg)
        suf = e.split('\\')[-1]
        pid = e.split('\\')[-2]

        suf = suf.replace('nii.gz', 'npz')
        outPutfile = os.path.join(outputroot, pid, suf)
        outputdir = os.path.join(outputroot, pid)

        if ~os.path.exists(outputdir):
            os.mkdir(outputdir)

        np.savez(outPutfile, vol_data=RefimgArray)


def nii2npz2pkl(input_list, outputdir):
    pass

if __name__ == "__main__":


    #pred = np.array([0, 1, 0, 0, 0, 1, 1, 1, 1, 1])
    #label = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    #cm = get_confusion_matrix(label, pred, labels=[0, 1])
    # print(get_classification_report(cm))
    obj_path = r'E:\NPCICdataset\Patient_Image\test_nii'
    outputdir = r'E:\NPCICdataset\Patient_Image\test_npz'

    set_path = gain_set_file_name(obj_path, set_type='T1.nii.gz')
    nii2npz(set_path, outputdir)
    print(set_path)



