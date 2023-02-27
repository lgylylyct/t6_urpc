import os
import pickle
import numpy as np
from Core.config_seg import config
from collections import OrderedDict
from Untils.utils import get_patch_size
from batchgenerators.dataloading.data_loader import SlimDataLoaderBase

class DataLoader2D(SlimDataLoaderBase):
    def __init__(self, data, batch_size, patch_size):
        super().__init__(data, batch_size, None)
        self.oversample_foreground_percent = 1 / 3
        # larger patch size is required for proper data augmentation
        self.patch_size = get_patch_size(patch_size, (-np.pi, np.pi), (0, 0), (0, 0), (0.7, 1.4))

    def generate_train_batch(self):
        # random select data
        sels = np.random.choice(list(self._data.keys()), self.batch_size, True)
        # read data, form slice
        images, labels = [], []
        for i, name in enumerate(sels):
            data = np.load(self._data[name]['path'])['data']
            # randomly select slice
            sel_idx = np.random.choice(data.shape[1])
            data = data[:, sel_idx]
            shape = np.array(data.shape[1:])
            pad_length = self.patch_size - shape
            pad_left = pad_length // 2
            pad_right = pad_length - pad_length // 2
            data = np.pad(data, ((0, 0), (pad_left[0], pad_right[0]), (pad_left[1], pad_right[1])))
            images.append(data[:-1])
            labels.append(data[-1:])
        image = np.stack(images)
        label = np.stack(labels)
        return {'data': image, 'label': label}


def get_trainloader(fold):
    # list data path and properties
    with open(os.path.join(config.DATASET.ROOT, 'splits.pkl'), 'rb') as f:
        splits = pickle.load(f)[fold]
    trains = splits['train']
    dataset = OrderedDict()
    for name in trains:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(config.DATASET.ROOT, name + '.npz')
        with open(os.path.join(config.DATASET.ROOT, name + '.pkl'), 'rb') as f:
            dataset[name]['locs'] = pickle.load(f)

    return DataLoader2D(dataset, config.TRAIN.BATCH_SIZE, config.TRAIN.PATCH_SIZE)


def get_trainloader_one_fold(dataset_root, dataset_root_2):
    # list data path and properties
    with open(os.path.join(dataset_root, 'spilt.pkl'), 'rb') as f:
        splits = pickle.load(f)
    trains = splits
    dataset = OrderedDict()
    for name in trains:
        dataset[name] = OrderedDict()
        dataset[name]['path'] = os.path.join(dataset_root_2, name + '.npz')
        with open(os.path.join(dataset_root_2, name + '.pkl'), 'rb') as f:
            dataset[name]['locs'] = pickle.load(f)

    return DataLoader2D(dataset, config.TRAIN.BATCH_SIZE, config.TRAIN.PATCH_SIZE)


##name 是每个病人对应的名字
##所以其实这个dataset 是一个字典，然后字典下面每个dataset[name][path]存放npz之间的路径  dataset[name][locs]是对应的文件之间的信息（spacing之类的信息）

if __name__ == '__main__':
    from DataSet.DatasetSimSiam.augmenter import get_train_generator

    dataset_root = r'E:\NPCICdataset\Patient_Image\seg_test\nnUNet_raw_data\Task001_Npc'
    dataset_root_2 = r'E:\NPCICdataset\Patient_Image\seg_test\nnUNet_process\Task001_Npc'  ##这里存放的是整理之后的数据的位置 以及对应的结果堆叠之间的结果
    test_dataloader_simsiam_2d = get_trainloader_one_fold(dataset_root, dataset_root_2)
    data_dict = next(test_dataloader_simsiam_2d)
    print(data_dict['data'])
    test_generator = get_train_generator(test_dataloader_simsiam_2d, scales=(1., 0.5, 0.25, 0.125))
    #num_iter = config.TRAIN.NUM_BATCHES
    data_dict = next(test_generator)
    print(data_dict['data'])
    #print(data_dict['label'])
    # data = data_dict['data']
    # label = data_dict['label']

