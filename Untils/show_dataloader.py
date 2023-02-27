from torch.optim.lr_scheduler import CosineAnnealingLR
from torchinfo import summary
# import timm
# import torchvision
# import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm
# import Ranger

def show_pic(dataloader,classes):        # 展示dataloader里的6张图片
    examples = enumerate(dataloader)     # 组合成一个索引序列
    batch_idx, (example_data, example_targets) = next(examples)
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        # plt.tight_layout()
        img = example_data[i]
        print('pic shape:', img.shape)
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        plt.imshow(img, interpolation='none')
        plt.title(classes[example_targets[i].item()])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def show_pic_torchio(dataloader):   # 展示dataloader里的6张图片
    examples = enumerate(dataloader)         # 组合成一个索引序列
    batch_idx, (example_data, example_location) = next(examples)[0], (next(examples)[1]["data"], next(examples)[1]["location"])
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        # plt.tight_layout()
        img = example_data[i]
        print('pic shape:', img.shape)
        img = img.swapaxes(0, 1)
        img = img.swapaxes(1, 2)
        plt.imshow(img, interpolation='none')
        #plt.title(classes[example_location[i].item()])
        plt.xticks([])
        plt.yticks([])
    plt.show()

def show_pic_torchio_pic(dataloader,show_all):  # 展示dataloader里的6张图片

    for i, batch_exp in enumerate(dataloader):
        x = batch_exp["data"]    ##也只有这一个data，如果不是这个那就是对应的就是标签
        x_image_data = x["data"]
        x_affine_data = x["affine"]
        x_path_data = x["path"]
        x_stem_data = x["stem"]
        x_type_data = x["type"]
        for i_ in range(6):
            plt.subplot(2, 3, i_ + 1)
            img = x_image_data[0, 0, :, :, i_]
            print('pic shape:', img.shape)
            plt.imshow(img, interpolation='none', cmap='Greys')
            plt.xticks([])
            plt.yticks([])
        if show_all:
            plt.show()

def test_dataloader(loader):
    for step, data in enumerate(loader):
        image, label = data
    return image, label

if __name__ == '__main__':
    import torch
    import torch.optim as optim
    import torch.backends.cudnn as cudnn
    from torchsummary import summary

    from Core.config import config
    from Core.function import train
    from Model.intro_vae import Encoder, Decoder
    from DataSet.data_set_2d_intro_vae import get_trainloader
    from Untils.utils import save_checkpoint, create_logger, setup_seed
    from Untils.show_dataloader import show_pic
    from Untils.debug import checkImageMatrix
    trainloader = get_trainloader(config.DATASET.ROOT, config.DATASET.MOD)  ##放入对应的文件的对应的名字，得到相应的
    for i in range(8):
        examples = enumerate(trainloader)
        (path1, path2) = next(examples)[1]['data']['path']
        #print(path1)
        print(path2)


