import os
import torchio as tio
import SimpleITK as sitk
from torch.utils.data import DataLoader
from Untils.debug import checkImageMatrix
from Untils.show_dataloader import show_pic,show_pic_torchio,test_dataloader,show_pic_torchio_pic
from Core.config import config
from matplotlib import pyplot as plt


def get_trainloader(root, mod_type):
    names = os.listdir(root)
    subjects = []

    for name in names:
        subject = tio.Subject(
            data=tio.ScalarImage(os.path.join(root, name, mod_type))
        )
        subjects.append(subject)
    ##这个subject是一个非常强大的封装，封装了图像对象以及其的相关信息，常见的例子是在对象当中封装图像本身，对应的mask，以及图像的信息以及患者的姓名...之类的信息
    # load() 在内存当中加载
    # plot() 绘制图像
    # shape spacing spatial 属性校验值

    transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        tio.CropOrPad(target_shape=config.TRAIN.PATCH_SIZE + [16])
    ])

    dataset = tio.SubjectsDataset(subjects, transforms)
    ##在这里面的这个dataset就是dataset 然后使用的对应的是dataset[n]，返回的就是你原来的subject
    ##在原始的继承当中  在这里面dataset.__getitem__(n-1)  就可以subject 是个这个特殊类型的类

    queue = tio.Queue(
        subjects_dataset=dataset,
        max_length=config.TRAIN.QUEUE_LENGTH,
        samples_per_volume=config.TRAIN.SAMPLES_PER_VOLUME,
        sampler=tio.UniformSampler(config.TRAIN.PATCH_SIZE+[1]), ##这里面是patch的大小
        shuffle_subjects=config.DATASET.subject_shuffle,
        shuffle_patches=config.DATASET.patch_shuffle,
    )  ##使用队列机制在进去gpu之前预处理patch，然后在GPU空闲的时候将处理好的patch加载进去
    loader = DataLoader(queue, config.TRAIN.BATCH_SIZE, num_workers=0)
    try_loader = DataLoader(dataset,config.TRAIN.BATCH_SIZE, num_workers=0)
    return loader, try_loader


def get_trainloader_2d(root, mod_type):
    names = os.listdir(root)
    subjects = []

    for name in names:
        subject = tio.Subject(
            data=tio.ScalarImage(os.path.join(root, name, mod_type))
        )
        subjects.append(subject)
    ##这个subject是一个非常强大的封装，封装了图像对象以及其的相关信息，常见的例子是在对象当中封装图像本身，对应的mask，以及图像的信息以及患者的姓名...之类的信息
    # load() 在内存当中加载
    # plot() 绘制图像
    # shape spacing spatial 属性校验值

    transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        tio.CropOrPad(target_shape=config.TRAIN.PATCH_SIZE + [16])
    ])

    dataset = tio.SubjectsDataset(subjects, transforms)
    ##在这里面的这个dataset就是dataset 然后使用的对应的是dataset[n]，返回的就是你原来的subject
    ##在原始的继承当中  在这里面dataset.__getitem__(n-1)  就可以subject 是个这个特殊类型的类

    queue = tio.Queue(
        subjects_dataset=dataset,
        max_length=config.TRAIN.QUEUE_LENGTH,
        samples_per_volume=config.TRAIN.SAMPLES_PER_VOLUME,
        sampler=tio.UniformSampler(config.TRAIN.PATCH_SIZE+[1]), ##这里面是patch的大小
        shuffle_subjects=config.DATASET.subject_shuffle,
        shuffle_patches=config.DATASET.patch_shuffle,
    )  ##使用队列机制在进去gpu之前预处理patch，然后在GPU空闲的时候将处理好的patch加载进去
    loader = DataLoader(queue, config.TRAIN.BATCH_SIZE, num_workers=0)
    try_loader = DataLoader(dataset,config.TRAIN.BATCH_SIZE, num_workers=0)
    return loader, try_loader


if __name__ == '__main__':
    #loader,try_loader = get_trainloader(config.DATASET.ROOT, config.DATASET.MOD)
    _, try_loader_ = get_trainloader_2d(config.DATASET.ROOT, config.DATASET.MOD)
    # show_pic_torchio(loader)
    #
    for i, batch_exp in enumerate(loader):
         x = batch_exp['data'] ##这里的data对应的是上面封装subject时候的data,之后你会得到对应的五个字典

    for i, batch_exp in enumerate(dataloader):
        x = batch_exp["data"][tio.DATA]   ##也只有这一个data，如果不是这个那就是对应的就是标签
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




        ##检查对应的是不是相同的



    #checkImageMatrix(loader)

    print('show_loader')
