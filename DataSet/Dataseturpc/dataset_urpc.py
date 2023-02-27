import os
import torchio as tio
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from Core.config_introvae import config

"""主要写一个关于对应的生成网络的对应的dataset，仿照对应的introvae的写法、得到对应的dataseloaer"""

def get_trainloader(root, mod = 'with_tumor.nii'):
    names = os.listdir(root)

    subjects = []

    for name in names:
        subject = tio.Subject(
            data = tio.ScalarImage(os.path.join(root, name, mod))
        )
        subjects.append(subject)
    ## 在生成图像的预处理过程当中，得到的对应的操作只有归一化  裁剪对应的patch得到对应的
    transforms = tio.Compose([
        tio.RescaleIntensity(out_min_max=(-1, 1)),
        tio.CropOrPad(target_shape=config.TRAIN.PATCH_SIZE+[10])
    ])
    ##
    ##讲对应的subject以及对应的数据增强方式包裹到一起得到对应的增强方式

    dataset = tio.SubjectsDataset(subjects, transforms)

    queue = tio.Queue(
        dataset,
        max_length=3840,
        samples_per_volume=1,
        sampler=tio.UniformSampler(config.TRAIN.PATCH_SIZE+[1]),
        num_workers=0
    )

    loader = DataLoader(queue, batch_size=config.TRAIN.BATCH_SIZE, num_workers=0)
    return loader, dataset



if __name__ == "__main__":

    #root = r'E:\NPCICdataset\Patient_Image\test_intro_vae_result\001_ori_register_nii'
    root = r'E:\NPCICdataset\Patient_Image\test_intro_vae_result\002_spilt_mask_nii'

    train_loader, train_dataset = get_trainloader(root, mod='without_tumor.nii')
    spacing = train_dataset[1].spacing
    train_dataset[1].plot()
    show_slice = train_dataset[1]['data']['data'][0, :, :, 3]
    print(show_slice.shape)

    plt.imshow(show_slice, cmap="gray")
    plt.show()
    print(spacing)



