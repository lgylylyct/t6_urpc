import platform
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import Untils.operation as U
import logging
import random


def _debugArgs(args):
    # Train
    args.is_train = True
    args.gpu = -1

    args.exp_type = "MitoEM"
    args.config = "MitoEM.pretext_task.Z001_RD"
    args.exp_name = "XXX"

    # Resume
    # args.resume_exp = "UGFB-BYOLNetwork-XXX-2022_08_20_21_37_45\Trials1"

    # Test  # UGFB-BYOLNetwork-XXX-2022_08_20_21_37_45
    # args.is_train = False
    # args.time = "2022_08_20_21_37_45"
    # args.all_test = True

    # Time Consume
    # args.time_consume = True


def _debugCong(config):
    config["print_paras_index"] = []
    config["print_paras_name"] = []
    # config["modalities"] = ["T1", "T2"]

    # config["all_testsets"] = ["testWuZhou", "testFoShan"]

    # config["epochs"] = 1
    # config["train_batch_size"] = 2
    # config["val_batch_size"] = 1
    # config["test_batch_size"] = 1
    # config["testset"] = "val"

    config["is_amp"] = False
    config["buffer"] = False
    config["debug_iteration"] = 1
    # config["train_dataload_shuffle"] = False

    # config["min_epoch_rate"] = 0

    # config["only_mask_region"] = False

    # # geometry augmentation
    # config["aug_shift_scale_rotate"] = 1
    # config["aug_translation"] = 0
    # config["aug_rotate"] = 0
    # config["aug_flipHW"] = 0
    # config["aug_rotate90HW"] = 0
    # # intensity augmentation
    # config["aug_intensity_CLAHE"] = 0
    # config["aug_intensity_gamma"] = 0
    # config["aug_intensity_shift"] = 0
    # config["aug_intensity_scale"] = 0
    # # perturbation augmentation
    # config["aug_gaussian_noise"] = 0
    # config["aug_gaussian_smooth"] = 0

    # config["save_init_model"] = True
    # config["introduce_normal"] = True
    # config["show_patient_id"] = ["45194"]
    # config["show"] = [False, False, False, False, True, False]
    # config["is_max_mask_slice"] = True

    # Test
    # config["current_KF"] = 5
    # config["end_KF"] = 3
    # config["num_KF"] = 5
    # config["current_trial"] = 1
    # config["num_trial"] = 3
    config["test_mode"] = "Best"

    # config['data_name'] = 'L_KF10X'
    # config["val_batch_size"] = 0
    # config['test_batch_size'] = 1
    # config['epochs'] = 3
    # config['not_translayer'] = 'layer1-layer2-layer4'
    pass


def getInfoLogger():
    info_logger = logging.Logger("GCLR")
    info_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    print_headler = logging.StreamHandler()
    print_headler.setLevel(logging.INFO)
    print_headler.setFormatter(formatter)
    info_logger.addHandler(print_headler)
    return info_logger


def getGrid(num):
    s2 = int(np.ceil(np.sqrt(num)))
    if num % s2 == 0:
        s1 = int(num / s2)
    else:
        s1 = int(num / s2) + 1
    return s1, s2


def checkImageMatrix(matrix_dict: dict, save_dir=None, single=False):
    if platform.system() == "Linux" and save_dir is None:
        return

    for i, (name, matrix) in enumerate(matrix_dict.items()):
        if not isinstance(matrix, np.ndarray) and not isinstance(matrix, torch.Tensor):
            continue

        ############## convert to numpy ##############
        if isinstance(matrix, torch.Tensor):
            matrix = matrix.clone()
            matrix = U.toNumpy(matrix)
        else:
            matrix = matrix.copy()
            matrix = matrix.squeeze()

        ############## if 4D, get the first channel ##############
        if name.endswith("__C"):
            if len(matrix.shape) == 5:
                matrix = matrix[0, ...]
        else:
            if len(matrix.shape) == 4:
                matrix = matrix[0, ...]

        ############## if matrix is label, use jet cmap ##############
        is_jet = isinstance(matrix, torch.LongTensor) or isinstance(matrix, np.int) or matrix.max() == 10

        ############## imshow ##############
        fig = plt.figure(figsize=(20, 10), num=name)

        if len(matrix.shape) == 3 and isinstance(single, bool) and single:
            matrix = matrix[random.randint(0, len(matrix) - 1)]
        elif len(matrix.shape) == 3 and str(single).isnumeric():
            if single < (len(matrix) - 1):
                matrix = matrix[single]
            else:
                matrix = matrix[0]
                print("single_index{} > len(matrix){}, so return matrix[0]".format(single, len(matrix)))

        if name.endswith("__C"):
            if len(matrix.shape) == 3:
                ax = fig.subplots()
                ax.set_title(name)
                ax.set_axis_off()
                ax.imshow(matrix)

            elif len(matrix.shape) == 4:
                s1, s2 = getGrid(len(matrix))
                axs = fig.subplots(s1, s2)
                for j, (ax, matrix_j) in enumerate(zip(axs.reshape(-1), matrix)):
                    ax.set_title(name)
                    ax.imshow(matrix_j)
                for ax in axs.reshape(-1):
                    ax.set_axis_off()
        else:
            if len(matrix.shape) == 2:
                ax = fig.subplots()
                ax.set_title(name)
                ax.set_axis_off()
                ax.imshow(matrix, cmap="jet" if is_jet else "gray")

            elif len(matrix.shape) == 3:
                s1, s2 = getGrid(len(matrix))
                axs = fig.subplots(s1, s2)
                for j, (ax, matrix_j) in enumerate(zip(axs.reshape(-1), matrix)):
                    ax.set_title(name)
                    ax.imshow(matrix_j, cmap="jet" if is_jet else "gray")
                for ax in axs.reshape(-1):
                    ax.set_axis_off()

        if save_dir is not None:
            U.makeDirs(save_dir)
            fig.savefig(os.path.join(save_dir, name + ".png"))
            plt.close("all")

    if save_dir is None:
        plt.show()


def checkImages(imgs_dict: dict, save_path=None):
    if platform.system() == "Linux" and save_path is None:
        return

    fig = plt.figure(figsize=(16, 8))
    s1, s2 = getGrid(len(imgs_dict))
    axs = fig.subplots(s1, s2)
    for ax in axs.reshape(-1):
        ax.set_axis_off()

    for i, (ax, (name, img)) in enumerate(zip(axs.reshape(-1), imgs_dict.items())):
        if not isinstance(img, np.ndarray) and not isinstance(img, torch.Tensor):
            continue
        ############## convert to numpy ##############
        if isinstance(img, torch.Tensor):
            img = img.clone()
            img = U.toNumpy(img)
        else:
            img = img.copy()
            img = img.squeeze()

        ############## if 4D, get the first channel ##############
        if len(img.shape) > 2:
            continue

        ############## if matrix is label, use jet cmap ##############
        is_jet = isinstance(img, torch.LongTensor) or isinstance(img, np.int) or img.max() == 10

        ############## draw ##############
        ax.set_title(name)
        ax.imshow(img, cmap="jet" if is_jet else "gray")

        if save_path is not None:
            fig.savefig(save_path)
            plt.close("all")

    if save_path is None:
        plt.show()

