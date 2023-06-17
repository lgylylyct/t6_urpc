import argparse
import logging
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)


########## Set The WorkSpace Path ##########
dirname, filename = os.path.split(os.path.abspath(__file__))
os.chdir(dirname)
print("The Current WorkPlace: {}".format(os.getcwd()))

import random, sys, json, platform, importlib

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim

from torch.nn.modules.loss import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision import transforms


from DataSet.Datasetsemiu.dataset import BaseDataSets, RandomGenerator, TwoStreamBatchSampler
from Untils import losses, ramps
from Val.val_2d import test_single_volume_ds
from networks.swin_unetr import SwinUNETR
from networks.unet import UNet, UNet_DS, UNet_URPC, UNet_CCT

import Untils.Utils_Lin as U

import cv2

cpu_num = 1
cv2.setNumThreads(cpu_num)
os.environ["OMP_NUM_THREADS"] = str(cpu_num)
os.environ["OPENBLAS_NUM_THREADS"] = str(cpu_num)
os.environ["MKL_NUM_THREADS"] = str(cpu_num)
os.environ["VECLIB_MAXIMUM_THREADS"] = str(cpu_num)
os.environ["NUMEXPR_NUM_THREADS"] = str(cpu_num)
torch.set_num_threads(cpu_num)


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def train(args, snapshot_path):
    # set infomation logger
    info_logger_path = os.path.join(snapshot_path, "info.log")
    info_logger = logging.Logger("A")
    info_logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    print_headler = logging.StreamHandler()
    file_headler = logging.FileHandler(info_logger_path, mode="w", encoding="utf-8",)
    print_headler.setLevel(logging.DEBUG)
    file_headler.setLevel(logging.DEBUG)
    print_headler.setFormatter(formatter)
    file_headler.setFormatter(formatter)
    info_logger.addHandler(print_headler)
    info_logger.addHandler(file_headler)
    print("information of logger:", info_logger_path)
    
    info_logger.info("is semi-supervised exp: {}".format(args.semi_sup))

    ######## arg ##########
    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations

    ########## dataset ##########
    db_train = BaseDataSets(
        base_img_dir=args.img_path,
        base_mask_dir=args.mask_path,
        mod=args.mod,
        label_mod_type=args.label_mod_type,
        split="train",
        train_num=None,
        transform=transforms.Compose([RandomGenerator(args.patch_size)]),
    )
    db_val = BaseDataSets(
        base_img_dir=args.img_path,
        base_mask_dir=args.mask_path,
        mod=args.mod,
        label_mod_type=args.label_mod_type,
        split="val",
    )

    info_logger.info(
        "Total training data number is: {}, labeled training dat number is: {}".format(
            len(db_train), args.labeled_num
        )
    )
    labeled_idxs = list(range(0, args.labeled_num))
    unlabeled_idxs = list(range(args.labeled_num, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(
        labeled_idxs, unlabeled_idxs, batch_size, batch_size - args.labeled_bs
    )

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainloader = DataLoader(
        db_train,
        batch_sampler=batch_sampler,
        num_workers=args.number_worker,
        pin_memory=False,
        worker_init_fn=worker_init_fn,
        shuffle=False,
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=args.number_worker)
    info_logger.info("{} iterations per epoch".format(len(trainloader)))

    ########## model ##########
    if args.model == "SwinUNETR":
        model = SwinUNETR(
            img_size=224, out_channels=args.num_classes, in_channels=1, spatial_dims=2
        )
    elif args.model == "unet_urpc":
        model = UNet_URPC(in_chns=1, class_num=args.num_classes)
    model = model.to(args.device)

    ########## optimizer ##########
    if args.optim.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
        info_logger.info("use SGD optimizer")
        
    elif args.optim.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=base_lr, weight_decay=0.0001)
        info_logger.info("use Adam optimizer")
        
    elif args.optim.lower() == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=0.0001)
        info_logger.info("use AdamW optimizer")

    ########## loss function ##########
    kl_distance = nn.KLDivLoss(reduction="none")
    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)

    ########## traning ##########
    iter_num = 0
    max_epoch = max_iterations // len(trainloader)
    best_performance = 0.0
    for epoch_num in range(max_epoch):
        consistency_weight = get_current_consistency_weight(epoch_num)

        list_loss = []
        list_loss_ce = []
        list_loss_dice = []
        list_loss_consistency = []

        for i_batch, sampled_batch in enumerate(trainloader):
            model.train()
            volume_batch, label_batch = sampled_batch["image"], sampled_batch["label"]
            volume_batch, label_batch = (
                volume_batch.to(args.device),
                label_batch.to(args.device),
            )
            # U.checkImageMatrix({"volume_batch": volume_batch, "label_batch": label_batch})

            outputs, outputs_aux1, outputs_aux2, outputs_aux3 = model(volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
            outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
            outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)

            # CE loss -- supervised
            loss_ce = ce_loss(
                outputs[: args.labeled_bs], label_batch[: args.labeled_bs][:].long()
            )
            loss_ce_aux1 = ce_loss(
                outputs_aux1[: args.labeled_bs], label_batch[: args.labeled_bs][:].long()
            )
            loss_ce_aux2 = ce_loss(
                outputs_aux2[: args.labeled_bs], label_batch[: args.labeled_bs][:].long()
            )
            loss_ce_aux3 = ce_loss(
                outputs_aux3[: args.labeled_bs], label_batch[: args.labeled_bs][:].long()
            )

            # Dice loss -- supervised
            loss_dice = dice_loss(
                outputs_soft[: args.labeled_bs], label_batch[: args.labeled_bs].unsqueeze(1)
            )
            loss_dice_aux1 = dice_loss(
                outputs_aux1_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )
            loss_dice_aux2 = dice_loss(
                outputs_aux2_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )
            loss_dice_aux3 = dice_loss(
                outputs_aux3_soft[: args.labeled_bs],
                label_batch[: args.labeled_bs].unsqueeze(1),
            )

            # overall loss -- supervised
            supervised_loss = (
                loss_ce
                + loss_ce_aux1
                + loss_ce_aux2
                + loss_ce_aux3
                + loss_dice
                + loss_dice_aux1
                + loss_dice_aux2
                + loss_dice_aux3
            ) / 8

            # preds is the average of all segementation
            preds = (
                outputs_soft + outputs_aux1_soft + outputs_aux2_soft + outputs_aux3_soft
            ) / 4

            # kl_distance --> variance --> exp variance -- unsupervised
            kl_main = kl_distance(
                torch.log(outputs_soft[args.labeled_bs :] + 1e-8), preds[args.labeled_bs :]
            )
            variance_main = torch.sum(kl_main, dim=1, keepdim=True,)
            exp_variance_main = torch.exp(-variance_main)

            kl_aux1 = kl_distance(
                torch.log(outputs_aux1_soft[args.labeled_bs :] + 1e-8),
                preds[args.labeled_bs :],
            )
            variance_aux1 = torch.sum(kl_aux1, dim=1, keepdim=True,)
            exp_variance_aux1 = torch.exp(-variance_aux1)

            kl_aux2 = kl_distance(
                torch.log(outputs_aux2_soft[args.labeled_bs :] + 1e-8),
                preds[args.labeled_bs :],
            )
            variance_aux2 = torch.sum(kl_aux2, dim=1, keepdim=True,)
            exp_variance_aux2 = torch.exp(-variance_aux2)

            kl_aux3 = kl_distance(
                torch.log(outputs_aux3_soft[args.labeled_bs :] + 1e-8),
                preds[args.labeled_bs :],
            )
            variance_aux3 = torch.sum(kl_aux3, dim=1, keepdim=True,)
            exp_variance_aux3 = torch.exp(-variance_aux3)

            # consistency -- L2 loss -- unsupervised
            consistency_dist_main = (
                preds[args.labeled_bs :] - outputs_soft[args.labeled_bs :]
            ) ** 2
            consistency_loss_main = torch.mean(consistency_dist_main * exp_variance_main) / (
                torch.mean(exp_variance_main) + 1e-8
            ) + torch.mean(variance_main)

            consistency_dist_aux1 = (
                preds[args.labeled_bs :] - outputs_aux1_soft[args.labeled_bs :]
            ) ** 2
            consistency_loss_aux1 = torch.mean(consistency_dist_aux1 * exp_variance_aux1) / (
                torch.mean(exp_variance_aux1) + 1e-8
            ) + torch.mean(variance_aux1)

            consistency_dist_aux2 = (
                preds[args.labeled_bs :] - outputs_aux2_soft[args.labeled_bs :]
            ) ** 2
            consistency_loss_aux2 = torch.mean(consistency_dist_aux2 * exp_variance_aux2) / (
                torch.mean(exp_variance_aux2) + 1e-8
            ) + torch.mean(variance_aux2)

            consistency_dist_aux3 = (
                preds[args.labeled_bs :] - outputs_aux3_soft[args.labeled_bs :]
            ) ** 2
            consistency_loss_aux3 = torch.mean(consistency_dist_aux3 * exp_variance_aux3) / (
                torch.mean(exp_variance_aux3) + 1e-8
            ) + torch.mean(variance_aux3)

            consistency_loss = (
                consistency_loss_main
                + consistency_loss_aux1
                + consistency_loss_aux2
                + consistency_loss_aux3
            ) / 4

            # overall loss -- semi-supervised
            if args.semi_sup:
                loss = supervised_loss + consistency_weight * consistency_loss
            else:
                loss = supervised_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            list_loss.append(loss.item())
            list_loss_ce.append(loss_ce.item())
            list_loss_dice.append(loss_dice.item())
            list_loss_consistency.append(consistency_loss.item())

            iter_num = iter_num + 1

            # if (i_batch + 1) % (len(trainloader) // 4) == 0 or i_batch + 1 == len(trainloader):
            # pass

            # break

        info_logger.info(
            "epoch_num %4d / %d : loss : %.4f, loss_ce: %.4f, loss_dice: %.4f, loss_consistency: %.4f, weight_consistency: %.4f"
            % (
                epoch_num + 1,
                max_epoch,
                np.mean(list_loss),
                np.mean(list_loss_ce),
                np.mean(list_loss_dice),
                np.mean(list_loss_consistency),
                consistency_weight,
            )
        )

        if (epoch_num + 1) % (max_epoch // 100) != 0:
            continue

        # validation
        model.eval()
        metric_list = 0.0
        num_val = 0
        for i_batch, sampled_batch in enumerate(valloader):
            if sampled_batch["image"].shape[1] == 0:
                continue

            metric_i = test_single_volume_ds(
                sampled_batch["image"],
                sampled_batch["label"],
                model,
                patch_size=args.patch_size,
                classes=num_classes,
                device=args.device,
            )
            metric_list += np.array(metric_i)
            num_val += 1

            # if (i_batch + 1) % (len(valloader) // 4) == 0 or i_batch + 1 == len(valloader):
            #     info_logger.info("val: {}/{}".format(i_batch + 1, len(valloader)))
        metric_list = metric_list / num_val

        performance = np.mean(metric_list, axis=0)[0]
        mean_hd95 = np.mean(metric_list, axis=0)[1]

        if performance > best_performance:
            best_performance = performance
            save_mode_path = os.path.join(
                snapshot_path,
                "iter_{:0>5}_dice_{:.4f}.pth".format(iter_num, best_performance),
            )
            save_best = os.path.join(snapshot_path, "{}_best_model.pth".format(args.model))
            torch.save(model.state_dict(), save_mode_path)
            torch.save(model.state_dict(), save_best)

        info_logger.info(
            "iteration %d : mean_dice : %f mean_hd95 : %f \n"
            % (iter_num, performance, mean_hd95)
        )

        # if iter_num % 3000 == 0:
        #     save_mode_path = os.path.join(snapshot_path, "iter_" + str(iter_num) + ".pth")
        #     torch.save(model.state_dict(), save_mode_path)
        #     info_logger.info("save model to {}".format(save_mode_path))

    return "Training Finished!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--args", type=str, default="Sup_Unet_T2_D6_W1_AdamW")
    parser.add_argument("--deterministic", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337, help="random seed")
    if platform.system() == "Windows":
        parser.add_argument("--device", type=str, default="cpu")
        parser.add_argument("--number_worker", type=int, default=0)
    elif platform.system() == "Linux":
        parser.add_argument("--device", type=str, default="cuda:2")
        parser.add_argument("--number_worker", type=int, default=0)

    args_ = parser.parse_args()

    args = importlib.import_module("Trainer.args.{}".format(args_.args)).args

    args.args = args_.args
    args.exp = args_.args
    args.deterministic = args_.deterministic
    args.seed = args_.seed
    args.device = args_.device
    args.number_worker = args_.number_worker

    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "output_dir/{}".format(args.exp)
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)

    info_args_path = os.path.join(snapshot_path, "arg.txt")
    with open(info_args_path, "w") as file:
        file.write(json.dumps(args.__dict__, indent=4))
    print("information of args:", info_args_path)

    train(args, snapshot_path)
