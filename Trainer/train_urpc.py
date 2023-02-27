# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import datetime
import os
import time
from pathlib import Path
import tqdm

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.modules.loss import CrossEntropyLoss

# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

import Untils.misc as misc
from Untils.misc import NativeScalerWithGradNormCount as NativeScaler
from Untils import losses

from Core.args_urpc import get_args_parser
from Core.function_urpc import function_urpc
from Untils.utils import create_logger, setup_seed

from Core.config_urpc import config
from DataSet.Dataseturpc.dataset_urpc import get_trainloader
from networks.net_factory import net_factory



def main(args):
    ### prepare the dataset of the model of the follow train pipline
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #misc.init_distributed_mode(args)                                       ## train for DDP, but i don not use for it now
    logger = create_logger(config.LOG.OUT_DIR, config.LOGFILE)
    device = torch.device(args.device)
    seed = args.seed + misc.get_rank()
    setup_seed(seed)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED
    file_info1 = 'job dir: {}'.format(os.path.dirname(os.path.realpath(__file__)))
    file_info2 = "{}".format(args).replace(', ', ',\n')
    logger.info(file_info1)
    logger.info(file_info2)
    print('the log dir of this file is: {}'.format(config.LOG.OUT_DIR))
    trainloader, traindataset = get_trainloader(root=config.DATASET.ROOT, mod=config.MOD)


    base_lr = args.base_lr
    num_classes = args.num_classes
    batch_size = args.batch_size
    max_iterations = args.max_iterations


    #####################################################################################################
    model = net_factory(net_type=args.model, in_chns=1,
                        class_num=num_classes)
    model.to(device)
    model_without_ddp = model
    traintype_info = "Model = %s" % str(model_without_ddp)
    print(traintype_info)
    eff_batch_size = batch_size * args.accum_iter * misc.get_world_size()

    ##########################
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logger.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))  # 0.001  eff_batch_size = 64
    logger.info("actual lr: %.2e" % args.lr)  # 0.00025

    logger.info("accumulate grad iterations: %d" % args.accum_iter)
    logger.info("effective batch size: %d" % eff_batch_size)

    ###########################################define the par for the train################
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.SGD(model.parameters(), lr=base_lr,
                                momentum=0.9, weight_decay=0.0001)
    logger.info(optimizer)
    loss_scaler = NativeScaler()  ##  amp
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    ################### train the models for the datatime and the gain the pretrain weigth for the ###################################

    logger.info(f"Start training for {args.epochs} epochs")

    ce_loss = CrossEntropyLoss()
    dice_loss = losses.DiceLoss(num_classes)
    loss = [ce_loss, dice_loss]
    logger.info("{} iterations per epoch".format(len(trainloader)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    kl_distance = nn.KLDivLoss(reduction='none')
    iterator = tqdm(range(max_epoch), ncols=70)

    start_time = time.time()
    for epoch_num in iterator:

        train_stats = function_urpc(
            model, trainloader,
            optimizer, device, epoch_num,loss,loss_scaler,iterator,kl_distance
        )

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch_num, }
        logger.info(log_stats)
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))

    ### save the model which is best for the

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    # if args.output_dir:
    #     Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
