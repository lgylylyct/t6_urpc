# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
#follow by the main idea of the deive for the medical image and the mae
# --------------------------------------------------------
import argparse
import datetime
import json
import sys
import numpy as np
import os
import time
from pathlib import Path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch
import torch.backends.cudnn as cudnn
import timm.optim.optim_factory as optim_factory
from Core.args_vitmae import get_args_parser
from Core.config_introvae import config
import Untils.misc as misc
from Untils.misc import NativeScalerWithGradNormCount as NativeScaler
import Model.models_mae_vit as models_mae
from Core.function_vitmae import function_vitmae
from Untils.utils import create_logger, setup_seed
from DataSet.DatasetintroVAE.dataset_introvae_without_config import get_trainloader


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = '4'
    misc.init_distributed_mode(args)                                                       ## train for DDP, but i don not use for it now
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

    model = models_mae.__dict__[args.model](img_size=192, in_chans=1, norm_pix_loss=args.norm_pix_loss, mask_strategy=args.mask_strategy)
    model.to(device)
    model_without_ddp = model
    traintype_info = "Model = %s" % str(model_without_ddp)
    print(traintype_info)
    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    ##########################
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    logger.info("base lr: %.2e" % (args.lr * 256 / eff_batch_size))  #0.001  eff_batch_size = 64
    logger.info("actual lr: %.2e" % args.lr)    #0.00025

    logger.info("accumulate grad iterations: %d" % args.accum_iter)
    logger.info("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module

    ###########################################define the par for the train################
    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.add_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))     ## for overfitting
    logger.info(optimizer)
    loss_scaler = NativeScaler()   ##  amp
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)
    ################### train the models for the datatime and the gain the pretrain weigth for the ###################################

    logger.info(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            trainloader.sampler.set_epoch(epoch)
        train_stats = function_vitmae(
            model, trainloader,
            optimizer, device, epoch, loss_scaler,
            args=args
        )
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        logger.info(log_stats)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
