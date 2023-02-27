# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import sys
from typing import Iterable

import torch

import Untils.misc as misc
import Untils.lr_sched as lr_sched

''' 
     we don not want to use tensorboard for the log the result for the dataset 
     change the writer for the value of the log and change the wa  and we want to train one epoch for the dataset
'''

def function_vitmae(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    log_writer=None,
                    args=None):

    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    accum_iter = args.accum_iter    ## you can change this value for getting the more appropriate lr
    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))
##   关于记录数据的对应的记录信息的方式   先接触数据然后使用yield迭代出去进行训练
    for data_iter_step, sampple_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = sampple_dict['data']['data']
        # we use a per iteration (instead of per epoch) lr scheduler? but i don not know why to chose this dense mothod for finetune the
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)
        ##尝试让其有更多的可能和选择，如果出来的结果是一个对应的list，那在后续的实验当中也就不仅仅是一个
        if isinstance(samples, list):
            imgs = samples[0].to(device, non_blocking=True)
            heatmaps = samples[1].to(device, non_blocking=True)
        else:
            imgs = samples.to(device, non_blocking=True)   ## 锁业内存和虚拟内存，简单而言就是只放在GPU内却不进行取出，这样你的读取时间就会大大减小，但是对显存的要求也会更高
            #锁业内存并不会与虚拟内存进行交换（虚拟内存就是硬盘） 主机中的内存有两种对应的形式，一种是锁业，一种不是锁业，显卡中的内存全部都是锁业内存，所以当你的计算机的内存充足的时候，你就可以设置
            #对应的pin_memory = true  当然当你的系统卡住，或者说你的交换内存过多的时候，那就不能了，所以当你的pin_memory=true 的时候那就要同期的设置对应的 non_blocking
            imgs = imgs.squeeze(4)
            heatmaps = None
            ## 关于使用混合精度，那就是autocast+ GradScaler
        with torch.cuda.amp.autocast():  ##在前向过程当中开启对应的autocast  即为计算损失以及生成对应的预测值    而反向传播在autocast之外 loss.backward()  optimizer.step()
            if heatmaps is not None:                                             ## I CAN NOT understand the meaning for the heatmap
                loss, _, _ = model(imgs, mask_ratio=args.mask_ratio, heatmaps=heatmaps)
            else:
                loss, _, _ = model(imgs, mask_ratio=args.mask_ratio)   ##初始的遮盖率在0.75，按照论文里面应该是0.9

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()       ##正确打印模型的运行时间

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)


    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}