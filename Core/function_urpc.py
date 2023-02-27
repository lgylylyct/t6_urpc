import torch
import numpy as np
import torch.nn as nn
import torchio as tio
import torch.nn.functional
from medpy.metric.binary import dc
import tqdm
from Untils import losses, metrics, ramps

from Untils.utils import AverageMeter, determine_device

##需要停止一个对应的梯度的返回传递就可以对应的相应的是是z的梯度对应的相应的
def D(p, z):
    return - torch.nn.functional.cosine_similarity(p, z.detach(), dim=-1).mean()


def train(model, train_generator, optimizer, criterion, logger, config, epoch):
    model.train()
    segnet = model['segnet']
    projections = model['projections']
    predictions = model['predictions']
    losses = AverageMeter()
    # if scaler is not supported, it switches to default mode, the training can also continue
    scaler = torch.cuda.amp.GradScaler()     ## the first step for the amp in loss function calculation
    num_iter = config.TRAIN.NUM_BATCHES
    for idx in range(num_iter):
        data_dict = next(train_generator)
        data = data_dict['data']
        label = data_dict['label']
        if config.TRAIN.PARALLEL:
            devices = config.TRAIN.DEVICES
            data = data.cuda(devices[0])
            label = [l.cuda(devices[0]) for l in label]
        else:
            device = determine_device()
            data = data.to(device)
            label = [l.to(device) for l in label]
        # run training
        with torch.cuda.amp.autocast():
            #x, rec = data[:, :-1], data[:, -1:] ##in the toydata set for the testing ,we can change it to gain the right
            x, rec = torch.cat([data[:, :], data[:, -1:]], dim=1), data[:, -1:]
            out, embeds = segnet(x, rec)
            l_dc = criterion(out, label[0])
            # simsiam
            l_sim = .0
            for i in range(4):
                mask = 1 - label[i]
                v1, v2 = projections[i](embeds[i][0] * mask), projections[i](embeds[i][1] * mask)
                p1, p2 = predictions[i](v1), predictions[i](v2)
                l_sim += .5 * D(p1, v2) + .5 * D(p2, v1)
            loss = l_dc + l_sim
        losses.update(loss.item(), config.TRAIN.BATCH_SIZE)
        # do back-propagation
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
        scaler.step(optimizer)
        scaler.update()

        if idx % config.PRINT_FREQ == 0:
            msg = 'Epoch: [{0}][{1}/{2}]\t' \
                  'Loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, i, num_iter,
                loss=losses,
            )
            logger.info(msg)


def inference(model, dataset, logger, config):
    model.eval()

    perfs = [AverageMeter() for _ in range(2)]
    nonline = nn.Softmax(dim=1)
    scores = {}
    for case in dataset:
        patch_size = config.INFERENCE.PATCH_SIZE
        patch_overlap = config.INFERENCE.PATCH_OVERLAP
        # torchio does not support 2d slice natively, it can only treat it as pseudo 3d patch
        patch_size = [1] + patch_size
        patch_overlap = [0] + patch_overlap
        # data shape cannot be smaller than patch size, maybe pad is needed
        target_shape = np.max([patch_size, case['data'].shape[1:]], 0)
        transform = tio.CropOrPad(target_shape)
        case = transform(case)
        # sliding window sampler
        sampler = tio.inference.GridSampler(case, patch_size, patch_overlap)
        loader = torch.utils.data.DataLoader(sampler, config.INFERENCE.BATCH_SIZE)
        aggregator = tio.inference.GridAggregator(sampler, 'average')

        with torch.no_grad():
            for data_dict in loader:
                data = data_dict['data'][tio.DATA]
                label = data_dict['label'][tio.DATA]
                data = data.squeeze(2)
                label = label.squeeze(2)
                if config.TRAIN.PARALLEL:
                    devices = config.TRAIN.DEVICES
                    data = data.cuda(devices[0])
                    label = label.cuda(devices[0])
                else:
                    device = determine_device()
                    data = data.to(device)
                    label = label.to(device)
                with torch.cuda.amp.autocast():
                    x, rec = data[:, :-1], data[:, -1:]
                    out = model(x, rec)
                    out = nonline(out)
                locations = data_dict[tio.LOCATION]
                # I love and hate torchio ...
                out = out.unsqueeze(2)
                aggregator.add_batch(out, locations)
            # form final prediction
            pred = aggregator.get_output_tensor()
            pred = torch.argmax(pred, dim=0).cpu().numpy()
            label = case['label'][tio.DATA][0].numpy() > 0  # only wt segmentation is supported
            name = case['name']
            # quantitative analysis
            # only dice score is computed by default, you can also add hd95, assd and sensitivity et al
            scores[name] = {}
            for c in np.unique(label):
                scores[name][int(c)] = dc(pred == c, label == c)
                perfs[int(c)].update(scores[name][c])
        del case
    logger.info('------------ dice scores ------------')
    logger.info(scores)
    for c in range(2):
        logger.info(f'class {c} dice mean: {perfs[c].avg}')
    logger.info('------------ ----------- ------------')
    perf = np.mean([perfs[c].avg for c in range(1, 2)])
    return perf



def function_urpc(model,trainloader, args,optimizer,loss, kl_distance,iter_num,max_iterations):
    ##其实我在思考....这个是不是和facebook的写法一样叫做train_one_epoch才是更加合理合适的，不得不说他们的代码学习性好强
    model.train()
    base_lr = args.base_lr
    ce_loss = loss[0]
    dice_loss = loss[1]

    def get_current_consistency_weight(epoch):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

    for i_batch, sampled_batch in enumerate(trainloader):
        volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

        ## volume_batch is the imgae of the

        outputs, outputs_aux1, outputs_aux2, outputs_aux3 = model(
            volume_batch)
        outputs_soft = torch.softmax(outputs, dim=1)
        outputs_aux1_soft = torch.softmax(outputs_aux1, dim=1)
        outputs_aux2_soft = torch.softmax(outputs_aux2, dim=1)
        outputs_aux3_soft = torch.softmax(outputs_aux3, dim=1)

        loss_ce = ce_loss(outputs[:args.labeled_bs],
                          label_batch[:args.labeled_bs][:].long())
        loss_ce_aux1 = ce_loss(outputs_aux1[:args.labeled_bs],
                               label_batch[:args.labeled_bs][:].long())
        loss_ce_aux2 = ce_loss(outputs_aux2[:args.labeled_bs],
                               label_batch[:args.labeled_bs][:].long())
        loss_ce_aux3 = ce_loss(outputs_aux3[:args.labeled_bs],
                               label_batch[:args.labeled_bs][:].long())

        loss_dice = dice_loss(
            outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
        loss_dice_aux1 = dice_loss(
            outputs_aux1_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
        loss_dice_aux2 = dice_loss(
            outputs_aux2_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
        loss_dice_aux3 = dice_loss(
            outputs_aux3_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))

        supervised_loss = (loss_ce + loss_ce_aux1 + loss_ce_aux2 + loss_ce_aux3 +
                           loss_dice + loss_dice_aux1 + loss_dice_aux2 + loss_dice_aux3) / 8

        preds = (outputs_soft + outputs_aux1_soft +
                 outputs_aux2_soft + outputs_aux3_soft) / 4

        variance_main = torch.sum(kl_distance(
            torch.log(outputs_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
        exp_variance_main = torch.exp(-variance_main)

        variance_aux1 = torch.sum(kl_distance(
            torch.log(outputs_aux1_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
        exp_variance_aux1 = torch.exp(-variance_aux1)

        variance_aux2 = torch.sum(kl_distance(
            torch.log(outputs_aux2_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
        exp_variance_aux2 = torch.exp(-variance_aux2)

        variance_aux3 = torch.sum(kl_distance(
            torch.log(outputs_aux3_soft[args.labeled_bs:]), preds[args.labeled_bs:]), dim=1, keepdim=True)
        exp_variance_aux3 = torch.exp(-variance_aux3)

        consistency_weight = get_current_consistency_weight(iter_num // 150)
        consistency_dist_main = (
                                        preds[args.labeled_bs:] - outputs_soft[args.labeled_bs:]) ** 2

        consistency_loss_main = torch.mean(
            consistency_dist_main * exp_variance_main) / (torch.mean(exp_variance_main) + 1e-8) + torch.mean(
            variance_main)

        consistency_dist_aux1 = (
                                        preds[args.labeled_bs:] - outputs_aux1_soft[args.labeled_bs:]) ** 2
        consistency_loss_aux1 = torch.mean(
            consistency_dist_aux1 * exp_variance_aux1) / (torch.mean(exp_variance_aux1) + 1e-8) + torch.mean(
            variance_aux1)

        consistency_dist_aux2 = (
                                        preds[args.labeled_bs:] - outputs_aux2_soft[args.labeled_bs:]) ** 2
        consistency_loss_aux2 = torch.mean(
            consistency_dist_aux2 * exp_variance_aux2) / (torch.mean(exp_variance_aux2) + 1e-8) + torch.mean(
            variance_aux2)

        consistency_dist_aux3 = (
                                        preds[args.labeled_bs:] - outputs_aux3_soft[args.labeled_bs:]) ** 2
        consistency_loss_aux3 = torch.mean(
            consistency_dist_aux3 * exp_variance_aux3) / (torch.mean(exp_variance_aux3) + 1e-8) + torch.mean(
            variance_aux3)

        consistency_loss = (consistency_loss_main + consistency_loss_aux1 +
                            consistency_loss_aux2 + consistency_loss_aux3) / 4
        loss = supervised_loss + consistency_weight * consistency_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        lr_ = base_lr * (1.0 - iter_num / max_iterations) ** 0.9
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_

        iter_num = iter_num + 1


