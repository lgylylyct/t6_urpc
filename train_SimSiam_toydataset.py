import argparse
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from Model.CLFC_SimSiam import SegNet, prediction, projection
from Core.config_seg import config
from Loss.scheduler import PolyScheduler
from Core.function_seg import train, inference                                      ##在这个分割的函数过程当中包含了最重要的部分
from Loss.loss import DiceCELoss
from DataSet.DatasetSimSiam.dataset import get_validset, get_validset_one_fold
from DataSet.DatasetSimSiam.dataloader2 import get_trainloader, get_trainloader_one_fold
from DataSet.DatasetSimSiam.augmenter import get_train_generator
from Untils.utils import determine_device, save_checkpoint, create_logger, setup_seed


"需要解决的问题：小的数据集要换成大的数据集" \
"要有对应的数据的生成放到对应的位置以及对应的结果上去" \
"将验证集和训练集之间的结果与训练的模型对应之间要分开"\
"在测试阶段熟悉每个过程当中的结果以及对应的过程以及对应的参数设置以及对应的细节" \
"熟悉且了解参数设置的对应的原因"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', help='which data fold to train on', required=True, type=int, default=1)
    args = parser.parse_args()
    return args


def main():
    global perf
    setup_seed(config.SEED)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED

    model = nn.ModuleDict(
        {'segnet': SegNet(), 'projections': nn.ModuleList([projection(nf) for nf in [32, 64, 128, 256]]),
         'predictions': nn.ModuleList([prediction() for _ in range(4)])})

    if config.TRAIN.PARALLEL:                                   # ONLY CUDA IS SUPPORTED
        devices = config.TRAIN.DEVICES
        model = nn.DataParallel(model, devices).cuda(devices[0])

    else:  # support cuda, mps and ... cpu (really?)
        device = determine_device()
        model = model.to(device)

    optimizer = optim.SGD(model.parameters(), lr=config.TRAIN.LR, weight_decay=config.TRAIN.WEIGHT_DECAY, momentum=0.95,
                          nesterov=True)
    scheduler = PolyScheduler(optimizer, t_total=config.TRAIN.EPOCH)

    # deep supervision weights, normalize sum to 1
    criterion = DiceCELoss()

    #############################################################################################################################

    trainloader = get_trainloader_one_fold(config.TESTDATASET.datasetroot_1, config.TESTDATASET.datasetroot_2)
    train_generator = get_train_generator(trainloader, scales=(1., 0.5, 0.25, 0.125))   ##数据增强对应的结果
    # validation dataset
    # 暂时使用对应的代替
    validset = get_validset_one_fold(config.TESTDATASET.datasetroot_1, config.TESTDATASET.datasetroot_2)

    best_model = False
    best_perf = 0.0
    logger = create_logger('log', 'train.log')
    ## define the training pipline for the whole traing workflow
    for epoch in range(config.TRAIN.EPOCH):
        logger.info('learning rate is {}'.format(optimizer))

        train(model, train_generator, optimizer, criterion, logger, config, epoch)
        ##在步骤之外要更新这个
        scheduler.step()
        #running validation at every epoch in time comsuming

        if epoch % config.VALIDATIOnN_INTERVAL == 0:
            perf = inference(model['segnet'], validset, logger, config)


        if perf > best_perf:
            best_perf = perf
            best_model = True
        else:
            best_model = False

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'perf': perf,
            'optimizer': optimizer.state_dict()
        }, best_model, config.OUTPUT_DIR, filename='checkpoint.pth')



if __name__ == '__main__':
    from torchsummary import summary

    #args = parse_args()
    main()
