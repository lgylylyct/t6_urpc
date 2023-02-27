# -*- coding: utf-8 -*-
from yacs.config import CfgNode as CN

config = CN()
config.NUM_WORKERS = 0
config.PRINT_FREQ = 10     ##每次十次迭代就存储一次对应的结果
config.SEED = 11
config.DEVICE = 'cuda'
config.num = 16
config.MOD = 'without_tumor.nii'
config.OUTPUT_DIR = r'E:\SegExp\t3_synthesis\Result\ModelWeight'   ##  the address of the model which i use to save
#config.OUTPUT_DIR = r'E:\SegExp\t3_synthesis\Result\ModelWeight'
config.alpha = 0.5
config.beta = 0.05
config.eta = 120


config.CUDNN = CN()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.DATASET = CN()
config.DATASET.ROOT = r'E:\EBV\NPCICdataset\Patient_Image\train_intro_vae_result\002_spilt_mask_nii'
#config.DATASET.ROOT ='/public2/yanghening/hanxv_exp/SEGexp/DataSet/002_spilt_mask_nii/'
config.DATASET.patch_shuffle = False
config.DATASET.subject_shuffle = False


config.TRAIN = CN()
config.TRAIN.BATCH_SIZE = 2
config.TRAIN.PATCH_SIZE = [192, 192]
config.TRAIN.QUEUE_LENGTH = 3840
config.TRAIN.SAMPLES_PER_VOLUME = 1
config.TRAIN.EPOCH = 1000

config.LOG = CN()
config.LOG.OUT_DIR = r'E:\SegExp\t4_pretrain_vit_mae\Log'
#config.LOG.OUT_DIR = r'/public2/yanghening/hanxv_exp/SEGexp/t3_synthesis/Log/'

config.LOGFILE = 'train.log'


## the par of the training
config.TRAIN.learing_rate_en = 1e-4
config.TRAIN.learing_rate_de = 1e-4
#

## the par for saving the model
config.SAVE = CN()
config.SAVE.ENC =  50
config.SAVE.DEC = 50

# alpha, beta, eta = 0.5, 0.05, 120
# print('BATCHSIZE:{batch_size}\n'\
#     'ENC_lr:{enc_lr}\n'\
#       'DEC_lr:{dec_lr}\n'\
#       'LOG_DIR:{log_dir}\n'\
#       'DATA_TRAIN_ROOT:{DATA_TRAIN_ROOT}\n'\
#       'alpha:{alpha}\n'\
#       'beta:{beta}\n'\
#       'eta:{eta}\n'\
#       'PATCH_SIZE :{PATCH_SIZE}\n'\
#       .format(batch_size=config.TRAIN.BATCH_SIZE,
#               enc_lr=config.TRAIN.learing_rate_en,
#               dec_lr = config.TRAIN.learing_rate_de,
#               log_dir=config.LOG.OUT_DIR,
#               alpha = alpha,
#               beta = beta,
#               eta = eta,
#               DATA_TRAIN_ROOT = config.TRAIN.ROOT,
#               PATCH_SIZE  = config.TRAIN.PATCH_SIZE))
