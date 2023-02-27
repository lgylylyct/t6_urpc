# -*- coding: utf-8 -*-
from yacs.config import CfgNode as CN

"""这是一个系统配置参数的管理工具，需要创建CN这个容器来装载我们的参数，且这个容器可以嵌套"""

config = CN()              #
config.NUM_WORKERS = 0
config.PRINT_FREQ = 10     ##每次十次迭代就存储一次对应的结果
config.OUTPUT_DIR = r'E:\EBV\NPCICdataset\Patient_Image\test_intro_vae_result'
config.SEED = 11
config.DEVICE = 'cuda'
config.num = 16


config.CUDNN = CN()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.DATASET = CN()
config.DATASET.ROOT = r'E:\EBV\NPCICdataset\Patient_Image\test_nii'
config.DATASET.MOD = "T1C.nii.gz"
config.DATASET.patch_shuffle = False
config.DATASET.subject_shuffle = False


config.TRAIN = CN()
config.TRAIN.BATCH_SIZE = 16
config.TRAIN.PATCH_SIZE = [192, 192]
config.TRAIN.QUEUE_LENGTH = 3840
config.TRAIN.SAMPLES_PER_VOLUME = 1
config.TRAIN.EPOCH = 1000
config.TRAIN.ROOT = r'E:\EBV\NPCICdataset\Patient_Image\train_intro_vae_result\002_spilt_mask_nii'
config.MOD = 'without_tumor.nii'

config.LOG = CN()
config.LOG.OUT_DIR = r'E:\SegExp\t2_use_the_simsaim\Log'
config.LOGFILE = 'train.log'


## the par of the training
config.TRAIN.learing_rate_en = 1e-4
config.TRAIN.learing_rate_de = 5e-4


## the par for saving the model
config.SAVE = CN()
config.SAVE.ENC = 50
config.SAVE.DEC = 50


#
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