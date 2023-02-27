from yacs.config import CfgNode as CN

config = CN()
config.NUM_WORKERS = 1
config.PRINT_FREQ = 10
config.VALIDATION_INTERVAL = 5
config.OUTPUT_DIR = 'experiments'
config.SEED = 12345

config.CUDNN = CN()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

config.DATASET = CN()
#config.DATASET.ROOT = 'DATA/preprocessed/brats19'
config.DATASET.ROOT = r'F:\EBV_dataset\npy_regiser\zhongshan3\113_ToNumpy_T1C'


config.TRAIN = CN()
config.TRAIN.LR = 1e-2
config.TRAIN.WEIGHT_DECAY = 3e-5
config.TRAIN.BATCH_SIZE = 1
config.TRAIN.PATCH_SIZE = [512, 512]
config.TRAIN.NUM_BATCHES = 250
config.TRAIN.EPOCH = 200
config.TRAIN.PARALLEL = False
config.TRAIN.DEVICES = [0]

config.INFERENCE = CN()
config.INFERENCE.BATCH_SIZE = 4
config.INFERENCE.PATCH_SIZE = [192, 160]
config.INFERENCE.PATCH_OVERLAP = [96, 80]


config.TESTDATASET = CN()
config.TESTDATASET.datasetroot_1 = r'E:\NPCICdataset\Patient_Image\seg_test\nnUNet_raw_data\Task001_Npc'
config.TESTDATASET.datasetroot_2 = r'E:\NPCICdataset\Patient_Image\seg_test\nnUNet_process\Task001_Npc'
