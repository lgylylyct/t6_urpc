from Core.config import config
from Untils.utils import save_checkpoint, create_logger, setup_seed
import torch.backends.cudnn as cudnn

def set_seed(config_sed = config.SEED):
    setup_seed(config_sed)
    cudnn.benchmark = config.CUDNN.BENCHMARK
    cudnn.deterministic = config.CUDNN.DETERMINISTIC
    cudnn.enabled = config.CUDNN.ENABLED