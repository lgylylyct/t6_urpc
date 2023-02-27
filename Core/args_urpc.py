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
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import timm
# assert timm.__version__ == "0.3.2"  # version check
import timm.optim.optim_factory as optim_factory

def get_args_parser():
    parser = argparse.ArgumentParser('the args of the uncertainly_pyramid_consistency', add_help=False)
    parser.add_argument('--root_path', type=str,
                        default='../data/ACDC', help='Name of Experiment')
    parser.add_argument('--exp', type=str,
                        default='ACDC/Uncertainty_Rectified_Pyramid_Consistency', help='experiment_name')
    parser.add_argument('--model', type=str,
                        default='unet_urpc', help='model_name')
    parser.add_argument('--max_iterations', type=int,
                        default=30000, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int, default=24,
                        help='batch_size per gpu')
    parser.add_argument('--deterministic', type=int, default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float, default=0.01,
                        help='segmentation network learning rate')
    parser.add_argument('--patch_size', type=list, default=[192, 192],
                        help='patch size of network input')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='output channel of network')

    # label and unlabel
    parser.add_argument('--labeled_bs', type=int, default=12,
                        help='labeled_batch_size per gpu')
    parser.add_argument('--labeled_num', type=int, default=7,
                        help='labeled data')
    # costs
    parser.add_argument('--consistency', type=float,
                        default=0.1, help='consistency')
    parser.add_argument('--consistency_rampup', type=float,
                        default=200.0, help='consistency_rampup')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    return parser