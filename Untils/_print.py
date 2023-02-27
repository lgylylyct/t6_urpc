import os
import argparse
import torch
import datetime
import numpy as np
import time
import json
import torch.backends.cudnn as cudnn


def print_tem(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training', add_help=False)

    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--epochs', default=400, type=int)
    return parser


def random_seed_set(args):
    # seed = args.seed + misc.get_rank()  ##misc.get_rank = 0
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True


def gain_running_time():
    start_time = time.time()
    "there is the running code for the "
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def log_json():
    log_stats = 'the infor which you want to record for the result'
    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    print_tem(args)
