import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold', help='which data fold to train on', required=True, type=int, default=1)
    args = parser.parse_args()
    return args