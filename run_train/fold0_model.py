import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

LABEL = 'test1_DG_blind'
VALIDATION_FLAG = False
DEBUG = True
CONTINUE_TRAIN = False


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("label", nargs="?", default=LABEL)
    return parser


def main(label=LABEL):
    from data.my_config import config
    from run_train.train_fold import run_training

    return run_training(
        config,
        os.path.basename(__file__),
        validation_flag=VALIDATION_FLAG,
        label=label,
        debug=DEBUG,
        continue_train=CONTINUE_TRAIN,
    )


if __name__ == '__main__':
    args = build_parser().parse_args()
    main(label=args.label)
