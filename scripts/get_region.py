import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from utils.cal_rasa import get_region_ske


def run_get_region(path_wt, path_mut, chains_a, chains_b):
    get_region_ske(path_wt, chains_a, chains_b)
    get_region_ske(path_mut, chains_a, chains_b)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("pdb_path")
    parser.add_argument("chains_a")
    parser.add_argument("chains_b")
    return parser


def main():
    args = build_parser().parse_args()
    get_region_ske(args.pdb_path, args.chains_a, args.chains_b)


if __name__ == "__main__":
    main()
