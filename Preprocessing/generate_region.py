import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from run_predict import generate_region_file, parse_chain_argument


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a MIRAGE interface .region file for one complex PDB."
    )
    parser.add_argument("pdb_path", help="Path to the complex PDB file.")
    parser.add_argument("proteinA", help="Chain id(s) for protein A, e.g. A, HL, or H,L.")
    parser.add_argument("proteinB", help="Chain id(s) for protein B, e.g. D, B, or A.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .region path. Defaults to the input PDB path with .region suffix.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    pdb_path = Path(args.pdb_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else pdb_path.with_suffix(".region")

    generate_region_file(
        pdb_path,
        parse_chain_argument(args.proteinA),
        parse_chain_argument(args.proteinB),
        output_path,
    )
    print(f"Wrote region file: {output_path}")


if __name__ == "__main__":
    main()
