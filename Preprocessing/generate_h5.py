import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PROTTRANS_DIR = REPO_ROOT / "prottrans"


def generate_h5(pdb_path, output_path):
    pdb_to_fasta_script = PROTTRANS_DIR / "pdb_tofasta.py"
    embedder_script = PROTTRANS_DIR / "prott5_embedder.py"
    fasta_path = pdb_path.with_suffix(".fasta")

    if not pdb_to_fasta_script.exists():
        raise FileNotFoundError(f"Missing pdb_tofasta.py: {pdb_to_fasta_script}")
    if not embedder_script.exists():
        raise FileNotFoundError(f"Missing prott5_embedder.py: {embedder_script}")

    with fasta_path.open("w") as fasta_handle:
        subprocess.run(
            [sys.executable, str(pdb_to_fasta_script), "-multi", str(pdb_path)],
            stdout=fasta_handle,
            check=True,
        )

    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    subprocess.run(
        [sys.executable, str(embedder_script), "--input", str(fasta_path), "--output", str(output_path)],
        env=env,
        check=True,
    )
    return fasta_path


def build_parser():
    parser = argparse.ArgumentParser(
        description="Generate a MIRAGE ProtTrans .h5 embedding file for one complex PDB."
    )
    parser.add_argument("pdb_path", help="Path to the complex PDB file.")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .h5 path. Defaults to the input PDB path with .h5 suffix.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    pdb_path = Path(args.pdb_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve() if args.output else pdb_path.with_suffix(".h5")

    fasta_path = generate_h5(pdb_path, output_path)
    print(f"Wrote FASTA file: {fasta_path}")
    print(f"Wrote ProtTrans h5 file: {output_path}")


if __name__ == "__main__":
    main()
