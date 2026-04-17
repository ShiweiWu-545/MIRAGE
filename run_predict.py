import argparse
import copy
import io
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from Bio import BiopythonWarning
from Bio.PDB import PDBIO, PDBParser, Select, Selection
from Bio.PDB.Polypeptide import is_aa, three_to_index
from Bio.PDB.SASA import ShrakeRupley

sys.path.append(os.path.dirname(__file__))

from models.predictor import DDGPredictor


ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4
PREDICT_LENGTH = 128
REPO_ROOT = Path(__file__).resolve().parent

ONE_TO_THREE = {
    "A": "ALA",
    "R": "ARG",
    "N": "ASN",
    "D": "ASP",
    "C": "CYS",
    "Q": "GLN",
    "E": "GLU",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "L": "LEU",
    "K": "LYS",
    "M": "MET",
    "F": "PHE",
    "P": "PRO",
    "S": "SER",
    "T": "THR",
    "W": "TRP",
    "Y": "TYR",
    "V": "VAL",
    "-": "-",
}

THREE_TO_ONE = {v: k for k, v in ONE_TO_THREE.items()}

NON_STANDARD_SUBSTITUTIONS = {
    "2AS": "ASP",
    "3AH": "HIS",
    "5HP": "GLU",
    "ACL": "ARG",
    "AGM": "ARG",
    "AIB": "ALA",
    "ALM": "ALA",
    "ALO": "THR",
    "ALY": "LYS",
    "ARM": "ARG",
    "ASA": "ASP",
    "ASB": "ASP",
    "ASK": "ASP",
    "ASL": "ASP",
    "ASQ": "ASP",
    "AYA": "ALA",
    "BCS": "CYS",
    "BHD": "ASP",
    "BMT": "THR",
    "BNN": "ALA",
    "BUC": "CYS",
    "BUG": "LEU",
    "C5C": "CYS",
    "C6C": "CYS",
    "CAS": "CYS",
    "CCS": "CYS",
    "CEA": "CYS",
    "CGU": "GLU",
    "CHG": "ALA",
    "CLE": "LEU",
    "CME": "CYS",
    "CSD": "ALA",
    "CSO": "CYS",
    "CSP": "CYS",
    "CSS": "CYS",
    "CSW": "CYS",
    "CSX": "CYS",
    "CXM": "MET",
    "CY1": "CYS",
    "CY3": "CYS",
    "CYG": "CYS",
    "CYM": "CYS",
    "CYQ": "CYS",
    "DAH": "PHE",
    "DAL": "ALA",
    "DAR": "ARG",
    "DAS": "ASP",
    "DCY": "CYS",
    "DGL": "GLU",
    "DGN": "GLN",
    "DHA": "ALA",
    "DHI": "HIS",
    "DIL": "ILE",
    "DIV": "VAL",
    "DLE": "LEU",
    "DLY": "LYS",
    "DNP": "ALA",
    "DPN": "PHE",
    "DPR": "PRO",
    "DSN": "SER",
    "DSP": "ASP",
    "DTH": "THR",
    "DTR": "TRP",
    "DTY": "TYR",
    "DVA": "VAL",
    "EFC": "CYS",
    "FLA": "ALA",
    "FME": "MET",
    "GGL": "GLU",
    "GL3": "GLY",
    "GLZ": "GLY",
    "GMA": "GLU",
    "GSC": "GLY",
    "HAC": "ALA",
    "HAR": "ARG",
    "HIC": "HIS",
    "HIP": "HIS",
    "HMR": "ARG",
    "HPQ": "PHE",
    "HTR": "TRP",
    "HYP": "PRO",
    "IAS": "ASP",
    "IIL": "ILE",
    "IYR": "TYR",
    "KCX": "LYS",
    "LLP": "LYS",
    "LLY": "LYS",
    "LTR": "TRP",
    "LYM": "LYS",
    "LYZ": "LYS",
    "MAA": "ALA",
    "MEN": "ASN",
    "MHS": "HIS",
    "MIS": "SER",
    "MLE": "LEU",
    "MPQ": "GLY",
    "MSA": "GLY",
    "MSE": "MET",
    "MVA": "VAL",
    "NEM": "HIS",
    "NEP": "HIS",
    "NLE": "LEU",
    "NLN": "LEU",
    "NLP": "LEU",
    "NMC": "GLY",
    "OAS": "SER",
    "OCS": "CYS",
    "OMT": "MET",
    "PAQ": "TYR",
    "PCA": "GLU",
    "PEC": "CYS",
    "PHI": "PHE",
    "PHL": "PHE",
    "PR3": "CYS",
    "PRR": "ALA",
    "PTR": "TYR",
    "PYX": "CYS",
    "SAC": "SER",
    "SAR": "GLY",
    "SCH": "CYS",
    "SCS": "CYS",
    "SCY": "CYS",
    "SEL": "SER",
    "SEP": "SER",
    "SET": "SER",
    "SHC": "CYS",
    "SHR": "LYS",
    "SMC": "CYS",
    "SOC": "CYS",
    "STY": "TYR",
    "SVA": "SER",
    "TIH": "ALA",
    "TPL": "TRP",
    "TPO": "THR",
    "TPQ": "ALA",
    "TRG": "LYS",
    "TRO": "TRP",
    "TYB": "TYR",
    "TYI": "TYR",
    "TYQ": "TYR",
    "TYS": "TYR",
    "TYY": "TYR",
}

RESIDUE_SIDECHAIN_POSTFIXES = {
    "A": ["B"],
    "R": ["B", "G", "D", "E", "Z", "H1", "H2"],
    "N": ["B", "G", "D1", "D2"],
    "D": ["B", "G", "D1", "D2"],
    "C": ["B", "G"],
    "E": ["B", "G", "D", "E1", "E2"],
    "Q": ["B", "G", "D", "E1", "E2"],
    "G": [],
    "H": ["B", "G", "D1", "D2", "E1", "E2"],
    "I": ["B", "G1", "G2", "D1"],
    "L": ["B", "G", "D1", "D2"],
    "K": ["B", "G", "D", "E", "Z"],
    "M": ["B", "G", "D", "E"],
    "F": ["B", "G", "D1", "D2", "E1", "E2", "Z"],
    "P": ["B", "G", "D"],
    "S": ["B", "G"],
    "T": ["B", "G1", "G2"],
    "W": ["B", "G", "D1", "D2", "E1", "E2", "E3", "Z2", "Z3", "H2"],
    "Y": ["B", "G", "D1", "D2", "E1", "E2", "Z", "H"],
    "V": ["B", "G1", "G2"],
}

MAX_ASA = {
    "ALA": 113.0,
    "ARG": 241.0,
    "ASN": 158.0,
    "ASP": 151.0,
    "CYS": 140.0,
    "GLU": 183.0,
    "GLN": 189.0,
    "GLY": 85.0,
    "HIS": 194.0,
    "ILE": 182.0,
    "LEU": 180.0,
    "LYS": 211.0,
    "MET": 204.0,
    "PHE": 218.0,
    "PRO": 143.0,
    "SER": 122.0,
    "THR": 146.0,
    "TRP": 259.0,
    "TYR": 229.0,
    "VAL": 160.0,
}


def choose_device(device_arg):
    if device_arg:
        device = torch.device(device_arg)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA device, but torch.cuda.is_available() is False.")
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_predict_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_pickle_checkpoint(path, device):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {path}")

    original_loader = torch.storage._load_from_bytes

    def load_storage_on_device(buffer):
        return torch.load(io.BytesIO(buffer), map_location=device)

    torch.storage._load_from_bytes = load_storage_on_device
    try:
        with path.open("rb") as handle:
            checkpoint = pickle.load(handle)
    finally:
        torch.storage._load_from_bytes = original_loader

    if isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        state_dict, model_config = checkpoint
    elif isinstance(checkpoint, dict):
        state_dict, model_config = checkpoint, None
    else:
        raise ValueError(
            "Unsupported checkpoint format. Expected (state_dict, config) or a state_dict."
        )
    return state_dict, model_config


def strip_compile_prefix(state_dict):
    if not state_dict:
        return state_dict
    first_key = next(iter(state_dict))
    if first_key.split(".", 1)[0] != "_orig_mod":
        return state_dict
    return {key.split(".", 1)[1]: value for key, value in state_dict.items()}


def prepare_predict_config(model_config, fallback_config, device):
    cfg = copy.deepcopy(model_config if model_config is not None else fallback_config)
    cfg.model.device = str(device)
    cfg.model.pool = False
    return cfg


def normalize_resname(resname):
    return NON_STANDARD_SUBSTITUTIONS.get(resname, resname)


def is_supported_amino_acid(resname):
    return is_aa(normalize_resname(resname), standard=True)


def amino_acid_index(resname):
    return three_to_index(normalize_resname(resname))


def atom_postfix(atom):
    name = atom.get_name()
    if name in ("N", "CA", "C", "O"):
        return name
    if name[-1].isnumeric():
        return name[-2:]
    return name[-1:]


def residue_atom14_positions(residue):
    resname = normalize_resname(residue.get_resname())
    one_letter = THREE_TO_ONE[resname]
    atom_order = ["N", "CA", "C", "O"] + RESIDUE_SIDECHAIN_POSTFIXES[one_letter]

    suffix_to_atom = {}
    for atom in residue.get_atoms():
        atom_name = atom.fullname.strip()
        if re.match(r"^\d+H.*$", atom_name) or re.match(r"^H.*$", atom_name):
            continue
        suffix_to_atom[atom_postfix(atom)] = atom

    pos14 = np.full((14, 3), np.inf, dtype=np.float32)
    for atom_index, suffix in enumerate(atom_order):
        if suffix in suffix_to_atom:
            pos14[atom_index] = suffix_to_atom[suffix].get_coord().astype(np.float32)
    return pos14


def residue_key(chain, residue):
    return (
        chain.get_id(),
        residue.get_id()[0],
        int(residue.get_id()[1]),
        residue.get_id()[2],
        normalize_resname(residue.get_resname()),
    )


class ChainSelect(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain):
        return chain.get_id() in self.chain_ids


def write_selected_chains(pdb_path, chain_ids, output_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predict", str(pdb_path))
    io_obj = PDBIO()
    io_obj.set_structure(structure)
    io_obj.save(str(output_path), ChainSelect(chain_ids))


def compute_residue_sasa(input_pdb, output_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predict", str(input_pdb))
    sr = ShrakeRupley()
    sr.compute(structure, level="R")

    for model in structure:
        for chain in model:
            for residue in chain:
                residue_sasa = round(getattr(residue, "sasa", 0.0), 2)
                for atom in residue:
                    atom.set_bfactor(residue_sasa)

    io_obj = PDBIO()
    io_obj.set_structure(structure)
    io_obj.save(str(output_pdb))
    return output_pdb


def sasa_rows(asa_pdb):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predict", str(asa_pdb))
    rows = []
    pos = 1
    for chain in structure[0]:
        for residue in chain:
            resname = residue.get_resname()
            if not is_supported_amino_acid(resname):
                continue
            if not residue.has_id("CA"):
                continue
            norm_resname = normalize_resname(resname)
            rows.append(
                {
                    "key": residue_key(chain, residue),
                    "pos": pos,
                    "res": norm_resname,
                    "chain": chain.get_id(),
                    "asa": residue["CA"].bfactor,
                }
            )
            pos += 1
    return rows


def determine_region_group(rasa_complex, rasa_monomer):
    delta_rasa = rasa_monomer - rasa_complex
    if delta_rasa == 0:
        return "SUR" if rasa_complex > 0.25 else "INT"
    if delta_rasa > 0:
        if rasa_monomer < 0.25:
            return "SUP"
        if rasa_complex > 0.25:
            return "RIM"
        return "COR"
    raise ValueError("Complex rASA is larger than monomer rASA; region calculation failed.")


def write_region_from_sasa(complex_asa, protein_a_asa, protein_b_asa, region_path):
    complex_rows = sasa_rows(complex_asa)
    protein_rows = {}
    for protein_label, asa_path in ((0, protein_a_asa), (1, protein_b_asa)):
        for row in sasa_rows(asa_path):
            protein_rows[row["key"]] = {"asa_m": row["asa"], "protein": protein_label}

    output_rows = []
    missing = []
    for row in complex_rows:
        if row["key"] not in protein_rows:
            missing.append(f"{row['chain']}:{row['res']}:{row['pos']}")
            continue

        max_asa = MAX_ASA[row["res"]]
        asa_complex = row["asa"]
        asa_monomer = protein_rows[row["key"]]["asa_m"]
        rasa_complex = asa_complex / max_asa
        rasa_monomer = asa_monomer / max_asa
        output_rows.append(
            {
                "pos": row["pos"],
                "res": row["res"],
                "chain": row["chain"],
                "region": determine_region_group(rasa_complex, rasa_monomer),
                "protein": protein_rows[row["key"]]["protein"],
                "rasa_c": rasa_complex,
                "rasa_m": rasa_monomer,
                "d_rasa": rasa_monomer - rasa_complex,
                "max_asa": max_asa,
                "asa": asa_complex,
                "asa_m": asa_monomer,
            }
        )

    if missing:
        raise ValueError(
            "Some complex residues were not found in monomer SASA results: "
            + ", ".join(missing[:10])
        )

    columns = [
        "pos",
        "res",
        "chain",
        "region",
        "protein",
        "rasa_c",
        "rasa_m",
        "d_rasa",
        "max_asa",
        "asa",
        "asa_m",
    ]
    pd.DataFrame(output_rows, columns=columns).to_csv(region_path, index=False)


def generate_region_file(pdb_path, protein_a_chains, protein_b_chains, region_path):
    all_chains = protein_a_chains + protein_b_chains
    if not all_chains:
        raise ValueError("proteinA/proteinB did not provide any chain id.")

    with tempfile.TemporaryDirectory(prefix=f"{pdb_path.stem}_region_", dir=str(pdb_path.parent)) as tmp_dir:
        tmp_dir = Path(tmp_dir)
        complex_pdb = tmp_dir / f"{pdb_path.stem}_complex.pdb"
        protein_a_pdb = tmp_dir / f"{pdb_path.stem}_proteinA.pdb"
        protein_b_pdb = tmp_dir / f"{pdb_path.stem}_proteinB.pdb"
        complex_asa = tmp_dir / f"{pdb_path.stem}_complex_asa.pdb"
        protein_a_asa = tmp_dir / f"{pdb_path.stem}_proteinA_asa.pdb"
        protein_b_asa = tmp_dir / f"{pdb_path.stem}_proteinB_asa.pdb"

        write_selected_chains(pdb_path, all_chains, complex_pdb)
        write_selected_chains(pdb_path, protein_a_chains, protein_a_pdb)
        write_selected_chains(pdb_path, protein_b_chains, protein_b_pdb)

        compute_residue_sasa(complex_pdb, complex_asa)
        compute_residue_sasa(protein_a_pdb, protein_a_asa)
        compute_residue_sasa(protein_b_pdb, protein_b_asa)
        write_region_from_sasa(complex_asa, protein_a_asa, protein_b_asa, region_path)


def generate_prottrans_file(pdb_path, h5_path):
    prottrans_dir = REPO_ROOT / "prottrans"
    pdb_to_fasta_script = prottrans_dir / "pdb_tofasta.py"
    embedder_script = prottrans_dir / "prott5_embedder.py"
    fasta_path = pdb_path.with_suffix(".fasta")

    if not pdb_to_fasta_script.exists():
        raise FileNotFoundError(f"Missing pdb_tofasta.py: {pdb_to_fasta_script}")
    if not embedder_script.exists():
        raise FileNotFoundError(f"Missing prott5_embedder.py: {embedder_script}")

    with fasta_path.open("w") as fasta_handle:
        subprocess.run(
            [sys.executable, str(pdb_to_fasta_script), "-multi", str(pdb_path)],
            cwd=str(prottrans_dir),
            stdout=fasta_handle,
            check=True,
        )

    env = os.environ.copy()
    env.setdefault("TRANSFORMERS_OFFLINE", "1")
    subprocess.run(
        [sys.executable, str(embedder_script), "--input", str(fasta_path), "--output", str(h5_path)],
        cwd=str(prottrans_dir),
        env=env,
        check=True,
    )


def ensure_feature_files(pdb_path, protein_a_chains, protein_b_chains):
    region_path = pdb_path.with_suffix(".region")
    h5_path = pdb_path.with_suffix(".h5")

    if region_path.exists():
        print("Loaded region file.")
    else:
        print(f"Region file not found. Generating: {region_path}")
        generate_region_file(pdb_path, protein_a_chains, protein_b_chains, region_path)
        if not region_path.exists():
            raise FileNotFoundError(f"Region generation did not create: {region_path}")

    if h5_path.exists():
        print("Loaded ProtTrans h5.")
    else:
        print(f"ProtTrans h5 not found. Generating: {h5_path}")
        generate_prottrans_file(pdb_path, h5_path)
        if not h5_path.exists():
            raise FileNotFoundError(f"ProtTrans generation did not create: {h5_path}")

    return region_path, h5_path


def parse_chain_argument(value):
    if not value:
        return []
    return [item for item in value.replace(",", "").replace(" ", "")]


def parse_pdb_atom14(pdb_path, chains=None):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predict", str(pdb_path))
    model = structure[0]
    all_chains = Selection.unfold_entities(model, "C")

    if chains is not None:
        chain_set = set(chains)
        all_chains = [chain for chain in all_chains if chain.get_id() in chain_set]
        missing = [chain_id for chain_id in chains if chain_id not in {c.get_id() for c in all_chains}]
        if missing:
            raise ValueError(f"PDB is missing requested chain(s): {','.join(missing)}")

    aa, resseq, seq = [], [], []
    pos14, pos14_mask = [], []
    chain_id, chain_seq = [], []

    for chain_index, chain in enumerate(all_chains):
        seq_this = 0
        for residue in chain:
            resname = residue.get_resname()
            if not is_supported_amino_acid(resname):
                continue
            if not (residue.has_id("CA") and residue.has_id("C") and residue.has_id("N")):
                continue

            chain_id.append(chain.get_id())
            chain_seq.append(chain_index + 1)
            aa.append(amino_acid_index(resname))

            pos14_this = residue_atom14_positions(residue)
            finite_mask = np.isfinite(pos14_this)
            pos14.append(np.nan_to_num(pos14_this, posinf=99999.0))
            pos14_mask.append(finite_mask)

            resseq_this = int(residue.get_id()[1])
            if seq_this == 0:
                seq_this = 1
            else:
                delta = resseq_this - resseq[-1]
                seq_this += 1 if delta == 0 else delta
            resseq.append(resseq_this)
            seq.append(seq_this)

    if not aa:
        raise ValueError("No supported amino-acid residues with N/CA/C atoms were parsed from the PDB.")

    return {
        "aa": np.asarray(aa, dtype=np.int64),
        "resseq": np.asarray(resseq, dtype=np.int64),
        "seq": np.asarray(seq, dtype=np.int64),
        "chain_id": np.asarray(chain_id),
        "chain_seq": np.asarray(chain_seq, dtype=np.int64),
        "pos14": np.stack(pos14).astype(np.float32),
        "pos14_mask": np.stack(pos14_mask).astype(bool),
    }


def load_region_indices(region_path, max_length=PREDICT_LENGTH):
    region_path = Path(region_path)
    if not region_path.exists():
        raise FileNotFoundError(f"Region file not found: {region_path}")
    region = pd.read_csv(region_path)
    required = {"pos", "region"}
    missing = required - set(region.columns)
    if missing:
        raise ValueError(f"Region file is missing required column(s): {', '.join(sorted(missing))}")

    ordered_positions = []
    for region_name in ("COR", "SUP", "RIM"):
        ordered_positions.extend(region.loc[region["region"] == region_name, "pos"].astype(int).tolist())
    if not ordered_positions:
        raise ValueError("No COR/SUP/RIM residues were found in the region file.")
    return np.asarray([pos - 1 for pos in ordered_positions[:max_length]], dtype=np.int64)


def select_residue_indices(features, indices):
    if indices.min(initial=0) < 0 or indices.max(initial=0) >= len(features["aa"]):
        raise IndexError(
            f"Region index outside parsed PDB length. PDB residues={len(features['aa'])}, "
            f"max region index={int(indices.max()) + 1}."
        )

    mask = np.zeros(len(features["aa"]), dtype=bool)
    mask[indices] = True
    selected = {}
    for key, value in features.items():
        if isinstance(value, np.ndarray) and value.shape[0] == len(features["aa"]):
            selected[key] = value[mask]
        else:
            selected[key] = value
    return selected


def h5_dataset_key(h5_file, chain_id):
    for prefix in ("XXXX|", "PDB|", ""):
        candidate = f"{prefix}{chain_id}"
        if candidate in h5_file:
            return candidate
    for key in h5_file.keys():
        if key.split("|")[-1] == chain_id:
            return key
    return None


def load_prottrans_embeddings(h5_path, chain_order, expected_length):
    h5_path = Path(h5_path)
    if not h5_path.exists():
        raise FileNotFoundError(f"ProtTrans h5 file not found: {h5_path}")

    arrays = []
    with h5py.File(h5_path, "r") as handle:
        missing = []
        for chain_id in chain_order:
            key = h5_dataset_key(handle, chain_id)
            if key is None:
                missing.append(chain_id)
            else:
                arrays.append(handle[key][:])
        if missing:
            available = ", ".join(handle.keys())
            raise KeyError(
                f"ProtTrans h5 is missing chain(s): {','.join(missing)}. "
                f"Available keys: {available}"
            )

    prottrans = np.concatenate(arrays, axis=0).astype(np.float32)
    if prottrans.shape[0] != expected_length:
        raise ValueError(
            f"ProtTrans length ({prottrans.shape[0]}) does not match parsed PDB residues "
            f"({expected_length}). Check chain order and h5 file."
        )
    if prottrans.shape[1] != 1024:
        raise ValueError(f"Expected ProtTrans embedding dim 1024, got {prottrans.shape[1]}.")
    return prottrans


def pad_array(value, length=PREDICT_LENGTH, pad_value=0):
    if value.shape[0] > length:
        value = value[:length]
    if value.shape[0] == length:
        return value
    pad_shape = (length - value.shape[0],) + value.shape[1:]
    pad = np.full(pad_shape, pad_value, dtype=value.dtype)
    return np.concatenate([value, pad], axis=0)


def build_model_batch(pdb_path, region_path, h5_path, chains=None):
    parsed = parse_pdb_atom14(pdb_path, chains=chains)
    chain_order = list(dict.fromkeys(parsed["chain_id"].tolist()))
    prottrans = load_prottrans_embeddings(h5_path, chain_order, expected_length=len(parsed["aa"]))
    parsed["res_prottrans"] = prottrans

    selected_indices = load_region_indices(region_path)
    selected = select_residue_indices(parsed, selected_indices)
    selected_count = len(selected["aa"])
    if selected_count == 0:
        raise ValueError("No residues remained after applying the region selection.")

    tensor_batch = {
        "pos14": torch.from_numpy(pad_array(selected["pos14"], pad_value=999.0)).unsqueeze(0).float(),
        "pos14_mask": torch.from_numpy(pad_array(selected["pos14_mask"], pad_value=False)).unsqueeze(0).bool(),
        "aa": torch.from_numpy(pad_array(selected["aa"], pad_value=20)).unsqueeze(0).long(),
        "seq": torch.from_numpy(pad_array(selected["seq"], pad_value=0)).unsqueeze(0).long(),
        "chain_seq": torch.from_numpy(pad_array(selected["chain_seq"], pad_value=0)).unsqueeze(0).long(),
        "res_prottrans": torch.from_numpy(
            pad_array(selected["res_prottrans"], pad_value=0.0)
        ).unsqueeze(0).float(),
    }

    meta = {
        "parsed_residues": len(parsed["aa"]),
        "selected_residues": selected_count,
        "chains": chain_order,
        "pdb_path": str(pdb_path),
        "region_path": str(region_path),
        "h5_path": str(h5_path),
    }
    return tensor_batch, meta


def recursive_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    if isinstance(obj, dict):
        return {key: recursive_to_device(value, device) for key, value in obj.items()}
    return obj


def predict_dg(model_path, pdb_path, region_path, h5_path, chains, device_arg=None):
    from data.my_config import config as fallback_config

    device = choose_device(device_arg)
    state_dict, checkpoint_config = load_pickle_checkpoint(model_path, device)
    cfg = prepare_predict_config(checkpoint_config, fallback_config, device)
    if cfg.model.mission != "DG":
        raise ValueError(f"Loaded config mission is {cfg.model.mission}, but DG prediction is required.")
    cfg.model.mission = "DG"
    if cfg.feature.Atom[1] != 14:
        raise ValueError(
            f"This predictor currently expects Atom[1] == 14, got {cfg.feature.Atom}."
        )
    if not cfg.feature.res_prottrans:
        raise ValueError("The loaded DG model expects config.feature.res_prottrans=True.")

    set_predict_seed(cfg.train.seed)
    batch, meta = build_model_batch(pdb_path, region_path, h5_path, chains=chains)
    batch = recursive_to_device(batch, device)

    model = DDGPredictor(cfg).to(device)
    model.load_state_dict(strip_compile_prefix(state_dict), strict=True)
    model.eval()
    with torch.no_grad():
        prediction = model(batch, None)
    return float(prediction.detach().cpu().item()), meta


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Predict DG for one protein complex PDB using a trained MIRAGE DG checkpoint."
    )
    parser.add_argument("pdb_path", help="Path to the complex PDB file.")
    parser.add_argument("proteinA", help="Chain id(s) for protein A, e.g. A or HL.")
    parser.add_argument("proteinB", help="Chain id(s) for protein B, e.g. D or A.")
    parser.add_argument(
        "--model_path",
        default=str(REPO_ROOT / "data/model.pt"),
        help="Path to a saved checkpoint. The training code saves (state_dict, config).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="cpu, cuda, cuda:0, etc. Defaults to cuda when available, otherwise cpu.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    pdb_path = Path(args.pdb_path).expanduser().resolve()
    protein_a_chains = parse_chain_argument(args.proteinA)
    protein_b_chains = parse_chain_argument(args.proteinB)
    region_path, h5_path = ensure_feature_files(pdb_path, protein_a_chains, protein_b_chains)
    chains = protein_a_chains + protein_b_chains

    dg, meta = predict_dg(
        model_path=Path(args.model_path).expanduser().resolve(),
        pdb_path=pdb_path,
        region_path=region_path,
        h5_path=h5_path,
        chains=chains,
        device_arg=args.device,
    )

    print(f"MIRAGE predicted binding affinity (DG) for this complex: {dg:.6f} kcal/mol")
    print("DG definition: DG = -RT ln(KD), where R = 0.001987 kcal/(mol*K), T is Temperature(K), and KD is in mol/L (M).")
    print("Note: this is the positive binding-affinity score used by PPB-Affinity; thermodynamic binding free energy is DeltaG_bind = RT ln(KD) = -DG.")


if __name__ == "__main__":
    import warnings

    warnings.simplefilter("ignore", BiopythonWarning)
    main()
