"""
Dataset utilities for FRATTVAE.

Handles the full preprocessing pipeline: SMILES → BRICS fragment trees → ListDataset.
Results are cached to disk so the expensive decomposition only runs once per dataset/split.
"""
import os
import pickle
import hashlib
import logging
import warnings
warnings.simplefilter('ignore')

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from joblib import Parallel, delayed
from tqdm import tqdm

from rdkit import Chem

from .utils.preprocess import parallelMolsToBRICSfragments, smiles2mol, SmilesToMorganFingetPrints
from .utils.tree import get_tree_features

logger = logging.getLogger(__name__)


class ListDataset(Dataset):
    """Holds per-molecule (frag_indices, positions, prop) tensors of variable length."""

    def __init__(self, frag_indices: list, positions: list, prop: torch.Tensor) -> None:
        super().__init__()
        self.frag_indices = frag_indices
        self.positions = positions
        self.prop = prop

    def __len__(self):
        return len(self.frag_indices)

    def __getitem__(self, index):
        return self.frag_indices[index], self.positions[index], self.prop[index]


def collate_pad_fn(batch):
    """Collate variable-length fragment sequences into padded tensors."""
    frag_indices, positions, props = zip(*batch)
    frag_indices = pad_sequence(frag_indices, batch_first=True, padding_value=0)
    positions = pad_sequence(positions, batch_first=True, padding_value=0)
    props = torch.stack(props)
    return frag_indices, positions, props


def _smiles_list_hash(smiles_list: list) -> str:
    """Stable hash for a list of SMILES strings (used for cache filenames)."""
    h = hashlib.md5("|".join(smiles_list).encode()).hexdigest()[:12]
    return h


def build_frattvae_dataset(
    smiles_list: list,
    cache_dir: str,
    split_name: str = 'train',
    # Decomposition hyperparameters
    max_nfrags: int = 30,
    max_depth: int = 4,
    max_degree: int = 4,
    min_frag_size: int = 1,
    n_bits: int = 2048,
    radius: int = 2,
    use_chiral: bool = False,
    n_jobs: int = 1,
):
    """
    Build (or load cached) FRATTVAE dataset from a list of SMILES.

    Returns a dict with:
        dataset       : ListDataset of (frag_indices, positions, prop)
        uni_fragments : list[str]  – vocabulary of unique fragment SMILES
        frag_ecfps    : FloatTensor (num_frags, n_bits)
        ndummys       : LongTensor  (num_frags,)  – degree of each fragment
        freq_label    : FloatTensor (num_frags,)  – fragment frequency (for loss weighting)
    """
    os.makedirs(cache_dir, exist_ok=True)

    # Build a deterministic cache key from the data and hyperparams
    smiles_hash = _smiles_list_hash(smiles_list)
    cache_key = (f"{split_name}_{smiles_hash}_nf{max_nfrags}"
                 f"_d{max_depth}_w{max_degree}_nb{n_bits}_r{radius}")
    cache_path = os.path.join(cache_dir, f"frattvae_{cache_key}.pkl")

    if os.path.exists(cache_path):
        logger.info(f"Loading cached FRATTVAE dataset from {cache_path}")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)

    logger.info(f"Preprocessing {len(smiles_list)} molecules for FRATTVAE ({split_name})...")

    # Convert SMILES to RDKit molecules
    _mol_gen = Parallel(n_jobs=n_jobs, return_as='generator')(
        delayed(smiles2mol)(s) for s in smiles_list
    )
    mols = list(tqdm(_mol_gen, total=len(smiles_list), desc="Parsing SMILES", leave=True))

    # BRICS decomposition
    logger.info("Running BRICS decomposition...")
    fragments_list, bondtypes_list, bondMapNums_list, recon_flag, uni_fragments, freq_list = \
        parallelMolsToBRICSfragments(
            mols,
            useChiral=use_chiral,
            minFragSize=min_frag_size,
            maxFragNums=max_nfrags,
            maxDegree=max_degree,
            n_jobs=n_jobs,
        )

    recon_flag = np.array(recon_flag)
    valid_mask = recon_flag > 0
    logger.info(f"Reconstructable: {valid_mask.sum()}/{len(recon_flag)} "
                f"({valid_mask.mean()*100:.1f}%)")

    # Filter to successfully reconstructable molecules
    fragments_list = [f for f, ok in zip(fragments_list, valid_mask) if ok]
    bondtypes_list = [b for b, ok in zip(bondtypes_list, valid_mask) if ok]
    bondMapNums_list = [m for m, ok in zip(bondMapNums_list, valid_mask) if ok]

    # Compute fragment ECFPs  (index 0 = padding, zero vector)
    logger.info(f"Computing ECFPs for {len(uni_fragments)} unique fragments...")
    frag_ecfps_list = SmilesToMorganFingetPrints(
        uni_fragments[1:],   # skip padding token at index 0
        n_bits=n_bits,
        radius=radius,
        ignore_dummy=True,
        useChiral=use_chiral,
        n_jobs=n_jobs,
    )
    frag_ecfps_np = np.vstack([np.zeros((1, n_bits)), np.array(frag_ecfps_list, dtype=np.float32)])
    frag_ecfps = torch.from_numpy(frag_ecfps_np).float()

    # ndummys: number of dummy atoms (*) equals the fragment's degree in the tree
    ndummys = torch.tensor([s.count('*') for s in uni_fragments], dtype=torch.long)

    freq_label = torch.tensor(freq_list, dtype=torch.float32)

    # Build tree features for each molecule
    logger.info("Building fragment tree features...")
    dummy_ecfps = torch.zeros(len(uni_fragments), 1).float()   # shape placeholder; only indices matter
    tree_features = Parallel(n_jobs=n_jobs)(
        delayed(get_tree_features)(f, dummy_ecfps, b, m, max_depth, max_degree, False)
        for f, b, m in tqdm(zip(fragments_list, bondtypes_list, bondMapNums_list),
                            total=len(fragments_list), desc="Building trees")
    )
    # tree.py logs individual depth/width overflows at DEBUG level
    frag_indices_list, _, positions_list = zip(*tree_features)

    # Convert to tensors
    prop = torch.zeros(len(frag_indices_list), 1)   # placeholder; no property prediction
    dataset = ListDataset(list(frag_indices_list), list(positions_list), prop)

    result = {
        'dataset': dataset,
        'uni_fragments': uni_fragments,
        'frag_ecfps': frag_ecfps,
        'ndummys': ndummys,
        'freq_label': freq_label,
    }

    with open(cache_path, 'wb') as f:
        pickle.dump(result, f)
    logger.info(f"Cached FRATTVAE dataset to {cache_path}")

    return result
