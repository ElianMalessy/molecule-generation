"""
Molecular property computation and caching for the joint-model training objective.

Three properties are predicted per molecule:
  - plogP  (penalized logP = logP - SA - ring_penalty) — unbounded, good for gradient ascent
  - QED    (Quantitative Estimate of Drug-likeness) — [0, 1], bounded
  - SA     (Synthetic Accessibility Score) — [1, 10], lower is easier to synthesize

All three are Z-score normalised before being used as regression targets so
that MSE gradients are comparable in magnitude.
"""

import os
import torch
import numpy as np
from typing import Optional

from rdkit import Chem
from rdkit.Chem import QED, Descriptors
from rdkit.Contrib.SA_Score import sascorer


# ---------------------------------------------------------------------------
# Core property computation
# ---------------------------------------------------------------------------

def _ring_penalty(mol) -> float:
    """Number of atoms in the largest ring above size 6, else 0."""
    ri = mol.GetRingInfo()
    if not ri.NumRings():
        return 0.0
    max_size = max((len(r) for r in ri.AtomRings()), default=0)
    return max(0.0, max_size - 6.0)


def compute_mol_props(mol) -> Optional[tuple[float, float, float]]:
    """
    Compute (plogP, QED, SA) for an RDKit mol object.
    Returns None if the mol is None or properties cannot be computed.
    """
    if mol is None:
        return None
    try:
        logp = Descriptors.MolLogP(mol)
        sa   = sascorer.calculateScore(mol)      # 1 (easy) – 10 (hard)
        ring = _ring_penalty(mol)
        plogp = logp - sa - ring
        qed   = QED.qed(mol)
        return float(plogp), float(qed), float(sa)
    except Exception:
        return None


def compute_smiles_props(smiles: str) -> Optional[tuple[float, float, float]]:
    """Convenience wrapper for SMILES input."""
    mol = Chem.MolFromSmiles(smiles)
    return compute_mol_props(mol)


# ---------------------------------------------------------------------------
# Reconstruct RDKit mol from a PyG Data object
# ---------------------------------------------------------------------------

_BOND_TYPE_MAP = {
    0: Chem.BondType.SINGLE,
    1: Chem.BondType.DOUBLE,
    2: Chem.BondType.TRIPLE,
    3: Chem.BondType.AROMATIC,
}


def mol_from_data(data, atom_decoder: dict, charge_decoder: Optional[dict] = None):
    """
    Reconstruct an RDKit mol from a stored PyG Data object.

    Assumes:
      - data.x is a (N,) or (N,1) int tensor of 0-indexed atom-type class indices
      - data.edge_attr is (E,) or (E,1) int tensor of 0-indexed bond-type indices
        (0=single, 1=double, 2=triple, 3=aromatic) — already normalised by
        NormalizeZINCBonds for ZINC, or stored directly for MOSES.
    """
    x = data.x.squeeze(-1).numpy()
    edge_index = data.edge_index.numpy()
    edge_attr  = data.edge_attr.squeeze(-1).numpy()

    mol = Chem.RWMol()
    for atom_class in x:
        atomic_num = atom_decoder.get(int(atom_class), 6)
        atom = Chem.Atom(atomic_num)
        if charge_decoder:
            fc = charge_decoder.get(int(atom_class), 0)
            if fc:
                atom.SetFormalCharge(fc)
        mol.AddAtom(atom)

    added = set()
    for i in range(edge_index.shape[1]):
        u, v = int(edge_index[0, i]), int(edge_index[1, i])
        if u < v and (u, v) not in added:
            bt = _BOND_TYPE_MAP.get(int(edge_attr[i]), Chem.BondType.SINGLE)
            mol.AddBond(u, v, bt)
            added.add((u, v))

    try:
        Chem.SanitizeMol(mol)
        return mol
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Build / load a cached property tensor for a whole dataset split
# ---------------------------------------------------------------------------

def build_props_cache(
    dataset,
    atom_decoder: dict,
    charge_decoder: Optional[dict],
    cache_path: str,
) -> torch.Tensor:
    """
    Return a float32 tensor of shape (N, 3) containing [plogP, QED, SA] for
    every molecule in `dataset` (in index order).

    Results are loaded from `cache_path` if it already exists, otherwise
    computed (which takes a few minutes for ZINC/MOSES full splits) and saved.
    """
    if os.path.exists(cache_path):
        return torch.load(cache_path, weights_only=True)

    print(f"[properties] Computing property cache → {cache_path}")
    props = []
    for i, data in enumerate(dataset):
        mol = mol_from_data(data, atom_decoder, charge_decoder)
        p = compute_mol_props(mol)
        if p is None:
            # Fallback: median-like defaults so training doesn't crash on failures
            p = (0.0, 0.5, 5.0)
        props.append(p)
        if (i + 1) % 10000 == 0:
            print(f"  {i + 1}/{len(dataset)} molecules processed")

    tensor = torch.tensor(props, dtype=torch.float32)
    os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
    torch.save(tensor, cache_path)
    print(f"[properties] Saved {len(props)} property vectors to {cache_path}")
    return tensor


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def compute_normalisation_stats(props: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute per-property mean and std over the training set.
    Returns (mean, std) each of shape (3,).
    Clips std to a minimum of 1e-6 to avoid division by zero.
    """
    mean = props.mean(dim=0)
    std  = props.std(dim=0).clamp(min=1e-6)
    return mean, std


def normalise_props(props: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Z-score normalise a (..., 3) property tensor."""
    return (props - mean.to(props.device)) / std.to(props.device)


def denormalise_props(props_z: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Inverse Z-score transform."""
    return props_z * std.to(props_z.device) + mean.to(props_z.device)


# ---------------------------------------------------------------------------
# γ schedule: delayed ramp for property loss weight
# ---------------------------------------------------------------------------

def prop_gamma(epoch: int, warmup_epochs: int, gamma_max: float,
               ramp_epochs: int = 5) -> float:
    """
    Property loss weight schedule:
      - 0 for the first `warmup_epochs` epochs (let the VAE learn topology first)
      - linearly ramps from 0 → gamma_max over the next `ramp_epochs` epochs
      - holds at gamma_max thereafter

    Args:
        epoch:         Current (1-based) epoch number.
        warmup_epochs: Number of epochs with γ = 0.
        gamma_max:     Final weight value.
        ramp_epochs:   Number of epochs for the linear ramp (default 5).
    """
    if epoch <= warmup_epochs:
        return 0.0
    if ramp_epochs <= 0:
        return gamma_max
    frac = min(1.0, (epoch - warmup_epochs) / ramp_epochs)
    return frac * gamma_max
