# Store datasets specific mappings here to keep logic clean
from rdkit import Chem

MOSES_ATOM_DECODER = {0: 6, 1: 7, 2: 8, 3: 9, 4: 16, 5: 17, 6: 35, 7: 1}

# 1-indexed atom class (data stores 0-indexed; gvae_prepare_batch / ar_collate_fn
# apply +1 before feeding to the model) → atomic number.
# Classes 1-17 correspond to the 17 unique (atomic_num, formal_charge) pairs
# in the ZINC250k (aspuru-guzik/chemical_vae 250k_rndm_zinc_drugs_clean_3.csv).
ZINC_ATOM_DECODER = {
    1:  6,   # C
    2:  7,   # N
    3:  8,   # O
    4:  16,  # S
    5:  9,   # F
    6:  7,   # N+
    7:  17,  # Cl
    8:  8,   # O-
    9:  35,  # Br
    10: 7,   # N-
    11: 53,  # I
    12: 16,  # S-
    13: 15,  # P
    14: 8,   # O+
    15: 16,  # S+
    16: 6,   # C-
    17: 15,  # P+
}

# Non-zero formal charges only; classes absent here have charge 0.
ZINC_CHARGE_DECODER = {
    6:  +1,  # N+
    8:  -1,  # O-
    10: -1,  # N-
    12: -1,  # S-
    14: +1,  # O+
    15: +1,  # S+
    16: -1,  # C-
    17: +1,  # P+
}

# ---------------------------------------------------------------------------
# Valency / bond-order tables (shared by gvae.py and gvae_ar.py)
# ---------------------------------------------------------------------------

# Maximum total bond order allowed per heavy atom type.
# Charged atoms (+/-) are adjusted in the sampler.
MAX_VALENCE = {
    1:  1,   # H
    6:  4,   # C
    7:  3,   # N  (N+ → 4)
    8:  2,   # O  (O- → 1)
    9:  1,   # F
    15: 5,   # P
    16: 6,   # S
    17: 1,   # Cl
    35: 1,   # Br
    53: 1,   # I
}

# Valence units consumed per bond-type index (1-indexed after +1 shift in training).
# Bond index encoding: 0=no-bond, 1=single, 2=aromatic, 3=double, 4=triple.
# Aromatic bonds are counted as 1 valence unit (integer approximation for valency masking).
BOND_ORDER = {0: 0, 1: 1, 2: 1, 3: 2, 4: 3}


def get_rdkit_bond(bond_idx: int) -> Chem.BondType:
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.AROMATIC,
        3: Chem.BondType.DOUBLE,
        4: Chem.BondType.TRIPLE,
    }.get(bond_idx, Chem.BondType.SINGLE)
