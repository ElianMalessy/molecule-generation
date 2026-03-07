# Store datasets specific mappings here to keep logic clean
from rdkit import Chem

MOSES_ATOM_DECODER = {0: 6, 1: 7, 2: 8, 3: 9, 4: 16, 5: 17, 6: 35, 7: 1}

# 0-indexed atom class (matching _ZINC_ATOM_VOCAB in utils.py; data.x stores these
# 0-indexed values directly).  decode_to_smiles and mol_from_data both look up
# cls = atom_idx - 1 (after the +1 shift applied in training) → key range 0-16.
# Classes 0-16 correspond to the 17 unique (atomic_num, formal_charge) pairs
# in the ZINC250k (aspuru-guzik/chemical_vae 250k_rndm_zinc_drugs_clean_3.csv).
ZINC_ATOM_DECODER = {
    0:  6,   # C
    1:  7,   # N
    2:  8,   # O
    3:  16,  # S
    4:  9,   # F
    5:  7,   # N+
    6:  17,  # Cl
    7:  8,   # O-
    8:  35,  # Br
    9:  7,   # N-
    10: 53,  # I
    11: 16,  # S-
    12: 15,  # P
    13: 8,   # O+
    14: 16,  # S+
    15: 6,   # C-
    16: 15,  # P+
}

# Non-zero formal charges only; classes absent here have charge 0.
# Keys are 0-indexed (matching ZINC_ATOM_DECODER above).
ZINC_CHARGE_DECODER = {
    5:  +1,  # N+
    7:  -1,  # O-
    9:  -1,  # N-
    11: -1,  # S-
    13: +1,  # O+
    14: +1,  # S+
    15: -1,  # C-
    16: +1,  # P+
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
