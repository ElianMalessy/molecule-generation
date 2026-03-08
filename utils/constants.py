# Store datasets specific mappings here to keep logic clean
from rdkit import Chem

MOSES_ATOM_DECODER = {0: 6, 1: 7, 2: 8, 3: 9, 4: 16, 5: 17, 6: 35, 7: 1}

# 0-indexed atom class (inverse of ZINC_ATOM_VOCAB; data.x stores these
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

# Encoding vocab: (atomic_num, formal_charge) → 0-indexed atom class.
# Inverse of ZINC_ATOM_DECODER + ZINC_CHARGE_DECODER.
# gvae_prepare_batch / ar_collate_fn apply +1 so the model sees classes 1-17.
ZINC_ATOM_VOCAB: dict = {
    (6,   0):  0,  # C
    (7,   0):  1,  # N
    (8,   0):  2,  # O
    (16,  0):  3,  # S
    (9,   0):  4,  # F
    (7,  +1):  5,  # N+
    (17,  0):  6,  # Cl
    (8,  -1):  7,  # O-
    (35,  0):  8,  # Br
    (7,  -1):  9,  # N-
    (53,  0): 10,  # I
    (16, -1): 11,  # S-
    (15,  0): 12,  # P
    (8,  +1): 13,  # O+
    (16, +1): 14,  # S+
    (6,  -1): 15,  # C-
    (15, +1): 16,  # P+
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
    """Return RDKit BondType for a bond index. Used when rebuilding ground-truth molecules
    (e.g. mol_from_data in properties.py) where the full aromatic annotation is available
    and consistent."""
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.AROMATIC,
        3: Chem.BondType.DOUBLE,
        4: Chem.BondType.TRIPLE,
    }.get(bond_idx, Chem.BondType.SINGLE)


def decode_bond_type(bond_idx: int) -> Chem.BondType:
    """Return RDKit BondType for a *decoded* bond index.

    Aromatic bonds (index 2) are deliberately added as SINGLE bonds so that
    RDKit's SanitizeMol aromaticity-perception step can assign aromaticity from
    the ring topology instead of from explicit aromatic annotations.

    When a decoder outputs a mix of aromatic/non-aromatic bonds for what should
    be an aromatic ring, passing Chem.BondType.AROMATIC directly causes ~40% of
    sanitization failures.  Adding as SINGLE and letting RDKit perceive aromaticity
    is robust: if the ring is a valid Hückel system RDKit makes it aromatic and
    MolToSmiles gives lowercase atoms; if not, the explicit single bonds are still
    chemically valid and sanitize cleanly.
    """
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.SINGLE,    # aromatic decoded → SINGLE; RDKit perceives ring aromaticity
        3: Chem.BondType.DOUBLE,
        4: Chem.BondType.TRIPLE,
    }.get(bond_idx, Chem.BondType.SINGLE)
