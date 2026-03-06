# Store datasets specific mappings here to keep logic clean
from rdkit import Chem

MOSES_ATOM_DECODER = {0: 6, 1: 7, 2: 8, 3: 9, 4: 16, 5: 17, 6: 35, 7: 1}

ZINC_ATOM_DECODER = {
    0: 6, 1: 8, 2: 7, 3: 9, 4: 6, 5: 16, 6: 17, 7: 8,
    8: 7, 9: 35, 10: 7, 11: 7, 12: 7, 13: 7, 14: 16, 15: 53,
    16: 15, 17: 8, 18: 7, 19: 8, 20: 16, 21: 15, 22: 15, 23: 6,
    24: 15, 25: 16, 26: 6, 27: 15
}

ZINC_CHARGE_DECODER = {
    7: -1, 8: 1, 10: 1, 11: 1, 12: 1, 13: -1, 14: -1,
    17: 1, 18: -1, 19: 1, 20: 1, 23: -1, 24: 1, 25: 1,
    26: -1, 27: 1
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

# Valence units consumed per bond-type index.
# Aromatic bonds are tracked as 1 (integer approximation).
BOND_ORDER = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1}


def get_rdkit_bond(bond_idx: int) -> Chem.BondType:
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }.get(bond_idx, Chem.BondType.SINGLE)
