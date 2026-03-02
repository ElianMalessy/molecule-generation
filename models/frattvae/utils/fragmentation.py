import os
from itertools import chain
from rdkit import Chem
from rdkit.Chem import RWMol, BRICS

from .medchemfrag import decomposition


def FindBRICS(mol, bonds: list = None, AtomDone: set = None):
    bonds = bonds if bonds is not None else []
    AtomDone = AtomDone if AtomDone is not None else set([])

    for idxs, _ in BRICS.FindBRICSBonds(mol):
        idxs = sorted(idxs)
        if (idxs in bonds) or (len(set(idxs) & AtomDone) > 1):
            continue
        else:
            bonds.append(sorted(idxs))
            AtomDone = AtomDone | set(idxs)
    return bonds, AtomDone


def FindRings(mol, bonds: list = None, AtomDone: set = None):
    bonds = bonds if bonds is not None else []
    AtomDone = AtomDone if AtomDone is not None else set([])
    for bond in mol.GetBonds():
        begin = bond.GetBeginAtom()
        end = bond.GetEndAtom()

        if (begin.GetIdx() in AtomDone) & (end.GetIdx() in AtomDone):
            continue

        if (begin.IsInRing() | end.IsInRing()) & (not bond.IsInRing()) & (bond.GetBondTypeAsDouble() < 2):
            if begin.IsInRing() & end.IsInRing():
                neighbor = 1
            elif begin.IsInRing():
                neighbor = len(end.GetNeighbors()) - 1
            elif end.IsInRing():
                neighbor = len(begin.GetNeighbors()) - 1
            else:
                neighbor = 0

            if neighbor > 0:
                idxs = sorted([begin.GetIdx(), end.GetIdx()])
                if idxs not in bonds:
                    bonds.append(idxs)
                    AtomDone = AtomDone | set(idxs)
    return bonds, AtomDone


def find_BRICSbonds(mol) -> list:
    return sorted([sorted(idxs) for idxs, _ in BRICS.FindBRICSBonds(mol)], key=lambda idxs: idxs[1])


def find_rings(mol) -> list:
    return sorted(FindRings(mol)[0], key=lambda idxs: idxs[1])


def find_MedChemFrag(mol) -> list:
    return sorted([sorted(idxs) for idxs in decomposition(mol)], key=lambda idxs: idxs[1])


def find_BRICSbonds_and_rings(mol) -> list:
    """Find bonds which are BRICS bonds or single bonds between rings."""
    bonds = find_BRICSbonds(mol)
    return sorted(FindRings(mol, bonds=bonds)[0], key=lambda idxs: idxs[1])
