import sys
from collections import Counter
from copy import deepcopy
from itertools import chain
import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from tqdm import tqdm

from .decompose import MolFromFragments, MolToBRICSfragments, MapNumsToAdj

try:
    from molvs import standardize_smiles as _standardize_smiles
except ImportError:
    def _standardize_smiles(s):
        return Chem.CanonSmiles(s) or s

lg = RDLogger.logger()
lg.setLevel(RDLogger.CRITICAL)


def smiles2mol(s):
    m = Chem.MolFromSmiles(s)
    if m is None:
        print(f'[ERROR] {s} is not valid.', flush=True)
    return m


def frag2ecfp(frag, radius: int = 2, n_bits: int = 2048, useChiral: bool = True, ignore_dummy: bool = False):
    if ignore_dummy:
        frag = Chem.RemoveHs(AllChem.ReplaceSubstructs(frag, Chem.MolFromSmiles('*'), Chem.MolFromSmiles('[H]'), replaceAll=True)[0])
    return AllChem.GetMorganFingerprintAsBitVect(frag, radius, n_bits, useChirality=useChiral)


def FragmentsToIndices(fragments_list: list, fragment_idxs: dict, verbose: int = 0):
    if verbose:
        return [[fragment_idxs[f] for f in frags] for frags in tqdm(fragments_list)]
    else:
        return [[fragment_idxs[f] for f in frags] for frags in fragments_list]


def debugMolToBRICSfragments(mol, useChiral: bool = True, ignore_double: bool = False,
                              minFragSize: int = 1, maxFragNums: int = 50, maxDegree: int = 32):
    recon = 1
    iters = 0
    max_iters = 30
    try:
        s = Chem.MolToSmiles(mol) if mol is not None else None
        Chem.FindMolChiralCenters(mol)
        hasChiral = bool(len(Chem.FindPotentialStereo(mol, flagPossible=False))) if useChiral else False
        frags, bond_types, bondMapNums = MolToBRICSfragments(mol, minFragSize=minFragSize, maxDegree=maxDegree,
                                                             useChiral=hasChiral, useStereo=ignore_double)
        while (len(frags) > maxFragNums):
            iters += 1
            minFragSize += 1
            frags, bond_types, bondMapNums = MolToBRICSfragments(mol, minFragSize=minFragSize, maxDegree=maxDegree,
                                                                  useChiral=hasChiral, useStereo=ignore_double)
            if iters > max_iters:
                raise ValueError(f'Over max iteration; {max_iters}. Remove it or increase max_nfrags.')

        adj = MapNumsToAdj(bondMapNums, bond_types)
        s1 = Chem.MolToSmiles(mol, isomericSmiles=hasChiral)
        s2 = Chem.MolToSmiles(MolFromFragments(frags, adj, asMol=True), isomericSmiles=hasChiral)
        if (s1 != s2) & (s1 != _standardize_smiles(s2)):
            if hasChiral:
                s1_dash = Chem.CanonSmiles(s1, useChiral=0)
                s2_dash = Chem.CanonSmiles(s2, useChiral=0)
                if (s1_dash == s2_dash) | (s1_dash == _standardize_smiles(s2_dash)):
                    recon = 2
                else:
                    recon = 0
                    print(f"{s1_dash}, {s2_dash} is 2D unreconstructable.", flush=True)
            else:
                recon = 0
                print(f"{s}, {s2} is unreconstructable.", flush=True)
        else:
            recon = 3 if hasChiral else 1
    except Exception as e:
        recon = 0
        print(f'{mol} is an ERROR; {str(e)}', flush=True)

    if recon == 0:
        frags, bond_types, bondMapNums = None, None, None

    return frags, bond_types, bondMapNums, recon


def parallelMolsToBRICSfragments(mols: list, useChiral: bool = True, ignore_double: bool = False,
                                  minFragSize: int = 1, maxFragNums: int = 50, maxDegree: int = 32,
                                  df_frag=None, asFragments: bool = False, n_jobs: int = -1, verbose: int = 0):
    _gen = Parallel(n_jobs=n_jobs, return_as='generator')(
        delayed(debugMolToBRICSfragments)(
            mol, minFragSize=minFragSize, maxFragNums=maxFragNums, maxDegree=maxDegree,
            useChiral=useChiral, ignore_double=ignore_double
        ) for mol in mols
    )
    results = list(tqdm(_gen, total=len(mols), desc='BRICS decomposition', leave=True))
    fragments_list, bondtypes_list, bondMapNums_list, recon_flag = zip(*results)
    fragments_list = [f for f in fragments_list if f is not None]
    bondtypes_list = [b for b in bondtypes_list if b is not None]
    bondMapNums_list = [m for m in bondMapNums_list if m is not None]

    all_fragments = list(chain.from_iterable(fragments_list))
    frag_freq = Counter(all_fragments)
    uni_fragments_raw, freq_list_raw = map(list, zip(*frag_freq.items()))
    df = pd.DataFrame({'SMILES': uni_fragments_raw, 'frequency': freq_list_raw})
    df = df.assign(length=df.SMILES.str.len())
    df = df.sort_values(['frequency', 'length'], ascending=[False, True])

    if df_frag is not None and len(df_frag) > 0:
        df = pd.concat([df_frag, df]).drop_duplicates(subset='SMILES', keep='first').reset_index(drop=True)
        uni_fragments = df.SMILES.tolist()
        freq_list = df.frequency.tolist()
    else:
        uni_fragments = df.SMILES.tolist()
        freq_list = df.frequency.tolist()
        uni_fragments = ['*'] + uni_fragments
        freq_list = [0] + freq_list

    if not asFragments:
        fragment_idxs = dict(zip(uni_fragments, range(len(uni_fragments))))
        fragments_list = FragmentsToIndices(fragments_list, fragment_idxs, verbose=verbose)

    return fragments_list, bondtypes_list, bondMapNums_list, list(recon_flag), uni_fragments, freq_list


def _smilesToMorganFingarPrintsAsBitVect(smiles, radius: int, n_bits: int, useChiral: bool = True, ignore_dummy: bool = True):
    ecfp = frag2ecfp(Chem.MolFromSmiles(smiles), radius, n_bits, useChiral=useChiral, ignore_dummy=ignore_dummy)
    ecfp_bits = np.array(ecfp, dtype=int).tolist()
    return ecfp, ecfp_bits


def SmilesToMorganFingetPrints(fragments: list, n_bits: int, dupl_bits: int = 0, radius: int = 2,
                                ignore_dummy: bool = True, useChiral: bool = True, n_jobs: int = 1):
    """
    fragments: a list of fragment SMILES strings.
    Returns a list of bit-vector lists of length n_bits + dupl_bits.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(_smilesToMorganFingarPrintsAsBitVect)(f, radius, n_bits, useChiral=useChiral, ignore_dummy=ignore_dummy)
        for f in fragments
    )
    ecfps, ecfp_list = map(list, zip(*results))

    if dupl_bits > 0:
        _, indices = np.unique(ecfp_list, axis=0, return_index=True)
        uni_ecfps = [ecfps[i] for i in indices]
        dupl_idxs = []
        for e in uni_ecfps:
            dupl = np.where(np.array(DataStructs.BulkTanimotoSimilarity(e, ecfps)) == 1)[0]
            if len(dupl) >= 2:
                dupl_idxs.append(sorted(dupl.tolist()))

        if len(dupl_idxs) != 0:
            max_len = max(list(map(len, dupl_idxs)))
            if max_len > 2 ** dupl_bits:
                raise ValueError(f'the number of duplicated ecfp {max_len} is greater than 2**{dupl_bits}')
            bits = [list(map(lambda x: int(x), list(f"{i:0{dupl_bits}b}"))) for i in range(max_len)]
            distinct_ecfp_list = [e + bits[0] for e in ecfp_list]
            for dupl in dupl_idxs:
                for i, d in enumerate(dupl):
                    distinct_ecfp_list[d] = distinct_ecfp_list[d][:-dupl_bits] + bits[i]
            ecfp_list = distinct_ecfp_list
        print(f'the number of duplicated ecfp is {len(dupl_idxs)}', flush=True)

    return ecfp_list
