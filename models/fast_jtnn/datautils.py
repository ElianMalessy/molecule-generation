import os
import random
import pickle
import torch
from torch.utils.data import Dataset, IterableDataset
from typing import List, Tuple, Any, Iterator

from models.fast_jtnn.jtnn_enc import JTNNEncoder
from models.fast_jtnn.mpn import MPN
from models.fast_jtnn.jtmpn import JTMPN

class PairTreeFolder(IterableDataset):
    def __init__(self, data_folder: str, vocab: Any, batch_size: int, shuffle: bool = True, y_assm: bool = True, replicate: int = None):
        self.data_folder = data_folder
        self.data_files = os.listdir(data_folder)
        self.batch_size = batch_size
        self.vocab = vocab
        self.y_assm = y_assm
        self.shuffle = shuffle
        if replicate is not None:
            self.data_files = self.data_files * replicate

    def __iter__(self) -> Iterator[Tuple[Any, Any]]:
        files = self.data_files.copy()
        if self.shuffle:
            random.shuffle(files)
            
        for fn in files:
            fn_path = os.path.join(self.data_folder, fn)
            with open(fn_path, 'rb') as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data)

            # Modern chunking
            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if batches and len(batches[-1]) < self.batch_size:
                batches.pop()

            for batch in batches:
                batch0, batch1 = zip(*batch)
                yield tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)

class MolTreeFolder(IterableDataset):
    def __init__(self, data_folder: str, vocab: Any, batch_size: int, shuffle: bool = True, assm: bool = True, replicate: int = None):
        self.data_folder = data_folder
        self.data_files = os.listdir(data_folder)
        self.batch_size = batch_size
        self.vocab = vocab
        self.shuffle = shuffle
        self.assm = assm

        if replicate is not None:
            self.data_files = self.data_files * replicate

    def __iter__(self) -> Iterator[Any]:
        files = self.data_files.copy()
        if self.shuffle:
            random.shuffle(files)
            
        for fn in files:
            fn_path = os.path.join(self.data_folder, fn)
            with open(fn_path, 'rb') as f:
                data = pickle.load(f)
                
            if self.shuffle: 
                random.shuffle(data)

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if batches and len(batches[-1]) < self.batch_size:
                batches.pop()

            for batch in batches:
                yield tensorize(batch, self.vocab, assm=self.assm)

class PairTreeDataset(Dataset):
    def __init__(self, data: List[Any], vocab: Any, y_assm: bool):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        batch0, batch1 = zip(*self.data[idx])
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)

class MolTreeDataset(Dataset):
    def __init__(self, data: List[Any], vocab: Any, assm: bool = True):
        self.data = data
        self.vocab = vocab
        self.assm = assm

    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Any:
        return tensorize(self.data[idx], self.vocab, assm=self.assm)

def tensorize(tree_batch: List[Any], vocab: Any, assm: bool = True) -> Tuple:
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder, mess_dict = JTNNEncoder.tensorize(tree_batch)
    mpn_holder = MPN.tensorize(smiles_batch)

    if not assm:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i, mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            if node.is_leaf or len(node.cands) == 1: 
                continue
            cands.extend([(cand, mol_tree.nodes, node) for cand in node.cands])
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.tensor(batch_idx, dtype=torch.long)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder, batch_idx)

def set_batch_nodeID(mol_batch: List[Any], vocab: Any):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
