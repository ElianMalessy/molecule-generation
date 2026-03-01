from torch_geometric.datasets import ZINC
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import BaseTransform

import os
import pandas as pd
from dataclasses import dataclass
from tqdm import tqdm

from moses_dataset import MosesPyGDataset
from models.fast_jtnn import Vocab, MolTreeDataset, MolTree

@dataclass
class Config:
    model: str = 'GVAE'
    dataset: str = 'ZINC'
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    patience: int = 10
    
    max_atoms: int = 38
    latent_dim: int = 128  
    kl_weight: float = 1.0
    kl_anneal_steps: int = 40000
    num_samples: int = 10000
    

class NormalizeZINCBonds(BaseTransform):
    """
    ZINC bonds are naturally 1, 2, 3, 4. 
    We shift them to 0, 1, 2, 3 to match MOSES and 0-indexing standards.
    """
    def forward(self, data):
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr - 1
        return data

def get_dataloaders(config: Config, logger):
    """Returns train_loader, val_loader, and dataset-specific metadata."""
    if config.model == 'GVAE':
        if config.dataset == 'ZINC':
            transform = NormalizeZINCBonds()
            train_dataset = ZINC(root='data/ZINC', subset=False, split='train', pre_transform=transform)
            val_dataset = ZINC(root='data/ZINC', subset=False, split='val', pre_transform=transform)
            num_node_features, num_edge_features = 28, 4
        else:
            train_dataset = MosesPyGDataset(root='data/MOSES', split='train', max_atoms=config.max_atoms)
            val_dataset = MosesPyGDataset(root='data/MOSES', split='test', max_atoms=config.max_atoms)
            num_node_features, num_edge_features = 9, 5

        train_loader = PyGDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4)
        val_loader = PyGDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4)
        
        return train_loader, val_loader, {'num_nodes': num_node_features, 'num_edges': num_edge_features}

    elif config.model == 'JTVAE':
        train_smiles = get_smiles_list(config.dataset, split='train')
        val_smiles = get_smiles_list(config.dataset, split='val')
        
        vocab = get_or_build_jtvae_vocab(train_smiles, config.dataset, logger)
        train_dataset = MolTreeDataset(train_smiles, vocab, assm=True)
        val_dataset = MolTreeDataset(val_smiles, vocab, assm=True)
        
        train_loader = TorchDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, collate_fn=lambda x:x)
        val_loader = TorchDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=4, collate_fn=lambda x:x)
        
        return train_loader, val_loader, {'vocab': vocab}

    raise ValueError(f"Invalid model/dataset configuration: {config.model} / {config.dataset}")



def get_smiles_list(dataset_name, split):
    if dataset_name == 'ZINC':
        dataset = ZINC(root='data/ZINC', subset=False, split=split)
        return [data.smiles for data in dataset]
    else:
        moses_split = 'train' if split == 'train' else 'test'
        url = f"https://media.githubusercontent.com/media/molecularsets/moses/master/data/{moses_split}.csv"
        df = pd.read_csv(url)
        col = 'SMILES' if 'SMILES' in df.columns else ('smiles' if 'smiles' in df.columns else df.columns[0])
        return df[col].tolist()

def get_or_build_jtvae_vocab(smiles_list, dataset_name, logger):
    """Builds or loads the JTNN Vocab file containing the subgraph cliques."""
    vocab_path = f'data/{dataset_name}_jtvae_vocab.txt'
    if os.path.exists(vocab_path):
        logger.info(f"Loading JTVAE vocab from {vocab_path}...")
        with open(vocab_path, 'r') as f:
            return Vocab([line.strip() for line in f.readlines()])
    
    logger.info("Building JTVAE Vocabulary from cliques (this takes a few minutes)...")
    cset = set()
    for s in tqdm(smiles_list, desc="Extracting subgraphs"):
        try:
            mol_tree = MolTree(s)
            for node in mol_tree.nodes:
                cset.add(node.smiles)
        except Exception:
            continue
            
    vocab_smiles = list(cset)
    os.makedirs('data', exist_ok=True)
    with open(vocab_path, 'w') as f:
        for s in vocab_smiles:
            f.write(s + '\n')
    return Vocab(vocab_smiles)


