from torch_geometric.datasets import ZINC
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import BaseTransform

import os
import pandas as pd
from dataclasses import dataclass

from moses_dataset import MosesPyGDataset
from models.frattvae import build_frattvae_dataset, collate_pad_fn


@dataclass
class Config:
    model: str = 'GVAE'
    dataset: str = 'ZINC'
    batch_size: int = 128
    fratt_batch_size: int = 2048        # FRATTVAE batch size (paper: 2048, A100 can handle it)
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    seed: int = 42
    patience: int = 10

    max_atoms: int = 38
    latent_dim: int = 128           # GVAE latent dim
    fratt_latent_dim: int = 256     # FRATTVAE latent dim (paper: d_latent=256)
    kl_weight: float = 1.0
    kl_anneal_steps: int = 40000
    fratt_lr: float = 1e-4          # FRATTVAE learning rate (paper: 1e-4)
    num_samples: int = 10000
    num_workers: int = 4

    # FRATTVAE-specific hyperparameters (defaults match original paper)
    fratt_depth: int = 32           # max tree depth  (paper: maxLength=32)
    fratt_width: int = 16           # max tree degree (paper: maxDegree=16)
    fratt_d_model: int = 512        # transformer hidden dim (paper: d_model=512)
    fratt_d_ff: int = 2048          # transformer FFN dim    (paper: d_ff=2048)
    fratt_layers: int = 6           # number of transformer layers (paper: nlayer=6)
    fratt_nhead: int = 8            # attention heads
    fratt_n_bits: int = 2048        # Morgan fingerprint bits
    fratt_max_nfrags: int = 30      # max fragments per molecule during decoding
    label_loss_weight: float = 2.0  # cross-entropy weight relative to KL
    n_jobs: int = 8                 # parallel workers for BRICS preprocessing
    max_train_mols: int = 0         # cap training set size (0 = no cap, for quick tests)


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
            num_node_features, num_edge_features = 29, 5
        else:
            train_dataset = MosesPyGDataset(root='data/MOSES', split='train', max_atoms=config.max_atoms)
            val_dataset = MosesPyGDataset(root='data/MOSES', split='test', max_atoms=config.max_atoms)
            num_node_features, num_edge_features = 9, 5

        train_loader = PyGDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
        val_loader = PyGDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)

        return train_loader, val_loader, {'num_nodes': num_node_features, 'num_edges': num_edge_features}

    elif config.model == 'FRATTVAE':
        cache_dir = os.path.join('data', config.dataset, 'frattvae_cache')
        frattvae_kwargs = dict(
            max_nfrags=config.fratt_max_nfrags,
            max_depth=config.fratt_depth,
            max_degree=config.fratt_width,
            n_bits=config.fratt_n_bits,
        )

        train_smiles = get_smiles_list(config.dataset, split='train')
        val_split = 'val' if config.dataset == 'ZINC' else 'test'
        val_smiles = get_smiles_list(config.dataset, split=val_split)

        if config.max_train_mols > 0:
            train_smiles = train_smiles[:config.max_train_mols]
            val_smiles = val_smiles[:max(256, config.max_train_mols // 10)]
            logger.info(f"Capped dataset: {len(train_smiles)} train / {len(val_smiles)} val molecules")

        frattvae_kwargs['n_jobs'] = config.n_jobs

        logger.info("Building / loading FRATTVAE training dataset...")
        train_data = build_frattvae_dataset(train_smiles, cache_dir, split_name='train', **frattvae_kwargs)
        logger.info("Building / loading FRATTVAE validation dataset...")
        val_data = build_frattvae_dataset(val_smiles, cache_dir, split_name=val_split, **frattvae_kwargs)

        train_loader = TorchDataLoader(
            train_data['dataset'],
            batch_size=config.fratt_batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_pad_fn,
        )
        val_loader = TorchDataLoader(
            val_data['dataset'],
            batch_size=config.fratt_batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            collate_fn=collate_pad_fn,
        )

        # Use train vocab/ecfps as the canonical reference (val may have fewer fragments)
        metadata = {
            'num_frags':      len(train_data['uni_fragments']),
            'uni_fragments':  train_data['uni_fragments'],
            'frag_ecfps':     train_data['frag_ecfps'],
            'ndummys':        train_data['ndummys'],
            'freq_label':     train_data['freq_label'],
        }
        return train_loader, val_loader, metadata

    raise ValueError(f"Invalid model/dataset configuration: {config.model} / {config.dataset}")


def get_smiles_list(dataset_name, split):
    if dataset_name == 'ZINC':
        # Mirror the PyG ZINC (full) split sizes: 220011 train / 24445 val / 4999 test
        _ZINC_SPLITS = {'train': (0, 220011), 'val': (220011, 244456), 'test': (244456, None)}
        cache_path = os.path.join('data', 'ZINC', f'smiles_{split}.txt')
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return [line.strip() for line in f if line.strip()]
        from datasets import load_dataset
        hf_ds = load_dataset('edmanft/zinc250k', split='train')
        all_smiles = [s.strip() for s in hf_ds['smiles']]
        # Save all splits to disk on first run
        os.makedirs(os.path.join('data', 'ZINC'), exist_ok=True)
        for sp, (start, end) in _ZINC_SPLITS.items():
            with open(os.path.join('data', 'ZINC', f'smiles_{sp}.txt'), 'w') as f:
                f.write('\n'.join(all_smiles[start:end]))
        start, end = _ZINC_SPLITS[split]
        return all_smiles[start:end]
    else:
        moses_split = 'train' if split == 'train' else 'test'
        url = f"https://media.githubusercontent.com/media/molecularsets/moses/master/data/{moses_split}.csv"
        df = pd.read_csv(url)
        col = 'SMILES' if 'SMILES' in df.columns else ('smiles' if 'smiles' in df.columns else df.columns[0])
        return df[col].tolist()
