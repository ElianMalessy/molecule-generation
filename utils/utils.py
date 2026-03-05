import random
import numpy as np
import torch
from functools import partial
from torch_geometric.datasets import ZINC
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.transforms import BaseTransform

import os
import pandas as pd
from dataclasses import dataclass, field

from moses_dataset import MosesPyGDataset
from utils.properties import build_props_cache, compute_normalisation_stats
from models.frattvae import build_frattvae_dataset, collate_pad_fn


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def kl_capacity(step: int, capacity_max: float, anneal_steps: int) -> float:
    """Linear ramp of KL capacity target from 0 → capacity_max over anneal_steps, then holds.

    The loss becomes kl_weight * |KL − C| so the model is penalised equally
    for being above OR below the target, preventing both collapse and
    over-regularisation early in training.
    """
    if anneal_steps <= 0:
        return capacity_max
    return min(capacity_max * step / anneal_steps, capacity_max)


@dataclass
class GVAEConfig:
    batch_size: int = 128
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    max_atoms: int = 38
    latent_dim: int = 128
    kl_weight: float = 0.3
    kl_anneal_steps: int = 100000    # steps over which both β=kl_weight and capacity ramp
    free_bits_per_dim: float = 0.02  # min KL per latent dim (nats); 0.02×128=2.56 nats floor
    kl_capacity_max: float = 25.0   # target KL ceiling (nats); ramps from 0
    valency_mask: bool = False       # apply valency masking during decoding
    # --- joint property prediction ---
    prop_pred: bool = False          # attach property prediction head
    prop_weight: float = 5.0         # γ: property loss weight at full scale
    prop_warmup_epochs: int = 8      # epochs before γ starts ramping up


@dataclass
class GVAENFConfig:
    """Same architecture as GVAEConfig plus a planar flow stack."""
    batch_size: int = 128
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    max_atoms: int = 38
    latent_dim: int = 128
    kl_weight: float = 0.3          # same as GVAE: same recon scale (~55-60 nats),
                                    # so same kl_weight keeps the KL/recon fraction
                                    # consistent. The IAF log-det already adds to
                                    # kl_flow, so 0.3 gives slightly more KL pressure
                                    # than GVAE — no need to raise the weight further.
    kl_anneal_steps: int = 100000    # steps over which both β=kl_weight and capacity ramp
    free_bits_per_dim: float = 0.02  # min KL per latent dim (nats); 0.02×128=2.56 nats floor
    kl_capacity_max: float = 25.0   # NF: same ceiling as GVAE (weak MLP decoder is the bottleneck)
    num_flows: int = 4               # number of IAF steps
    flow_hidden_dim: int = 256       # hidden dim of each MADE inside IAF
    valency_mask: bool = False
    # --- joint property prediction ---
    prop_pred: bool = False
    prop_weight: float = 5.0
    prop_warmup_epochs: int = 8


@dataclass
class GVAEARConfig:
    """GVAE with autoregressive Transformer decoder."""
    batch_size: int = 128
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    max_atoms: int = 38
    latent_dim: int = 128
    kl_weight: float = 1.0
    kl_anneal_steps: int = 60_000    # same ramp rate as before (0.00025 nats/step); reaches
                                     # C_max at ~35 epochs then holds — AR decoder needs less
    free_bits_per_dim: float = 0.02  # min KL per latent dim (nats); 0.02×128=2.56 nats floor
    kl_capacity_max: float = 15.0    # AR decoder reconstructs at ~3.5 nats; 15 nats gives
                                     # ~11.5 spare for property-correlated encoding
    valency_mask: bool = False
    # --- AR Transformer decoder ---
    ar_d_model: int = 256            # Transformer hidden dim
    ar_n_heads: int = 8              # attention heads
    ar_n_layers: int = 4             # Transformer layers
    ar_d_ff: int = 512               # feed-forward dim
    ar_dropout: float = 0.1
    # --- joint property prediction ---
    prop_pred: bool = False
    prop_weight: float = 5.0
    prop_warmup_epochs: int = 0      # no warmup — context_dropout forces z to be useful from
                                     # epoch 1, so property gradients can shape z as it forms
    context_dropout: float = 0.15   # fraction of input tokens replaced with 0 during training


@dataclass
class GVAEARNFConfig:
    """GVAE_AR with IAF normalizing flow encoder."""
    batch_size: int = 128
    epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    max_atoms: int = 38
    latent_dim: int = 128
    kl_weight: float = 1.0
    kl_anneal_steps: int = 57_000    # same ramp rate as original (0.000351 nats/step); reaches
                                     # C_max at ~33 epochs — NF improves posterior quality, not qty
    free_bits_per_dim: float = 0.01  # min KL per latent dim (nats); 0.01×128=1.28 nats floor
    kl_capacity_max: float = 20.0    # NF: 20 nats sufficient; AR context handles local structure
    num_flows: int = 4
    flow_hidden_dim: int = 256
    valency_mask: bool = False
    # --- AR Transformer decoder ---
    ar_d_model: int = 256
    ar_n_heads: int = 8
    ar_n_layers: int = 4
    ar_d_ff: int = 512
    ar_dropout: float = 0.1
    # --- joint property prediction ---
    prop_pred: bool = False
    prop_weight: float = 5.0
    prop_warmup_epochs: int = 0      # no warmup — context_dropout forces z to be useful from
                                     # epoch 1, so property gradients can shape z as it forms
    context_dropout: float = 0.15   # fraction of input tokens replaced with 0 during training


@dataclass
class FRATTVAEConfig:
    batch_size: int = 2048           # paper: 2048
    epochs: int = 1000
    lr: float = 1e-4                 # paper: 1e-4
    patience: int = 10
    latent_dim: int = 256            # paper: d_latent=256
    kl_weight: float = 0.0005        # paper: kl_w=0.0005
    depth: int = 32                  # paper: maxLength=32
    width: int = 16                  # paper: maxDegree=16
    d_model: int = 512               # paper: d_model=512
    d_ff: int = 2048                 # paper: d_ff=2048
    num_layers: int = 6              # paper: nlayer=6
    nhead: int = 8                   # paper: nhead=8
    n_bits: int = 2048               # Morgan fingerprint bits
    max_nfrags: int = 30             # max fragments per molecule during decoding
    label_loss_weight: float = 2.0   # paper: l_w=2.0
    n_jobs: int = 8                  # parallel workers for BRICS preprocessing


@dataclass
class Config:
    model: str = 'GVAE'
    dataset: str = 'ZINC'
    seed: int = 42
    num_samples: int = 10000
    num_workers: int = 4
    max_train_mols: int = 0          # cap training set size (0 = no cap, for quick tests)
    gvae: GVAEConfig = field(default_factory=GVAEConfig)
    gvae_nf: GVAENFConfig = field(default_factory=GVAENFConfig)
    gvae_ar: GVAEARConfig = field(default_factory=GVAEARConfig)
    gvae_ar_nf: GVAEARNFConfig = field(default_factory=GVAEARNFConfig)
    frattvae: FRATTVAEConfig = field(default_factory=FRATTVAEConfig)


class PropsDataset(torch.utils.data.Dataset):
    """
    Lightweight wrapper that injects a pre-computed property vector (plogP, QED, SA)
    into each PyG Data object as `data.props`.
    """
    def __init__(self, base_dataset, props: torch.Tensor):
        self.base  = base_dataset
        self.props = props   # (N, 3) float32

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        data = self.base[idx].clone()
        data.props = self.props[idx].unsqueeze(0)   # (1, 3) → PyG batches to (B, 3)
        return data


class NormalizeZINCBonds(BaseTransform):
    """
    ZINC bonds are naturally 1, 2, 3, 4.
    We shift them to 0, 1, 2, 3 to match MOSES and 0-indexing standards.
    """
    def forward(self, data):
        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
            data.edge_attr = data.edge_attr - 1
        return data


def ar_collate_fn(data_list, *, max_atoms, eos_id, max_seq_len):
    """Custom collate for GVAE_AR: build AR sequences in DataLoader workers.

    BFS serialization runs on CPU in parallel worker processes, so the GPU
    never blocks on Python list operations during forward().
    Returns (Batch, input_tokens, target_tokens, target_types, seq_lens)
    all as CPU tensors; the training loop moves them to device.
    """
    from models.gvae_ar import build_ar_batch  # lazy import avoids circular dependency
    batch = Batch.from_data_list(data_list)
    x_in  = batch.x.squeeze(-1) + 1
    ea_in = batch.edge_attr.squeeze(-1) + 1
    target_nodes, _ = to_dense_batch(x_in, batch.batch, max_num_nodes=max_atoms)
    target_edges = to_dense_adj(
        batch.edge_index, batch.batch,
        edge_attr=ea_in, max_num_nodes=max_atoms,
    ).squeeze(-1).long()
    input_tokens, target_tokens, target_types, seq_lens = build_ar_batch(
        target_nodes, target_edges, eos_id, max_seq_len,
    )
    return batch, input_tokens, target_tokens, target_types, seq_lens


def get_dataloaders(config: Config, logger):
    """Returns train_loader, val_loader, and dataset-specific metadata."""
    if config.model in ('GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF'):
        gc = {'GVAE': config.gvae, 'GVAE_NF': config.gvae_nf,
              'GVAE_AR': config.gvae_ar, 'GVAE_AR_NF': config.gvae_ar_nf}[config.model]
        if config.dataset == 'ZINC':
            transform = NormalizeZINCBonds()
            train_dataset = ZINC(root='data/ZINC', subset=False, split='train', pre_transform=transform)
            val_dataset = ZINC(root='data/ZINC', subset=False, split='val', pre_transform=transform)
            num_node_features, num_edge_features = 29, 5
        else:
            train_dataset = MosesPyGDataset(root='data/MOSES', split='train', max_atoms=gc.max_atoms)
            val_dataset = MosesPyGDataset(root='data/MOSES', split='test', max_atoms=gc.max_atoms)
            num_node_features, num_edge_features = 9, 5

        metadata: dict = {'num_nodes': num_node_features, 'num_edges': num_edge_features}

        if gc.prop_pred:
            from utils.constants import ZINC_ATOM_DECODER, ZINC_CHARGE_DECODER, MOSES_ATOM_DECODER
            atom_dec    = ZINC_ATOM_DECODER if config.dataset == 'ZINC' else MOSES_ATOM_DECODER
            charge_dec  = ZINC_CHARGE_DECODER if config.dataset == 'ZINC' else None
            cache_root  = os.path.join('data', config.dataset, 'prop_cache')
            os.makedirs(cache_root, exist_ok=True)

            train_props = build_props_cache(
                train_dataset, atom_dec, charge_dec,
                os.path.join(cache_root, 'props_train.pt'))
            val_props = build_props_cache(
                val_dataset, atom_dec, charge_dec,
                os.path.join(cache_root, 'props_val.pt'))

            prop_mean, prop_std = compute_normalisation_stats(train_props)
            metadata['prop_mean'] = prop_mean   # (3,)
            metadata['prop_std']  = prop_std    # (3,)

            train_dataset = PropsDataset(train_dataset, train_props)
            val_dataset   = PropsDataset(val_dataset,   val_props)

        # AR models pre-compute BFS sequences in the DataLoader workers so
        # the GPU never waits on Python list operations inside forward().
        # Use the standard TorchDataLoader instead of PyGDataLoader: our
        # ar_collate_fn already calls Batch.from_data_list internally, so
        # PyG's extra batching logic must not run on top of our 5-tuple output.
        if config.model in ('GVAE_AR', 'GVAE_AR_NF'):
            max_seq_len = gc.max_atoms * (gc.max_atoms + 1) // 2 + 1
            collate_fn = partial(ar_collate_fn, max_atoms=gc.max_atoms,
                                 eos_id=num_node_features, max_seq_len=max_seq_len)
            train_loader = TorchDataLoader(train_dataset, batch_size=gc.batch_size, shuffle=True,
                                           num_workers=config.num_workers, collate_fn=collate_fn)
            val_loader   = TorchDataLoader(val_dataset,   batch_size=gc.batch_size, shuffle=False,
                                           num_workers=config.num_workers, collate_fn=collate_fn)
        else:
            train_loader = PyGDataLoader(train_dataset, batch_size=gc.batch_size, shuffle=True,
                                         num_workers=config.num_workers)
            val_loader   = PyGDataLoader(val_dataset,   batch_size=gc.batch_size, shuffle=False,
                                         num_workers=config.num_workers)

        return train_loader, val_loader, metadata

    elif config.model == 'FRATTVAE':
        cache_dir = os.path.join('data', config.dataset, 'frattvae_cache')
        frattvae_kwargs = dict(
            max_nfrags=config.frattvae.max_nfrags,
            max_depth=config.frattvae.depth,
            max_degree=config.frattvae.width,
            n_bits=config.frattvae.n_bits,
        )

        train_smiles = get_smiles_list(config.dataset, split='train')
        val_split = 'val' if config.dataset == 'ZINC' else 'test'
        val_smiles = get_smiles_list(config.dataset, split=val_split)

        if config.max_train_mols > 0:
            train_smiles = train_smiles[:config.max_train_mols]
            val_smiles = val_smiles[:max(256, config.max_train_mols // 10)]
            logger.info(f"Capped dataset: {len(train_smiles)} train / {len(val_smiles)} val molecules")

        frattvae_kwargs['n_jobs'] = config.frattvae.n_jobs

        logger.info("Building / loading FRATTVAE training dataset...")
        train_data = build_frattvae_dataset(train_smiles, cache_dir, split_name='train', **frattvae_kwargs)
        logger.info("Building / loading FRATTVAE validation dataset...")
        val_data = build_frattvae_dataset(val_smiles, cache_dir, split_name=val_split, **frattvae_kwargs)

        train_loader = TorchDataLoader(
            train_data['dataset'],
            batch_size=config.frattvae.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            collate_fn=collate_pad_fn,
        )
        val_loader = TorchDataLoader(
            val_data['dataset'],
            batch_size=config.frattvae.batch_size,
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
            # valid_smiles from the val split: aligned with val_loader (shuffle=False).
            # None if the cache pre-dates this field (delete cache to regenerate).
            'val_smiles':     val_data.get('valid_smiles'),
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
        cache_path = os.path.join('data', 'MOSES', f'smiles_{moses_split}.txt')
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return [line.strip() for line in f if line.strip()]
        url = f"https://media.githubusercontent.com/media/molecularsets/moses/master/data/{moses_split}.csv"
        last_err = None
        for attempt in range(1, 6):
            try:
                df = pd.read_csv(url)
                break
            except Exception as e:
                last_err = e
                import time
                wait = 2 ** attempt
                print(f"MOSES download attempt {attempt} failed ({e}); retrying in {wait}s...")
                time.sleep(wait)
        else:
            raise RuntimeError(f"Failed to download MOSES {moses_split} after 5 attempts: {last_err}")
        col = 'SMILES' if 'SMILES' in df.columns else ('smiles' if 'smiles' in df.columns else df.columns[0])
        smiles = df[col].tolist()
        os.makedirs(os.path.join('data', 'MOSES'), exist_ok=True)
        with open(cache_path, 'w') as f:
            f.write('\n'.join(smiles))
        return smiles
