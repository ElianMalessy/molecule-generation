import random
import numpy as np
import torch
from functools import partial
from torch_geometric.data import Data, Batch
from torch_geometric.utils import to_dense_batch, to_dense_adj
from torch.utils.data import DataLoader as TorchDataLoader
from torch_geometric.loader import DataLoader as PyGDataLoader

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
    patience: int = 10
    max_atoms: int = 38
    latent_dim: int = 128
    kl_weight: float = 1.0           # capacity penalty weight: loss += kl_weight * |KL - C|.
                                     # Must be ≥1.0 for the flat decoder: at 0.3 the encoder
                                     # can maintain KL=65 >> C=25 because the reconstruction
                                     # benefit of extra bits outweighs the 0.3*40=12 nat penalty.
                                     # At 1.0 the penalty (1.0*40=40 nats) overwhelms it.
    kl_anneal_steps: int = 100000    # steps over which both β=kl_weight and capacity ramp
    free_bits_per_dim: float = 0.02  # min KL per latent dim (nats); 0.02×128=2.56 nats floor
    kl_capacity_max: float = 25.0   # target KL ceiling (nats); ramps from 0
    valency_mask: bool = False       # apply valency masking during decoding
    # --- joint property prediction ---
    prop_pred: bool = False          # attach property prediction head
    prop_weight: float = 1.5         # γ: property loss weight (constant)


@dataclass
class GVAENFConfig:
    """Same architecture as GVAEConfig plus a planar flow stack."""
    batch_size: int = 128
    epochs: int = 1000
    lr: float = 1e-3
    patience: int = 10
    max_atoms: int = 38
    latent_dim: int = 128
    kl_weight: float = 1.0           # same reasoning as GVAEConfig: must be 1.0 to prevent
                                    # KL >> capacity, especially since the IAF adds log_det.
    kl_anneal_steps: int = 100000    # steps over which both β=kl_weight and capacity ramp
    free_bits_per_dim: float = 0.02  # min KL per latent dim (nats); 0.02×128=2.56 nats floor
    kl_capacity_max: float = 25.0   # NF: same ceiling as GVAE (weak MLP decoder is the bottleneck)
    num_flows: int = 4               # number of IAF steps
    flow_hidden_dim: int = 256       # hidden dim of each MADE inside IAF
    valency_mask: bool = False
    # --- joint property prediction ---
    prop_pred: bool = False
    prop_weight: float = 1.0


@dataclass
class GVAEARConfig:
    """GVAE with autoregressive Transformer decoder."""
    batch_size: int = 256
    epochs: int = 1000
    lr: float = 2e-3
    patience: int = 15
    max_atoms: int = 38
    latent_dim: int = 128
    kl_weight: float = 0.05
    kl_anneal_steps: int = 60_000    # β ramp: 0 → kl_weight over this many steps.
                                     # At batch=256 this covers ~70 epochs; encoder learns
                                     # informative posterior before KL penalty is applied.
    free_bits_per_dim: float = 0.02  # min KL per latent dim (nats); 0.02×128=2.56 nats floor
    kl_capacity_max: float = 15.0    # UNUSED for AR models (β-annealing replaces capacity hinge);
                                     # kept for serialisation compatibility.
    valency_mask: bool = False
    # --- AR Transformer decoder ---
    ar_d_model: int = 256            # Transformer hidden dim
    ar_n_heads: int = 8              # attention heads
    ar_n_layers: int = 2             # Transformer layers
    ar_d_ff: int = 512               # feed-forward dim
    ar_dropout: float = 0.1
    # --- joint property prediction ---
    prop_pred: bool = False
    prop_weight: float = 0.6
    context_dropout: float = 0.35    # fraction of input tokens replaced with 0 during training.
                                     # At 0.15 the decoder retains 85 % sequential context and
                                     # learns to ignore z — encoder μ collapses to one mode.
                                     # 0.35 forces the decoder to use z for structural decisions;
                                     # 0.5 provides stronger forcing now that edge class weights
                                     # make bond prediction require z for the hidden positions.


@dataclass
class GVAEARNFConfig:
    """GVAE_AR with IAF normalizing flow encoder."""
    batch_size: int = 512
    epochs: int = 1000
    lr: float = 4e-3
    patience: int = 15
    max_atoms: int = 38
    latent_dim: int = 128
    kl_weight: float = 0.1
    kl_anneal_steps: int = 57_000    # β ramp: 0 → kl_weight over this many steps.
                                     # At batch=512 this covers ~60 epochs.
    free_bits_per_dim: float = 0.01  # min KL per latent dim (nats); 0.01×128=1.28 nats floor
    kl_capacity_max: float = 20.0    # UNUSED for AR models (β-annealing replaces capacity hinge);
                                     # kept for serialisation compatibility.
    num_flows: int = 4
    flow_hidden_dim: int = 256
    valency_mask: bool = False
    # --- AR Transformer decoder ---
    ar_d_model: int = 256
    ar_n_heads: int = 8
    ar_n_layers: int = 2
    ar_d_ff: int = 512
    ar_dropout: float = 0.1
    # --- joint property prediction ---
    prop_pred: bool = False
    prop_weight: float = 0.6
    context_dropout: float = 0.35


@dataclass
class FRATTVAEConfig:
    batch_size: int = 2048           # paper: 2048
    epochs: int = 1000
    lr: float = 1e-4                 # paper: 1e-4
    patience: int = 20
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




# ---------------------------------------------------------------------------
# ZINC250k dataset helpers
# ---------------------------------------------------------------------------

# (atomic_num, formal_charge) → 0-indexed atom class (0=C, …, 16=P+).
# gvae_prepare_batch / ar_collate_fn apply +1 so the model sees classes 1-17.
_ZINC_ATOM_VOCAB: dict = {
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


def _smiles_to_pyg_data(smi: str):
    """Convert a SMILES string to a PyG Data object.

    Atom features (x): shape (N, 1), long, 0-indexed class 0-16.
    Edge features (edge_attr): shape (2E, 1), long, 0-indexed:
        0=single, 1=aromatic, 2=double, 3=triple.
    Both directions stored (undirected graph convention).
    Returns None if RDKit cannot parse the SMILES.
    """
    from rdkit import Chem
    _bond_vocab = {
        Chem.BondType.SINGLE:   0,
        Chem.BondType.AROMATIC: 1,
        Chem.BondType.DOUBLE:   2,
        Chem.BondType.TRIPLE:   3,
    }
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    # Re-parse canonical SMILES to get a consistent atom ordering.
    mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
    if mol is None:
        return None

    xs = [_ZINC_ATOM_VOCAB.get((a.GetAtomicNum(), a.GetFormalCharge()), 0)
          for a in mol.GetAtoms()]
    x = torch.tensor(xs, dtype=torch.long).unsqueeze(1)  # (N, 1)

    src, dst, ea = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bt = _bond_vocab.get(bond.GetBondType(), 0)
        src += [i, j]; dst += [j, i]; ea += [bt, bt]

    if src:
        edge_index = torch.tensor([src, dst], dtype=torch.long)
        edge_attr  = torch.tensor(ea, dtype=torch.long).unsqueeze(1)  # (2E, 1)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr  = torch.zeros((0, 1), dtype=torch.long)

    data        = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.smiles = Chem.MolToSmiles(mol)  # canonical
    return data


def _build_zinc250k_datasets(seed: int = 42):
    """Download ZINC250k from HuggingFace, make a 95/5 train/val split, and
    convert all molecules to PyG Data objects.  Results are cached as .pt files
    so subsequent calls are instant.  The SMILES lists are also written to
    smiles_train.txt / smiles_val.txt so FRATTVAE can reuse them via
    get_smiles_list().

    Returns (train_list, val_list) — plain Python lists of PyG Data objects.
    """
    os.makedirs('data/ZINC', exist_ok=True)
    # v2 suffix marks the standardized (molvs) cache; old v1 files are bypassed.
    cache_train = os.path.join('data', 'ZINC', 'zinc250k_pyg_train_v2.pt')
    cache_val   = os.path.join('data', 'ZINC', 'zinc250k_pyg_val_v2.pt')

    def _load_cache():
        """Load from cache, regenerating .txt side-files if absent."""
        txt_train = os.path.join('data', 'ZINC', 'smiles_train.txt')
        txt_val   = os.path.join('data', 'ZINC', 'smiles_val.txt')
        train_list = torch.load(cache_train, weights_only=False)
        val_list   = torch.load(cache_val,   weights_only=False)
        if not os.path.exists(txt_train) or not os.path.exists(txt_val):
            print("Regenerating smiles_*.txt from .pt cache…")
            with open(txt_train, 'w') as f:
                f.write('\n'.join(d.smiles for d in train_list))
            with open(txt_val, 'w') as f:
                f.write('\n'.join(d.smiles for d in val_list))
            print(f"Written {len(train_list)} train + {len(val_list)} val SMILES.")
        return train_list, val_list

    # Fast path: cache already present — no lock needed.
    if os.path.exists(cache_train) and os.path.exists(cache_val):
        return _load_cache()

    # Slow path: build from scratch.  All SLURM array tasks start simultaneously,
    # so we use an exclusive file lock to ensure only ONE task does the expensive
    # build; the others wait and then load from the cache built by the winner.
    import fcntl
    lock_path = os.path.join('data', 'ZINC', 'zinc250k_build.lock')
    with open(lock_path, 'w') as _lock_fh:
        print(f"[PID {os.getpid()}] Waiting for dataset build lock…")
        fcntl.flock(_lock_fh, fcntl.LOCK_EX)   # blocks until no other process holds it
        # Re-check: another process may have built the cache while we waited.
        if os.path.exists(cache_train) and os.path.exists(cache_val):
            print(f"[PID {os.getpid()}] Cache built by another process — loading.")
            return _load_cache()

        print(f"[PID {os.getpid()}] Building ZINC250k cache (this happens once)…")

        print("Downloading ZINC250k from HuggingFace (edmanft/zinc250k)…")
        from datasets import load_dataset
        hf_ds      = load_dataset('edmanft/zinc250k', split='train')
        all_smiles = list(hf_ds['smiles'])

        rng = random.Random(seed)
        rng.shuffle(all_smiles)
        n_train      = int(0.95 * len(all_smiles))
        train_smiles = all_smiles[:n_train]
        val_smiles   = all_smiles[n_train:]

        # Standardize SMILES with molvs (same pipeline as original FRATTVAE paper).
        # ~0.8 % of molecules change: sulfinyl S(=O) → zwitterion [S+]([O-]).
        # Both representations are in our atom/bond vocabulary, but standardizing
        # ensures consistency with the paper's training distribution.
        print("Standardizing SMILES with molvs…")
        from rdkit import Chem as _Chem
        from molvs import Standardizer as _Std
        # molvs logs every rule application at INFO level; silence it.
        import logging as _logging
        for _lg in ('molvs', 'molvs.standardize', 'molvs.normalize', 'molvs.charge'):
            _logging.getLogger(_lg).setLevel(_logging.ERROR)
        _stand = _Std()

        def _std_smi(smi):
            try:
                mol = _Chem.MolFromSmiles(smi)
                if mol is None:
                    return None
                return _Chem.MolToSmiles(_stand.standardize(mol))
            except Exception:
                return None

        train_smiles = [s for raw in train_smiles if (s := _std_smi(raw)) is not None]
        val_smiles   = [s for raw in val_smiles   if (s := _std_smi(raw)) is not None]
        print(f"After standardization: {len(train_smiles)} train, {len(val_smiles)} val.")

        # Persist SMILES lists for FRATTVAE / quick re-use.
        for fname, smi_list in [('smiles_train.txt', train_smiles),
                                 ('smiles_val.txt',   val_smiles)]:
            with open(os.path.join('data', 'ZINC', fname), 'w') as f:
                f.write('\n'.join(smi_list))

        print(f"Converting {len(train_smiles)} train + {len(val_smiles)} val SMILES → PyG…")
        train_list = [d for s in train_smiles if (d := _smiles_to_pyg_data(s)) is not None]
        val_list   = [d for s in val_smiles   if (d := _smiles_to_pyg_data(s)) is not None]
        print(f"Converted: {len(train_list)} train, {len(val_list)} val molecules.")

        # Save to temp files first, then atomically rename so that a concurrent
        # reader never observes a partially-written cache.
        tmp_train = cache_train + '.tmp'
        tmp_val   = cache_val   + '.tmp'
        torch.save(train_list, tmp_train)
        torch.save(val_list,   tmp_val)
        os.replace(tmp_train, cache_train)
        os.replace(tmp_val,   cache_val)
        # Lock is released when the `with` block exits.
        return train_list, val_list


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
            train_dataset, val_dataset = _build_zinc250k_datasets(seed=config.seed)
            # 18 node slots: 0=pad, 1-17=atom classes (after +1 shift in training)
            # 5 edge slots:  0=no-bond, 1=single, 2=aromatic, 3=double, 4=triple
            num_node_features, num_edge_features = 18, 5
        else:
            train_dataset = MosesPyGDataset(root='data/MOSES', split='train', max_atoms=gc.max_atoms)
            val_dataset = MosesPyGDataset(root='data/MOSES', split='test', max_atoms=gc.max_atoms)
            num_node_features, num_edge_features = 9, 5

        metadata: dict = {'num_nodes': num_node_features, 'num_edges': num_edge_features}

        # Compute class weights for node CE (all GVAE variants).
        # ZINC's carbon class (~70 % of atoms) dominates unweighted CE, causing the
        # decoder to default to all-carbon predictions.  We anchor C at weight=1.0
        # and scale rarer classes by sqrt(cnt_C / cnt_class), capped at 10×.
        # For GVAE_AR/GVAE_AR_NF: append EOS weight=1.0 at index num_node_features.
        # EOS is the only termination signal — boosting it causes premature stops;
        # suppressing it causes infinite loops — keep it neutral.
        if config.model in ('GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF'):
            if hasattr(train_dataset, '_data'):
                raw_x = train_dataset._data.x.squeeze(-1).long() + 1
            else:
                raw_x = torch.cat([d.x.squeeze(-1) for d in train_dataset]).long() + 1
            cnt     = torch.bincount(raw_x, minlength=num_node_features).float()
            present = cnt > 0
            max_cnt = cnt[present].max()                        # count of the most common class (C)
            w       = torch.zeros(num_node_features)
            w[present] = (max_cnt / cnt[present]).sqrt()        # C → 1.0, rare → > 1.0
            w[1:] = w[1:].clamp(max=10.0)                      # cap at 10× to prevent training instability
            w[0] = 1.0  # PAD slot (flat decoder: supervised; AR: unused index)
            if config.model in ('GVAE_AR', 'GVAE_AR_NF'):
                w = torch.cat([w, torch.ones(1)])  # EOS at index num_node_features
            metadata['node_class_weights'] = w
            logger.info(f"Node class weights (\u221a(cnt_C/cnt)) \u2014 "
                        f"C: {w[1]:.3f}  O: {w[2]:.3f}  N: {w[3]:.3f}  "
                        f"max_rare: {w[1:].max():.2f}")

        # Compute edge class weights for all GVAE variants (flat and AR decoder).
        # Within valid atom pairs, ~90 % are no-bond; unweighted CE defaults to
        # predicting no-bond everywhere → disconnected chains instead of ring systems.
        # For AR models this is especially severe: ~90.6 % of all edge tokens in the
        # BFS sequence are class-0 (no-bond), so the AR edge head collapses to always
        # predicting 0, achieving low CE without ever learning actual bond patterns.
        # Identical fix to node weights: anchor no-bond at 1.0, amplify each bond
        # type by sqrt(cnt_no_bond / cnt_bond), capped at 10×.
        if config.model in ('GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF'):
            if hasattr(train_dataset, '_data') and hasattr(train_dataset, 'slices') and 'x' in train_dataset.slices:
                raw_e = train_dataset._data.edge_attr.squeeze(-1).long() + 1
                sizes = (train_dataset.slices['x'][1:] - train_dataset.slices['x'][:-1]).long()
            else:
                raw_e = torch.cat([d.edge_attr.squeeze(-1) for d in train_dataset]).long() + 1
                sizes = torch.tensor([d.x.size(0) for d in train_dataset], dtype=torch.long)
            ebinc   = torch.bincount(raw_e, minlength=num_edge_features).float()[:num_edge_features]
            bcnt    = ebinc / 2.0          # undirected: each bond stored twice
            # Number of valid (real×real) upper-triangle pairs across all training molecules
            n_pairs = float((sizes * (sizes - 1) // 2).sum().item())
            n_no_bond = max(n_pairs - bcnt[1:].sum().item(), 1.0)
            ew      = torch.ones(num_edge_features)
            ew[1:]  = (n_no_bond / bcnt[1:].clamp(min=1.0)).sqrt().clamp(max=10.0)
            ew[0]   = 1.0   # anchor no-bond
            metadata['edge_class_weights'] = ew
            logger.info(f"Edge class weights (\u221a(cnt_no_bond/cnt)) \u2014 "
                        f"no_bond: {ew[0]:.3f}  "
                        + "  ".join(f"bond{i}: {ew[i]:.3f}" for i in range(1, num_edge_features)))

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

        # ── Remap val fragment indices to the training vocabulary ────────────────
        # Each split's build_frattvae_dataset constructs an independent vocabulary
        # (val has 9 431 frags vs train's 30 893 for ZINC).  Val frag_indices are
        # offsets into the val vocab, not the train vocab, so using them with the
        # train frag_ecfps table gives completely wrong ECFP features and wrong
        # cross-entropy targets.  We remap every val molecule's fragment indices
        # to training vocab positions via a dict lookup; unknown validation
        # fragments (not in training vocab) map to slot 0 (= padding / zero-vec).
        train_vocab     = train_data['uni_fragments']
        val_vocab       = val_data['uni_fragments']
        train_frag_idx  = {smi: i for i, smi in enumerate(train_vocab)}
        val_ds_raw      = val_data['dataset']
        new_fi, new_pos, new_prop = [], [], []
        n_unknown_slots = 0
        for i in range(len(val_ds_raw)):
            fi, pos, prop = val_ds_raw[i]
            remapped = torch.tensor(
                [train_frag_idx.get(val_vocab[idx.item()], 0) for idx in fi],
                dtype=torch.long,
            )
            # Count newly-unknown slots (extra zeros beyond original padding)
            n_unknown_slots += int((remapped == 0).sum()) - int((fi == 0).sum())
            new_fi.append(remapped)
            new_pos.append(pos)
            new_prop.append(prop)
        from models.frattvae.dataset import ListDataset as _ListDataset
        val_data['dataset'] = _ListDataset(new_fi, new_pos, torch.stack(new_prop))
        logger.info(
            f"Val vocab remapped to training vocab ({len(train_vocab)} fragments); "
            f"{n_unknown_slots} fragment slots replaced with padding (unknown fragments)."
        )

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
        # The canonical SMILES files are written by _build_zinc250k_datasets().
        # Supported splits: 'train' (95 %) and 'val' (5 %) of ZINC250k.
        cache_path = os.path.join('data', 'ZINC', f'smiles_{split}.txt')
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return [line.strip() for line in f if line.strip()]
        # First call: build/download dataset, which persists the txt files.
        _build_zinc250k_datasets(seed=42)
        if os.path.exists(cache_path):
            with open(cache_path) as f:
                return [line.strip() for line in f if line.strip()]
        raise FileNotFoundError(
            f"ZINC smiles file not found after dataset build: {cache_path}. "
            f"Supported splits are 'train' and 'val'."
        )
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
