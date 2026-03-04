"""
GraphVAE and GraphVAENF model variants with flat MLP decoders.

Both variants share the GINEConvEncoder.  GraphVAENF additionally wraps it
with an Inverse Autoregressive Flow to make the variational posterior more
expressive.

Shared utilities (PropertyHead, decode_to_smiles, valency tables,
gvae_prepare_batch) are kept here because they are also imported by
models/gvae_ar.py and the training code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch, to_dense_adj

import numpy as np
from rdkit import Chem

from models.encoder import GINEConvEncoder
from models.flows import InverseAutoregressiveFlow


# ---------------------------------------------------------------------------
# Optional property prediction head (plogP, QED, SA)
# ---------------------------------------------------------------------------

class PropertyHead(nn.Module):
    """3-layer MLP: latent_dim → 256 → 128 → 3.

    Predicts normalised [plogP, QED, SA] from the posterior mean μ.
    Callers are responsible for denormalising for interpretation.
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 128),        nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """Returns (B, 3) predicted [plogP_z, QED_z, SA_z] in normalised space."""
        return self.net(mu)


# ---------------------------------------------------------------------------
# Valency masking utilities (shared with gvae_ar.py)
# ---------------------------------------------------------------------------

# Maximum total bond order allowed per heavy atom type.
# Charged atoms (+/-) are adjusted in the sampler.
_MAX_VALENCE = {
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
_BOND_ORDER = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1}


def _get_rdkit_bond(bond_idx):
    return {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }.get(bond_idx, Chem.BondType.SINGLE)


# ---------------------------------------------------------------------------
# Flat-decoder greedy SMILES decoding (GVAE / GVAE_NF)
# ---------------------------------------------------------------------------

def decode_to_smiles(node_logits_np, edge_logits_np, max_atoms,
                     atom_decoder_dict, charge_decoder, valency_mask=True):
    """Greedy decode one molecule from flat MLP logits.

    Args:
        node_logits_np : (max_atoms, num_node_features)
        edge_logits_np : (max_atoms, max_atoms, num_edge_features)
        max_atoms      : int
        atom_decoder_dict : {class_idx: atomic_num}
        charge_decoder    : {class_idx: formal_charge} or None
        valency_mask      : mask chemically invalid bond choices
    Returns:
        canonical SMILES string or None
    """
    node_preds = np.argmax(node_logits_np, axis=-1)

    mol = Chem.RWMol()
    node_idx_map = {}
    max_valence  = {}
    running_val  = {}

    for j, atom_idx in enumerate(node_preds):
        if atom_idx == 0:
            continue
        cls = int(atom_idx) - 1
        atomic_num = atom_decoder_dict.get(cls, 6)
        rd_atom    = Chem.Atom(atomic_num)
        fc = 0
        if charge_decoder is not None:
            fc = charge_decoder.get(cls, 0)
            if fc != 0:
                rd_atom.SetFormalCharge(fc)
        idx = mol.AddAtom(rd_atom)
        node_idx_map[j] = idx
        running_val[j]  = 0
        base = _MAX_VALENCE.get(atomic_num, 4)
        if fc > 0:
            base += 1
        elif fc < 0:
            base = max(base - 1, 0)
        max_valence[j] = base

    num_bond_types = edge_logits_np.shape[-1]
    for j in range(max_atoms):
        if j not in node_idx_map:
            continue
        for k in range(j + 1, max_atoms):
            if k not in node_idx_map:
                continue
            logits = edge_logits_np[j, k].copy()
            if valency_mask:
                for b in range(1, num_bond_types):
                    order = _BOND_ORDER.get(b, 1)
                    if (running_val[j] + order > max_valence[j] or
                            running_val[k] + order > max_valence[k]):
                        logits[b] = -np.inf
            bond_idx = int(np.argmax(logits))
            if bond_idx == 0:
                continue
            if not mol.GetBondBetweenAtoms(node_idx_map[j], node_idx_map[k]):
                mol.AddBond(node_idx_map[j], node_idx_map[k], _get_rdkit_bond(bond_idx))
                order = _BOND_ORDER.get(bond_idx, 1)
                running_val[j] += order
                running_val[k] += order

    try:
        result = Chem.SanitizeMol(mol, catchErrors=True)
        if result != Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
            return None
        smi = Chem.MolToSmiles(mol)
        return smi if (smi and Chem.MolFromSmiles(smi) is not None) else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GraphVAE: flat MLP decoder
# ---------------------------------------------------------------------------

class GraphVAE(nn.Module):
    """Standard GraphVAE with a flat MLP decoder.

    Encoder:  GINEConvEncoder (4-layer GINEConv + global_add_pool)
    Decoder:  two MLP heads → node logits (B, N, F_node) and
              edge logits (B, N, N, F_edge)
    """

    def __init__(self, num_node_features: int, num_edge_features: int,
                 latent_dim: int = 128, max_atoms: int = 38,
                 prop_pred: bool = False):
        super().__init__()
        self.max_atoms         = max_atoms
        self.latent_dim        = latent_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        self.encoder   = GINEConvEncoder(num_node_features, num_edge_features,
                                         hidden_dim=256, latent_dim=latent_dim)
        self.prop_head = PropertyHead(latent_dim) if prop_pred else None

        self.decoder_nodes = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, max_atoms * num_node_features),
        )
        self.decoder_edges = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, max_atoms * max_atoms * num_edge_features),
        )

    def encode(self, x, edge_index, edge_attr, batch):
        return self.encoder(x, edge_index, edge_attr, batch)

    def reparameterize(self, mu, logvar):
        return GINEConvEncoder.reparameterize(mu, logvar)

    def decode(self, z):
        B = z.size(0)
        node_logits = self.decoder_nodes(z).view(B, self.max_atoms, self.num_node_features)
        edge_logits = self.decoder_edges(z).view(
            B, self.max_atoms, self.max_atoms, self.num_edge_features)
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        return node_logits, edge_logits

    def forward(self, x, edge_index, edge_attr, batch):
        mu, logvar = self.encode(x, edge_index, edge_attr, batch)
        z = self.reparameterize(mu, logvar)
        node_logits, edge_logits = self.decode(z)
        return node_logits, edge_logits, mu, logvar

    def predict_props(self, mu: torch.Tensor) -> torch.Tensor:
        if self.prop_head is None:
            raise RuntimeError("GraphVAE was built without prop_pred=True")
        return self.prop_head(mu)

    def sample_smiles(self, z, atom_decoder_dict={}, charge_decoder=None,
                      valency_mask=True):
        node_logits, edge_logits = self.decode(z)
        node_np = node_logits.detach().cpu().float().numpy()
        edge_np = edge_logits.detach().cpu().float().numpy()
        return [
            decode_to_smiles(node_np[i], edge_np[i], self.max_atoms,
                             atom_decoder_dict, charge_decoder, valency_mask)
            for i in range(z.size(0))
        ]


# ---------------------------------------------------------------------------
# GraphVAENF: flat MLP decoder + IAF normalizing flow
# ---------------------------------------------------------------------------

class GraphVAENF(nn.Module):
    """GraphVAE with an Inverse Autoregressive Flow in the encoder path.

    The flow makes the variational posterior q(z|x) more expressive.
    Prior:  p(zK) = N(0, I).  During sampling, draw z ~ N(0, I) and decode
    directly — do NOT pass z through the flow.
    """

    def __init__(self, num_node_features: int, num_edge_features: int,
                 latent_dim: int = 128, max_atoms: int = 38,
                 num_flows: int = 4, flow_hidden_dim: int = 256,
                 prop_pred: bool = False):
        super().__init__()
        self.max_atoms         = max_atoms
        self.latent_dim        = latent_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        self.encoder = GINEConvEncoder(num_node_features, num_edge_features,
                                       hidden_dim=256, latent_dim=latent_dim)
        self.flow    = InverseAutoregressiveFlow(latent_dim, num_flows=num_flows,
                                                hidden_dim=flow_hidden_dim)

        self.prop_head = PropertyHead(latent_dim) if prop_pred else None

        self.decoder_nodes = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, max_atoms * num_node_features),
        )
        self.decoder_edges = nn.Sequential(
            nn.Linear(latent_dim, 512), nn.ReLU(),
            nn.Linear(512, max_atoms * max_atoms * num_edge_features),
        )

    def encode(self, x, edge_index, edge_attr, batch):
        return self.encoder(x, edge_index, edge_attr, batch)

    def reparameterize(self, mu, logvar):
        return GINEConvEncoder.reparameterize(mu, logvar)

    def decode(self, z):
        B = z.size(0)
        node_logits = self.decoder_nodes(z).view(B, self.max_atoms, self.num_node_features)
        edge_logits = self.decoder_edges(z).view(
            B, self.max_atoms, self.max_atoms, self.num_edge_features)
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        return node_logits, edge_logits

    def forward(self, x, edge_index, edge_attr, batch):
        mu, logvar = self.encode(x, edge_index, edge_attr, batch)
        z0 = self.reparameterize(mu, logvar)
        zK, sum_log_det = self.flow(z0)
        node_logits, edge_logits = self.decode(zK)
        return node_logits, edge_logits, mu, logvar, z0, zK, sum_log_det

    def predict_props(self, mu: torch.Tensor) -> torch.Tensor:
        """Predict from μ₀ (pre-flow).

        μ is used instead of zK because:
        - It is a stable, low-variance target the encoder can cleanly optimise.
        - Routing property gradients through the flow layers distorts the
          flow's N(0, I) calibration and collapses validity.
        """
        if self.prop_head is None:
            raise RuntimeError("GraphVAENF was built without prop_pred=True")
        return self.prop_head(mu)

    def sample_smiles(self, z, atom_decoder_dict={}, charge_decoder=None,
                      valency_mask=True):
        # z ~ N(0, I) is already zK — the prior is p(zK) = N(0, I).
        # Do NOT pass z through the flow; that maps N(0,I) → flow(N(0,I))
        # which is out-of-distribution for the decoder.
        node_logits, edge_logits = self.decode(z)
        node_np = node_logits.detach().cpu().float().numpy()
        edge_np = edge_logits.detach().cpu().float().numpy()
        return [
            decode_to_smiles(node_np[i], edge_np[i], self.max_atoms,
                             atom_decoder_dict, charge_decoder, valency_mask)
            for i in range(z.size(0))
        ]


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def _flat_recon_loss(node_logits, edge_logits, target_nodes, target_edges):
    """Shared reconstruction CE loss for both flat-decoder variants."""
    batch_size, N = target_nodes.shape

    node_ce = F.cross_entropy(
        node_logits.reshape(-1, node_logits.size(-1)),
        target_nodes.reshape(-1),
        ignore_index=0, reduction='none',
    ).view(batch_size, N)
    recon_nodes = node_ce.sum(dim=1).mean()

    valid_mask      = (target_nodes > 0).float()
    valid_pair_mask = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)
    triu = torch.triu_indices(N, N, offset=1)

    edge_logits_triu = edge_logits[:, triu[0], triu[1], :]
    target_triu      = target_edges[:, triu[0], triu[1]].clone()
    pair_mask_triu   = valid_pair_mask[:, triu[0], triu[1]]
    target_triu[pair_mask_triu == 0] = 0

    edge_ce = F.cross_entropy(
        edge_logits_triu.reshape(-1, edge_logits_triu.size(-1)),
        target_triu.reshape(-1), reduction='none',
    ).view(batch_size, -1)
    edge_ce = edge_ce * pair_mask_triu.view(batch_size, -1)
    recon_edges = edge_ce.sum(dim=1).mean()

    return recon_nodes + recon_edges


def gvae_loss(node_logits, edge_logits, target_nodes, target_edges,
              mu, logvar, kl_weight, free_bits: float = 0.0, capacity: float = 0.0):
    """Standard VAE loss for GraphVAE.  Returns (total, recon, kl).

    free_bits: minimum KL per latent dimension (nats).  Prevents collapse by
      ensuring no dimension contributes less than this to the KL term.
    capacity:  target KL (nats).  Loss = kl_weight * |KL - capacity|, so the
      model is penalised equally for being above or below the target.
    """
    recon = _flat_recon_loss(node_logits, edge_logits, target_nodes, target_edges)
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
    if free_bits > 0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl = kl_per_dim.sum(dim=1).mean()
    return recon + kl_weight * (kl - capacity).abs(), recon, kl


def gvae_nf_loss(node_logits, edge_logits, target_nodes, target_edges,
                 mu, logvar, z0, zK, sum_log_det, kl_weight, capacity: float = 0.0):
    """IAF-VAE loss for GraphVAENF.

    KL is computed via the change-of-variables formula:
        KL = E[log q(z0|x) - log p(zK)] - E[sum_log_det]

    capacity: target KL (nats).  Loss = kl_weight * |KL - capacity|.
      Free bits are not applied here because KL is not a sum of independent
      per-dimension terms once the flow is involved.

    Returns (total, recon, kl_flow).
    """
    recon   = _flat_recon_loss(node_logits, edge_logits, target_nodes, target_edges)
    std     = (0.5 * logvar).exp()
    log_q0  = -0.5 * (logvar + ((z0 - mu) / (std + 1e-8)).pow(2))
    log_pK  = -0.5 * zK.pow(2)
    kl_flow = (log_q0 - log_pK).sum(dim=1).mean() - sum_log_det.mean()
    return recon + kl_weight * (kl_flow - capacity).abs(), recon, kl_flow


# ---------------------------------------------------------------------------
# Batch preparation (shared with training code and gvae_ar.py)
# ---------------------------------------------------------------------------

def gvae_prepare_batch(data, device, max_atoms):
    """Move a PyG batch to device and build dense target tensors."""
    data = data.to(device)
    x_in         = data.x.squeeze(-1) + 1
    edge_attr_in = data.edge_attr.squeeze(-1) + 1
    target_nodes, _ = to_dense_batch(x_in, data.batch, max_num_nodes=max_atoms)
    target_edges = to_dense_adj(
        data.edge_index, data.batch,
        edge_attr=edge_attr_in, max_num_nodes=max_atoms,
    ).squeeze(-1).long()
    return x_in, data.edge_index, edge_attr_in, data.batch, target_nodes, target_edges
