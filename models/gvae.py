import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils import to_dense_batch, to_dense_adj

import numpy as np
from rdkit import Chem

# ---------------------------------------------------------------------------
# Optional property prediction head (plogP, QED, SA)
# ---------------------------------------------------------------------------

class PropertyHead(torch_nn.Module):
    """
    3-layer MLP that predicts a vector of 3 molecular properties from the
    posterior mean μ.  Predicts in the *normalised* space; callers are
    responsible for denormalising for interpretation.

    Architecture: latent_dim → 256 → 128 → 3
    """
    def __init__(self, latent_dim: int):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_nn.Linear(latent_dim, 256),
            torch_nn.ReLU(),
            torch_nn.Linear(256, 128),
            torch_nn.ReLU(),
            torch_nn.Linear(128, 3),
        )

    def forward(self, mu: torch.Tensor) -> torch.Tensor:
        """Returns (B, 3) predicted [plogP_z, QED_z, SA_z] in normalised space."""
        return self.net(mu)

# Maximum total bond order allowed per heavy/light atom type.
# For atoms that can be charged (e.g. N+, O-) we adjust in the sampler below.
_MAX_VALENCE = {
    1:  1,   # H
    6:  4,   # C
    7:  3,   # N  (N+ handled separately → 4)
    8:  2,   # O  (O- → 1)
    9:  1,   # F
    15: 5,   # P
    16: 6,   # S
    17: 1,   # Cl
    35: 1,   # Br
    53: 1,   # I
}

# How many valence units each bond-type index consumes.
# Aromatic bonds are treated as 1.5 rounded to 1 for integer tracking.
_BOND_ORDER = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1}


def _get_rdkit_bond(bond_idx):
    mapping = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
        4: Chem.BondType.AROMATIC,
    }
    return mapping.get(bond_idx, Chem.BondType.SINGLE)


def decode_to_smiles(node_logits_np, edge_logits_np, max_atoms,
                    atom_decoder_dict, charge_decoder, valency_mask=True):
    """
    Greedy decoding of one molecule with an explicit valency mask.

    1. Argmax node_logits → atom types.
    2. Iterate over upper-triangle atom pairs in row-major order.
       For each pair, mask any bond type whose order would push either atom
       over its maximum valence before taking argmax.
    3. Build RDKit mol, sanitize, return canonical SMILES or None.

    Args:
        node_logits_np: (max_atoms, num_node_features)  float numpy array
        edge_logits_np: (max_atoms, max_atoms, num_edge_features) float numpy array
        max_atoms:       int
        atom_decoder_dict: {class_idx: atomic_num}
        charge_decoder:    {class_idx: formal_charge} or None
    """
    node_preds = np.argmax(node_logits_np, axis=-1)   # (max_atoms,)

    mol = Chem.RWMol()
    node_idx_map   = {}   # grid-idx → mol atom idx
    max_valence    = {}   # grid-idx → max allowed valence
    running_val    = {}   # grid-idx → current valence used

    # --- Place atoms ---
    for j, atom_idx in enumerate(node_preds):
        if atom_idx == 0:
            continue                        # padding
        original_class = int(atom_idx) - 1
        atomic_num = atom_decoder_dict.get(original_class, 6)
        rd_atom    = Chem.Atom(atomic_num)
        fc = 0
        if charge_decoder is not None:
            fc = charge_decoder.get(original_class, 0)
            if fc != 0:
                rd_atom.SetFormalCharge(fc)

        idx = mol.AddAtom(rd_atom)
        node_idx_map[j] = idx
        running_val[j]  = 0
        base = _MAX_VALENCE.get(atomic_num, 4)
        # Positive formal charge opens an extra valence slot (e.g. N+ → 4);
        # negative charge closes one (e.g. O- → 1).
        if fc > 0:
            base = base + 1
        elif fc < 0:
            base = max(base - 1, 0)
        max_valence[j] = base

    # --- Place bonds with valency masking ---
    num_bond_types = edge_logits_np.shape[-1]
    for j in range(max_atoms):
        if j not in node_idx_map:
            continue
        for k in range(j + 1, max_atoms):
            if k not in node_idx_map:
                continue

            logits = edge_logits_np[j, k].copy()   # (num_bond_types,)

            # Mask bond types that would violate valency for either atom.
            if valency_mask:
                for b in range(1, num_bond_types):     # b=0 is "no bond", never masked
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

    # --- Sanitize ---
    try:
        result = Chem.SanitizeMol(mol, catchErrors=True)
        if result != Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
            return None
        smi = Chem.MolToSmiles(mol)
        return smi if (smi and Chem.MolFromSmiles(smi) is not None) else None
    except Exception:
        return None

class GraphVAE(torch_nn.Module):
    def __init__(self, num_node_features, num_edge_features, latent_dim=64, max_atoms=38,
                 prop_pred: bool = False):
        super(GraphVAE, self).__init__()
        
        self.max_atoms = max_atoms
        self.latent_dim = latent_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features 
        
        hidden_dim = 256
        
        # Encoder
        # Embeddings for nodes and edges
        self.node_emb = torch_nn.Embedding(num_node_features, hidden_dim)
        self.edge_emb = torch_nn.Embedding(num_edge_features, hidden_dim)
        
        def build_mlp():
            return torch_nn.Sequential(
                torch_nn.Linear(hidden_dim, hidden_dim),
                torch_nn.ReLU(),
                torch_nn.Linear(hidden_dim, hidden_dim)
            )

        # GINEConv allows us to incorporate edge attributes
        self.conv1 = GINEConv(build_mlp())
        self.conv2 = GINEConv(build_mlp())
        self.conv3 = GINEConv(build_mlp())
        self.conv4 = GINEConv(build_mlp())
        
        self.fc_mu = torch_nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = torch_nn.Linear(hidden_dim, latent_dim)
        
        # Optional property prediction head
        self.prop_head = PropertyHead(latent_dim) if prop_pred else None
        
        # Decoder
        self.decoder_nodes = torch_nn.Sequential(
            torch_nn.Linear(latent_dim, 256),
            torch_nn.ReLU(),
            torch_nn.Linear(256, max_atoms * num_node_features)
        )
        
        self.decoder_edges = torch_nn.Sequential(
            torch_nn.Linear(latent_dim, 512),
            torch_nn.ReLU(),
            torch_nn.Linear(512, max_atoms * max_atoms * num_edge_features)
        )

    def encode(self, x, edge_index, edge_attr, batch):
        x_emb = self.node_emb(x)
        e_emb = self.edge_emb(edge_attr)
        
        h = F.relu(self.conv1(x_emb, edge_index, e_emb))
        h = F.relu(self.conv2(h, edge_index, e_emb))
        h = F.relu(self.conv3(h, edge_index, e_emb))
        h = F.relu(self.conv4(h, edge_index, e_emb))
        
        h_graph = global_add_pool(h, batch)
        
        mu = self.fc_mu(h_graph)
        logvar = self.fc_logvar(h_graph)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        batch_size = z.size(0)
        
        node_logits = self.decoder_nodes(z)
        node_logits = node_logits.view(batch_size, self.max_atoms, self.num_node_features)
        
        edge_logits = self.decoder_edges(z)
        edge_logits = edge_logits.view(batch_size, self.max_atoms, self.max_atoms, self.num_edge_features)
        
        # Average logits for undirected edges to ensure symmetry
        edge_logits = (edge_logits + edge_logits.transpose(1, 2)) / 2
        
        return node_logits, edge_logits

    def forward(self, x, edge_index, edge_attr, batch):
        mu, logvar = self.encode(x, edge_index, edge_attr, batch)
        z = self.reparameterize(mu, logvar)
        node_logits, edge_logits = self.decode(z)
        return node_logits, edge_logits, mu, logvar

    def predict_props(self, mu: torch.Tensor) -> torch.Tensor:
        """
        Predict normalised property vector (B, 3) from posterior mean μ.
        Raises RuntimeError if the model was built without prop_pred=True.
        """
        if self.prop_head is None:
            raise RuntimeError("GraphVAE was built without prop_pred=True")
        return self.prop_head(mu)

    def sample_smiles(self, z, atom_decoder_dict={}, charge_decoder=None, valency_mask=True):
        node_logits, edge_logits = self.decode(z)
        node_np = node_logits.detach().cpu().float().numpy()
        edge_np = edge_logits.detach().cpu().float().numpy()

        return [
            decode_to_smiles(node_np[i], edge_np[i], self.max_atoms,
                             atom_decoder_dict, charge_decoder, valency_mask)
            for i in range(z.size(0))
        ]


def gvae_loss(node_logits, edge_logits, target_nodes, target_edges, mu, logvar, kl_weight):
    batch_size, N = target_nodes.shape

    # 1. Node Loss (Per Graph Sum)
    node_ce = F.cross_entropy(
        node_logits.reshape(-1, node_logits.size(-1)), 
        target_nodes.reshape(-1), 
        ignore_index=0,
        reduction='none' # CRITICAL: Do not mean yet
    ).view(batch_size, N)
    recon_loss_nodes = node_ce.sum(dim=1).mean() # Sum over nodes, mean over batch
    
    # 2. Edge Loss Setup
    valid_mask = (target_nodes > 0).float() 
    valid_pair_mask = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1) 
    
    triu_idx = torch.triu_indices(N, N, offset=1)
    edge_logits_triu = edge_logits[:, triu_idx[0], triu_idx[1], :] 
    target_edges_triu = target_edges[:, triu_idx[0], triu_idx[1]].clone() 
    valid_pair_mask_triu = valid_pair_mask[:, triu_idx[0], triu_idx[1]]
    
    target_edges_triu[valid_pair_mask_triu == 0] = 0

    # 3. Edge Loss (Per Graph Sum)
    edge_ce = F.cross_entropy(
        edge_logits_triu.reshape(-1, edge_logits_triu.size(-1)), 
        target_edges_triu.reshape(-1), 
        reduction='none'
    ).view(batch_size, -1)
    # Mask out padding pairs (where either atom is padding) before summing
    edge_ce = edge_ce * valid_pair_mask_triu.view(batch_size, -1)
    recon_loss_edges = edge_ce.sum(dim=1).mean() # Sum over valid edges, mean over batch
    
    recon_loss = recon_loss_nodes + recon_loss_edges
    
    # KL Divergence 
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_div = kl_div.mean() # Mean over batch
    
    total_loss = recon_loss + (kl_weight * kl_div)
    
    return total_loss, recon_loss, kl_div


def gvae_prepare_batch(data, device, max_atoms):
    data = data.to(device)
    
    # Shift labels by +1 so 0 is reserved for padding / no bond
    x_in = data.x.squeeze(-1) + 1
    edge_attr_in = data.edge_attr.squeeze(-1) + 1
    
    # Dense Targets
    target_nodes, _ = to_dense_batch(x_in, data.batch, max_num_nodes=max_atoms)
    target_edges = to_dense_adj(data.edge_index, data.batch, edge_attr=edge_attr_in, max_num_nodes=max_atoms)
    target_edges = target_edges.squeeze(-1).long()
    
    return x_in, data.edge_index, edge_attr_in, data.batch, target_nodes, target_edges
