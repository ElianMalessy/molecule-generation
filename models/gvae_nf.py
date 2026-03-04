import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool
from torch_geometric.utils import to_dense_adj, to_dense_batch

from models.gvae import decode_to_smiles, PropertyHead


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features, bias)
        self.register_buffer('mask', torch.ones(out_features, in_features))

    def set_mask(self, mask):
        self.mask.data.copy_(mask)

    def forward(self, x):
        return F.linear(x, self.weight * self.mask, self.bias)


class MADE(nn.Module):
    """2-layer MADE that outputs (mean, log_scale) autoregressively."""

    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.lin1 = MaskedLinear(dim, hidden_dim)
        self.lin2 = MaskedLinear(hidden_dim, dim * 2)

        input_order  = torch.arange(1, dim + 1)
        hidden_order = torch.arange(hidden_dim) % max(dim - 1, 1) + 1
        output_order = input_order.repeat(2)

        mask1 = (hidden_order.unsqueeze(1) >= input_order.unsqueeze(0)).float()
        mask2 = (output_order.unsqueeze(1) > hidden_order.unsqueeze(0)).float()
        self.lin1.set_mask(mask1)
        self.lin2.set_mask(mask2)

    def forward(self, z):
        h = F.relu(self.lin1(z))
        m, s = self.lin2(h).chunk(2, dim=-1)
        return m, s


class IAFStep(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 256):
        super().__init__()
        self.made = MADE(dim, hidden_dim)

    def forward(self, z):
        m, s = self.made(z)
        gate = torch.sigmoid(s)
        z_new = gate * z + (1.0 - gate) * m
        log_det = torch.log(gate + 1e-8).sum(dim=-1)
        return z_new, log_det


class InverseAutoregressiveFlow(nn.Module):
    def __init__(self, dim: int, num_flows: int = 4, hidden_dim: int = 256):
        super().__init__()
        self.flows = nn.ModuleList([IAFStep(dim, hidden_dim) for _ in range(num_flows)])

    def forward(self, z: torch.Tensor):
        sum_log_det = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        for flow in self.flows:
            z, ld = flow(z)
            sum_log_det = sum_log_det + ld
        return z, sum_log_det


class GraphVAENF(nn.Module):
    def __init__(self, num_node_features, num_edge_features,
                 latent_dim=128, max_atoms=38, num_flows=4, flow_hidden_dim=256,
                 prop_pred: bool = False):
        super().__init__()

        self.max_atoms = max_atoms
        self.latent_dim = latent_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

        hidden_dim = 256

        # --- Encoder (identical to GraphVAE) ---
        self.node_emb = nn.Embedding(num_node_features, hidden_dim)
        self.edge_emb = nn.Embedding(num_edge_features, hidden_dim)

        def build_mlp():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.conv1 = GINEConv(build_mlp())
        self.conv2 = GINEConv(build_mlp())
        self.conv3 = GINEConv(build_mlp())
        self.conv4 = GINEConv(build_mlp())

        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        self.flow = InverseAutoregressiveFlow(latent_dim, num_flows=num_flows, hidden_dim=flow_hidden_dim)

        # Optional property prediction head
        self.prop_head = PropertyHead(latent_dim) if prop_pred else None

        # --- Decoder (identical to GraphVAE) ---
        self.decoder_nodes = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, max_atoms * num_node_features),
        )
        self.decoder_edges = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, max_atoms * max_atoms * num_edge_features),
        )


    def encode(self, x, edge_index, edge_attr, batch):
        x_emb = self.node_emb(x)
        e_emb = self.edge_emb(edge_attr)

        h = F.relu(self.conv1(x_emb, edge_index, e_emb))
        h = F.relu(self.conv2(h,     edge_index, e_emb))
        h = F.relu(self.conv3(h,     edge_index, e_emb))
        h = F.relu(self.conv4(h,     edge_index, e_emb))

        h_graph = global_add_pool(h, batch)
        return self.fc_mu(h_graph), self.fc_logvar(h_graph)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

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
        """
        Predict normalised property vector (B, 3) from the posterior mean μ.

        We intentionally use μ (pre-flow) rather than zK = flow(z0), because:
        - μ is a stable, low-variance target that the encoder can cleanly learn
          to make property-predictive.
        - Routing property gradients through the stochastic z0 = μ + ε·σ path
          (or through the flow layers) distorts the flow's N(0,I) calibration
          and collapses generation validity.
        - The property head is a training-time regulariser for the encoder;
          it does not need to operate in decoder (zK) space.

        Raises RuntimeError if the model was built without prop_pred=True.
        """
        if self.prop_head is None:
            raise RuntimeError("GraphVAENF was built without prop_pred=True")
        return self.prop_head(mu)

    def sample_smiles(self, z, atom_decoder_dict={}, charge_decoder=None, valency_mask=True):
        zK, _ = self.flow(z)
        node_logits, edge_logits = self.decode(zK)
        node_np = node_logits.detach().cpu().float().numpy()
        edge_np = edge_logits.detach().cpu().float().numpy()

        return [
            decode_to_smiles(node_np[i], edge_np[i], self.max_atoms,
                             atom_decoder_dict, charge_decoder, valency_mask)
            for i in range(z.size(0))
        ]


def gvae_nf_loss(node_logits, edge_logits, target_nodes, target_edges,
                 mu, logvar, z0, zK, sum_log_det, kl_weight):
    batch_size, N = target_nodes.shape

    node_ce = F.cross_entropy(
        node_logits.reshape(-1, node_logits.size(-1)),
        target_nodes.reshape(-1),
        ignore_index=0,
        reduction='none',
    ).view(batch_size, N)
    recon_nodes = node_ce.sum(dim=1).mean()

    valid_mask = (target_nodes > 0).float()
    valid_pair_mask = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1)

    triu = torch.triu_indices(N, N, offset=1)
    edge_logits_triu = edge_logits[:, triu[0], triu[1], :]
    target_triu      = target_edges[:, triu[0], triu[1]].clone()
    pair_mask_triu   = valid_pair_mask[:, triu[0], triu[1]]
    target_triu[pair_mask_triu == 0] = 0

    edge_ce = F.cross_entropy(
        edge_logits_triu.reshape(-1, edge_logits_triu.size(-1)),
        target_triu.reshape(-1),
        reduction='none',
    ).view(batch_size, -1)
    edge_ce = edge_ce * pair_mask_triu.view(batch_size, -1)
    recon_edges = edge_ce.sum(dim=1).mean()

    recon_loss = recon_nodes + recon_edges

    std = (0.5 * logvar).exp()
    log_q0 = -0.5 * (logvar + ((z0 - mu) / (std + 1e-8)).pow(2))
    log_pK = -0.5 * zK.pow(2)

    kl_flow = (log_q0 - log_pK).sum(dim=1).mean() - sum_log_det.mean()

    total_loss = recon_loss + kl_weight * kl_flow
    return total_loss, recon_loss, kl_flow