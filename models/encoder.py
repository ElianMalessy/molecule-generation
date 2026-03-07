import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool


class GINEConvEncoder(nn.Module):
    """4-layer GINEConv graph encoder → (mu, logvar) in latent space.

    Parameters
    ----------
    num_node_features : int  – number of node class indices (after +1 shift)
    num_edge_features : int  – number of edge class indices (after +1 shift)
    hidden_dim        : int  – message-passing hidden dimension (default 256)
    latent_dim        : int  – VAE latent dimension (default 128)
    """

    def __init__(self, num_node_features: int, num_edge_features: int,
                 hidden_dim: int = 256, latent_dim: int = 128):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.node_emb = nn.Embedding(num_node_features, hidden_dim)
        self.edge_emb = nn.Embedding(num_edge_features, hidden_dim)

        def _mlp():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )

        self.conv1 = GINEConv(_mlp())
        self.conv2 = GINEConv(_mlp())
        self.conv3 = GINEConv(_mlp())
        self.conv4 = GINEConv(_mlp())

        self.fc_mu     = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        # Zero-init both heads so at epoch 1: mu=0, logvar=0 → q(z|x) = N(0,I),
        # KL = 0 nats.  Eliminates the epoch-1 KL spike from Kaiming-init weights
        # driving large mu via global_add_pool (which sums over all atoms).
        # fc_mu diverges from 0 as recon loss forces the encoder to encode structure;
        # fc_logvar diverges as the capacity ramp gives the encoder KL budget to use.
        nn.init.zeros_(self.fc_mu.weight)
        nn.init.zeros_(self.fc_mu.bias)
        nn.init.zeros_(self.fc_logvar.weight)
        nn.init.zeros_(self.fc_logvar.bias)

    def forward(self, x, edge_index, edge_attr, batch):
        """Encode a batch of graphs → (mu, logvar) each (B, latent_dim)."""
        x_emb = self.node_emb(x)
        e_emb = self.edge_emb(edge_attr)

        h = F.relu(self.conv1(x_emb, edge_index, e_emb))
        h = F.relu(self.conv2(h,     edge_index, e_emb))
        h = F.relu(self.conv3(h,     edge_index, e_emb))
        h = F.relu(self.conv4(h,     edge_index, e_emb))

        h_graph = global_add_pool(h, batch)
        # Clamp logvar so std = exp(0.5*logvar) stays in [exp(-2.5), exp(2.5)] ≈ [0.08, 12].
        # Combined with zero-init above this is only a safety bound; prevents IAF
        # MADE overflow on bfloat16 if activations ever drift large mid-training.
        return self.fc_mu(h_graph), self.fc_logvar(h_graph).clamp(-5, 5)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Sample z = mu + eps * sigma using the reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)
