import torch
import torch.nn as torch_nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_add_pool

class MolecularGraphVAE(torch_nn.Module):
    def __init__(self, num_node_features, num_edge_features, latent_dim=64, max_atoms=38):
        super(MolecularGraphVAE, self).__init__()
        
        self.max_atoms = max_atoms
        self.latent_dim = latent_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features 
        
        hidden_dim = 128
        
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


def vae_loss(node_logits, edge_logits, target_nodes, target_edges, mu, logvar, kl_weight=0.1):
    # Cross entropy for reconstruction because we are predicting discrete categories for both nodes and edges
    # 1. Node Loss (Ignore empty nodes with index 0, do not reward the model for predicting empty slots because that takes up most of the matrix)
    recon_loss_nodes = F.cross_entropy(
        node_logits.reshape(-1, node_logits.size(-1)), 
        target_nodes.reshape(-1), 
        ignore_index=0,
        reduction='mean'
    )
    
    
    _, N = target_nodes.shape

    # Create mask for valid node pairs (1 if both nodes are valid, 0 if either is padding)
    # We only care about a bond if both atoms in the pair actually exist.
    valid_mask = (target_nodes > 0).float() 
    valid_pair_mask = valid_mask.unsqueeze(2) * valid_mask.unsqueeze(1) # (B, N, N)
    
    # Get upper triangular indices (offset=1 ignores self-loops)
    triu_idx = torch.triu_indices(N, N, offset=1)
    
    # Extract upper triangular elements
    edge_logits_triu = edge_logits[:, triu_idx[0], triu_idx[1], :] # (B, num_pairs, num_edge_features)
    target_edges_triu = target_edges[:, triu_idx[0], triu_idx[1]].clone() # (B, num_pairs)
    valid_pair_mask_triu = valid_pair_mask[:, triu_idx[0], triu_idx[1]]
    
    # Map padded pairs to -1 (not 0 because 0 is a valid "no bond" class)
    target_edges_triu[valid_pair_mask_triu == 0] = -1
    
    # 2. Edge Loss (Upper-triangular & masked padding, ignore pairs where either node is padding)
    recon_loss_edges = F.cross_entropy(
        edge_logits_triu.reshape(-1, edge_logits_triu.size(-1)), 
        target_edges_triu.reshape(-1), 
        ignore_index=-1,
        reduction='mean'
    )
    
    recon_loss = recon_loss_nodes + recon_loss_edges
    
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    kl_div = torch.mean(kl_div)
    
    total_loss = recon_loss + (kl_weight * kl_div)
    
    return total_loss, recon_loss, kl_div
