import os
import random
import argparse
from dataclasses import dataclass
import numpy as np
import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
from tqdm import tqdm

from models.vae import MolecularGraphVAE, vae_loss
from moses_dataset import MosesPyGDataset

@dataclass
class Config:
    dataset: str = 'ZINC'
    batch_size: int = 128
    epochs: int = 100
    lr: float = 1e-3
    max_atoms: int = 38
    seed: int = 42
    num_node_features: int = 29
    num_edge_features: int = 5
    latent_dim: int = 64
    kl_weight: float = 0.1

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train Molecular Graph VAE")
    parser.add_argument('--dataset', type=str, default='ZINC', choices=['ZINC', 'MOSES'], 
                        help="Dataset to train on ('ZINC' or 'MOSES'). Default: ZINC")
    parser.add_argument('--batch_size', type=int, default=128, help="Batch size for training. Default: 128")
    parser.add_argument('--epochs', type=int, default=100, help="Number of training epochs. Default: 100")
    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate. Default: 1e-3")
    parser.add_argument('--max_atoms', type=int, default=38, help="Maximum number of atoms per molecule. Default: 38")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility. Default: 42")
    parser.add_argument('--num_node_features', type=int, default=29, help="Number of node features. Default: 29")
    parser.add_argument('--num_edge_features', type=int, default=5, help="Number of edge features. Default: 5")
    parser.add_argument('--latent_dim', type=int, default=64, help="Dimension of the latent space. Default: 64")
    parser.add_argument('--kl_weight', type=float, default=0.1, help="Weight of the KL divergence loss. Default: 0.1")
    
    args = parser.parse_args()
    return Config(**vars(args))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_batch(data, device, max_atoms):
    data = data.to(device)
    
    # Shift labels by +1 so 0 is reserved for padding / no bond
    x_in = data.x.squeeze(-1) + 1
    edge_attr_in = data.edge_attr.squeeze(-1) + 1
    
    # Dense Targets
    target_nodes, _ = to_dense_batch(x_in, data.batch, max_num_nodes=max_atoms)
    target_edges = to_dense_adj(data.edge_index, data.batch, edge_attr=edge_attr_in, max_num_nodes=max_atoms)
    target_edges = target_edges.squeeze(-1).long()
    
    return x_in, data.edge_index, edge_attr_in, data.batch, target_nodes, target_edges

def train_epoch(model, optimizer, loader, config, device):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    for data in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = prepare_batch(data, device, config.max_atoms)
        
        node_logits, edge_logits, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
        loss, recon, kl = vae_loss(node_logits, edge_logits, target_nodes, target_edges, mu, logvar, config.kl_weight)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl += kl.item() * data.num_graphs
        
    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kl/n

@torch.no_grad()
def val_epoch(model, loader, config, device):
    model.eval()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    for data in tqdm(loader, desc="Validation", leave=False):
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = prepare_batch(data, device, config.max_atoms)
        node_logits, edge_logits, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
        loss, recon, kl = vae_loss(node_logits, edge_logits, target_nodes, target_edges, mu, logvar, config.kl_weight)
        
        total_loss += loss.item() * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl += kl.item() * data.num_graphs
        
    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kl/n

def train(config: Config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading {config.dataset} dataset...")
    if config.dataset == 'ZINC':
        train_dataset = ZINC(root='data/ZINC', subset=False, split='train')
        val_dataset = ZINC(root='data/ZINC', subset=False, split='val')
    else:
        train_dataset = MosesPyGDataset(root='data/MOSES', split='train', max_atoms=config.max_atoms)
        val_dataset = MosesPyGDataset(root='data/MOSES', split='test', max_atoms=config.max_atoms)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    model = MolecularGraphVAE(
        num_node_features=config.num_node_features, 
        num_edge_features=config.num_edge_features, 
        latent_dim=config.latent_dim, 
        max_atoms=config.max_atoms
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    best_val_loss = float('inf')

    print(f"Starting training on {device}...")
    for epoch in range(1, config.epochs + 1):
        train_l, train_r, train_kl = train_epoch(model, optimizer, train_loader, config, device)
        val_l, val_r, val_kl = val_epoch(model, val_loader, config, device)
        
        print(f"Epoch {epoch:03d}")
        print(f"Train Loss: {train_l:.4f} | Recon: {train_r:.4f} | KL: {train_kl:.4f}")
        print(f"Val Loss: {val_l:.4f} | Recon: {val_r:.4f} | KL: {val_kl:.4f}")
        
        if val_l < best_val_loss:
            best_val_loss = val_l
            torch.save(model.state_dict(), 'checkpoints/vae_best.pth')
            print("Model checkpoint saved.")
        print("-" * 50)

if __name__ == "__main__":
    config = parse_args()
    set_seed(config.seed)
    os.makedirs('checkpoints', exist_ok=True)
    train(config)
