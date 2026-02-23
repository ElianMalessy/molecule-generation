import os
import random
import numpy as np
import torch
from torch_geometric.datasets import ZINC
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
from tqdm import tqdm

from models.vae import MolecularGraphVAE, vae_loss

BATCH_SIZE = 128
EPOCHS = 50
LR = 1e-3
MAX_ATOMS = 38
SEED = 42

# ZINC: 28 atom types (0-27) + 1 for padding (0) = 29. We shift all atom types by +1 so 0 can represent "no atom" for padding purposes.
NUM_NODE_FEATURES = 29 
# ZINC bonds: single, double, triple, aromatic (0-3) + 1 for no bond (0) = 5. We shift all bond types by +1 so 0 can represent "no bond" for padding purposes. This is important for the loss function to ignore non-existent edges.
NUM_EDGE_FEATURES = 5  

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_batch(data, device):
    data = data.to(device)
    
    # Shift labels by +1 so 0 is reserved for padding / no bond
    x_in = data.x.squeeze(-1) + 1
    edge_attr_in = data.edge_attr.squeeze(-1) + 1
    
    # Dense Targets
    target_nodes, _ = to_dense_batch(x_in, data.batch, max_num_nodes=MAX_ATOMS)
    target_edges = to_dense_adj(data.edge_index, data.batch, edge_attr=edge_attr_in, max_num_nodes=MAX_ATOMS)
    target_edges = target_edges.squeeze(-1).long()
    
    return x_in, data.edge_index, edge_attr_in, data.batch, target_nodes, target_edges

def train_epoch(model, optimizer, loader, kl_weight, device):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    for data in tqdm(loader, desc="Training", leave=False):
        optimizer.zero_grad()
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = prepare_batch(data, device)
        
        node_logits, edge_logits, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
        loss, recon, kl = vae_loss(node_logits, edge_logits, target_nodes, target_edges, mu, logvar, kl_weight)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl += kl.item() * data.num_graphs
        
    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kl/n

@torch.no_grad()
def val_epoch(model, loader, kl_weight, device):
    model.eval()
    total_loss, total_recon, total_kl = 0, 0, 0
    
    for data in tqdm(loader, desc="Validation", leave=False):
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = prepare_batch(data, device)
        node_logits, edge_logits, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
        loss, recon, kl = vae_loss(node_logits, edge_logits, target_nodes, target_edges, mu, logvar, kl_weight)
        
        total_loss += loss.item() * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl += kl.item() * data.num_graphs
        
    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kl/n


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = ZINC(root='data/ZINC', subset=False, split='train')
    val_dataset = ZINC(root='data/ZINC', subset=False, split='val')

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    model = MolecularGraphVAE(
        num_node_features=NUM_NODE_FEATURES, 
        num_edge_features=NUM_EDGE_FEATURES, 
        latent_dim=64, 
        max_atoms=MAX_ATOMS
    ).to(device)

    kl_weight = 0.1
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best_val_loss = float('inf')

    for epoch in range(1, EPOCHS + 1):
        train_l, train_r, train_kl = train_epoch(model, optimizer, train_loader, kl_weight, device)
        val_l, val_r, val_kl = val_epoch(model, val_loader, kl_weight, device)
        
        print(f"Epoch {epoch:03d}")
        print(f"Train Loss: {train_l:.4f} | Recon: {train_r:.4f} | KL: {train_kl:.4f}")
        print(f"Val Loss: {val_l:.4f} | Recon: {val_r:.4f} | KL: {val_kl:.4f}")
        
        if val_l < best_val_loss:
            best_val_loss = val_l
            torch.save(model.state_dict(), 'checkpoints/vae_best.pth')
            print("Model checkpoint saved.")
        print("-" * 50)

if __name__ == "__main__":
    set_seed(SEED)
    os.makedirs('checkpoints', exist_ok=True)
