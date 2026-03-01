import torch
from molecule_benchmarks import Benchmarker, SmilesDataset

import os
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

from models.gvae import GraphVAE, gvae_loss, gvae_prepare_batch
from models.fast_jtnn import JTNNVAE
from utils.utils import Config, get_dataloaders, get_smiles_list

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train Molecular Graph VAE")
    parser.add_argument('--model', type=str, default='GVAE', choices=['GVAE', 'JTVAE'])
    parser.add_argument('--dataset', type=str, default='ZINC', choices=['ZINC', 'MOSES'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()
    return Config(**vars(args))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch_gvae(model, optimizer, loader, config, global_step, device):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    for data in tqdm(loader, desc="Training GVAE", leave=False):
        optimizer.zero_grad()
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = gvae_prepare_batch(data, device, config.max_atoms)
        node_logits, edge_logits, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
        kl_weight = min(config.kl_weight, global_step / config.kl_anneal_steps)
        loss, recon, kl = gvae_loss(node_logits, edge_logits, target_nodes, target_edges, mu, logvar, kl_weight)
        
        loss.backward()
        optimizer.step()
        global_step += 1
        
        total_loss += loss.item() * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl += kl.item() * data.num_graphs
        
    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kl/n, global_step

@torch.no_grad()
def val_epoch_gvae(model, loader, config, device):
    model.eval()
    total_loss, total_recon, total_kl = 0, 0, 0
    for data in tqdm(loader, desc="Validation GVAE", leave=False):
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = gvae_prepare_batch(data, device, config.max_atoms)
        node_logits, edge_logits, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
        loss, recon, kl = gvae_loss(node_logits, edge_logits, target_nodes, target_edges, mu, logvar, config.kl_weight)
        
        total_loss += loss.item() * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl += kl.item() * data.num_graphs
        
    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kl/n

def train_epoch_jtvae(model, optimizer, loader, config, global_step):
    model.train()
    total_loss, total_kl = 0, 0
    for batch in tqdm(loader, desc="Training JTVAE", leave=False):
        optimizer.zero_grad()
        beta = min(config.kl_weight, global_step / config.kl_anneal_steps)
        try:
            loss, kl_div, wacc, tacc, sacc = model(batch, beta)
        except ValueError:
            out = model(batch, beta)
            loss, kl_div = out[0], out[1] if len(out) > 1 else torch.tensor(0.0)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 50)
        optimizer.step()
        global_step += 1
        
        batch_size = len(batch)
        total_loss += loss.item() * batch_size
        total_kl += kl_div.item() * batch_size if isinstance(kl_div, torch.Tensor) else kl_div * batch_size
        
    n = len(loader.dataset)
    return total_loss/n, 0.0, total_kl/n, global_step

@torch.no_grad()
def val_epoch_jtvae(model, loader, config, device):
    model.eval()
    total_loss, total_kl = 0, 0
    for batch in tqdm(loader, desc="Validation JTVAE", leave=False):
        try:
            loss, kl_div, _, _, _ = model(batch, config.kl_weight)
        except ValueError:
            out = model(batch, config.kl_weight)
            loss, kl_div = out[0], out[1] if len(out) > 1 else torch.tensor(0.0)
            
        batch_size = len(batch)
        total_loss += loss.item() * batch_size
        total_kl += kl_div.item() * batch_size if isinstance(kl_div, torch.Tensor) else kl_div * batch_size
        
    n = len(loader.dataset)
    return total_loss/n, 0.0, total_kl/n


def evaluate_model(model, config, device, metadata):
    logger.info(f"Generating {config.num_samples} samples for benchmarking...")
    model.eval()
    generated_smiles = []

    with torch.no_grad():
        if config.model == 'GVAE':
            # Inverse mapping from model's internal node indices back to original atom types for both datasets
            if config.dataset == 'MOSES':
                atom_decoder = {0: 6, 1: 7, 2: 8, 3: 9, 4: 16, 5: 17, 6: 35, 7: 1}
            else:
                atom_decoder = {0: 6, 1: 7, 2: 8, 3: 9, 4: 15, 5: 16, 6: 17, 7: 35, 8: 53}

            z = torch.randn(config.num_samples, config.latent_dim).to(device)
            generated_smiles = model.sample_smiles(z, metadata['num_nodes'], metadata['num_edges'], atom_decoder)
            
        elif config.model == 'JTVAE':
            # Fast JTNN expects batched generation logic
            for _ in tqdm(range(max(1, config.num_samples // config.batch_size)), desc="Sampling JTVAE"):
                z_tree = torch.randn(config.batch_size, config.latent_dim // 2).to(device)
                z_mol = torch.randn(config.batch_size, config.latent_dim // 2).to(device)
                smiles_batch = model.decode(z_tree, z_mol, prob_decode=False)
                
                if smiles_batch is None:
                    continue
                
                # If it successfully decoded but only returned a single string, wrap it in a list
                if isinstance(smiles_batch, str):
                    smiles_batch = [smiles_batch]
                    
                generated_smiles.extend(smiles_batch)

    # Filter out invalid smiles (Nones or empty) before passing to benchmarker
    valid_smiles = [s for s in generated_smiles if s]
    validity = len(valid_smiles) / max(1, len(generated_smiles))
    logger.info(f"Validity (Pre-benchmarking): {validity:.4f}")

    if not valid_smiles:
        logger.error("No valid molecules generated. Skipping Benchmarks.")
        return

    logger.info("Running molecule-benchmarks...")
        
    if config.dataset == 'MOSES':
        reference_data = SmilesDataset.load_moses_dataset()
    else:
        # ZINC does not have a built-in loader in molecule_benchmarks, so we manually build it
        t_smi = random.sample(get_smiles_list('ZINC', 'train'), 10000)
        v_smi = random.sample(get_smiles_list('ZINC', 'val'), 10000)
        reference_data = SmilesDataset(train_smiles=t_smi, validation_smiles=v_smi)

    benchmarker = Benchmarker(
        dataset=reference_data, 
        num_samples_to_generate=len(valid_smiles), 
        device=device.type if device.type != 'mps' else 'cpu' 
    )
    
    # Execute run
    metrics = benchmarker.benchmark(valid_smiles)
    logger.info(f"Benchmark Results: {metrics}")


def train(config: Config):
    if config.dataset == 'MOSES':
        # MOSES is designed for molecules with up to 30 heavy atoms, we set this to 30 to save memory and speed up training
        config.max_atoms = 30

    current_kl_weight = config.kl_weight
    if config.model == 'GVAE':
        # GVAE needs much less KL pressure to avoid vanishing gradients
        current_kl_weight = config.kl_weight * 0.1
    config.kl_weight = current_kl_weight

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f'checkpoints/{config.dataset}/{config.model}', exist_ok=True)
    checkpoint_path = f'checkpoints/{config.dataset}/{config.model}/best.pth'

    logger.info(f"Loading {config.dataset} dataset for {config.model}...")
    train_loader, val_loader, metadata = get_dataloaders(config, logger)
    
    if config.model == 'GVAE':
        model = GraphVAE(
            num_node_features=metadata['num_nodes'], 
            num_edge_features=metadata['num_edges'], 
            latent_dim=config.latent_dim, 
            max_atoms=config.max_atoms
        ).to(device)
    else:
        model = JTNNVAE(
            metadata['vocab'], 
            hidden_size=256, 
            latent_size=config.latent_dim // 2, 
            depthT=20, 
            depthG=3
        ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    global_step, best_val_loss, counter = 0, float('inf'), 0

    logger.info(f"Starting training on {device}...")
    for epoch in range(1, config.epochs + 1):
        if config.model == 'GVAE':
            train_l, train_r, train_kl, global_step = train_epoch_gvae(model, optimizer, train_loader, config, global_step, device)
            val_l, val_r, val_kl = val_epoch_gvae(model, val_loader, config, device)
        else:
            train_l, train_r, train_kl, global_step = train_epoch_jtvae(model, optimizer, train_loader, config, global_step)
            val_l, val_r, val_kl = val_epoch_jtvae(model, val_loader, config, device)
            
        scheduler.step()       

        logger.info(f"Epoch {epoch:03d} | Train L: {train_l:.4f} | Val L: {val_l:.4f} | KL: {val_kl:.4f}")
        
        if val_l < best_val_loss:
            best_val_loss = val_l
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("New best model saved.")
        else:
            counter += 1
            if counter >= config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    # Load best model for benchmarking
    model.load_state_dict(torch.load(checkpoint_path))
    evaluate_model(model, config, device, metadata)

if __name__ == "__main__":
    config = parse_args()
    set_seed(config.seed)
    train(config)
