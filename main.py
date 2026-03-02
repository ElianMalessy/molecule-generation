import torch
import torch.nn as nn
from molecule_benchmarks import Benchmarker, SmilesDataset

import os
import random
import logging
import argparse
import numpy as np
from tqdm import tqdm

from models.gvae import GraphVAE, gvae_loss, gvae_prepare_batch
from models.frattvae import FRATTVAE, batched_kl_divergence
from models.frattvae.utils.mask import create_mask
from utils.utils import Config, get_dataloaders, get_smiles_list
from utils.constants import MOSES_ATOM_DECODER, ZINC_ATOM_DECODER, ZINC_CHARGE_DECODER

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train Molecular VAE")
    parser.add_argument('--model', type=str, default='GVAE', choices=['GVAE', 'FRATTVAE'])
    parser.add_argument('--dataset', type=str, default='ZINC', choices=['ZINC', 'MOSES'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=10000)
    # FRATTVAE hyperparameters (ignored for GVAE)
    parser.add_argument('--fratt_depth', type=int, default=8, help='Max fragment tree depth (paper default: 32)')
    parser.add_argument('--fratt_width', type=int, default=4, help='Max fragment tree degree (paper default: 32)')
    parser.add_argument('--fratt_d_model', type=int, default=256, help='Transformer hidden dim')
    parser.add_argument('--fratt_d_ff', type=int, default=1024, help='Transformer FFN dim')
    parser.add_argument('--fratt_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--fratt_nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--fratt_n_bits', type=int, default=2048, help='Morgan fingerprint bits')
    parser.add_argument('--fratt_max_nfrags', type=int, default=30, help='Max fragments during decoding')
    parser.add_argument('--label_loss_weight', type=float, default=2.0, help='Label CE loss weight')
    parser.add_argument('--n_jobs', type=int, default=8, help='CPU workers for BRICS preprocessing')
    parser.add_argument('--num_workers', type=int, default=4, help='DataLoader worker processes')
    parser.add_argument('--max_train_mols', type=int, default=0, help='Cap training set size (0=all, useful for quick tests)')
    args = parser.parse_args()
    return Config(**vars(args))

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train_epoch_gvae(model, optimizer, loader, config, global_step, device, amp_dtype=None):
    model.train()
    total_loss, total_recon, total_kl = 0, 0, 0
    for data in tqdm(loader, desc="Training GVAE", leave=False):
        optimizer.zero_grad()
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = gvae_prepare_batch(data, device, config.max_atoms)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
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
def val_epoch_gvae(model, loader, config, global_step, device, amp_dtype=None):
    model.eval()
    total_loss, total_recon, total_kl = 0, 0, 0
    for data in tqdm(loader, desc="Validation GVAE", leave=False):
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = gvae_prepare_batch(data, device, config.max_atoms)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            node_logits, edge_logits, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
            beta = min(config.kl_weight, global_step / config.kl_anneal_steps)
            loss, recon, kl = gvae_loss(node_logits, edge_logits, target_nodes, target_edges, mu, logvar, beta)
        
        total_loss += loss.item() * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl += kl.item() * data.num_graphs
        
    n = len(loader.dataset)
    return total_loss/n, total_recon/n, total_kl/n

def _frattvae_criterion(freq_label: torch.Tensor, device) -> nn.CrossEntropyLoss:
    """Frequency-weighted CrossEntropyLoss for fragment token prediction."""
    freq = freq_label.clone().clamp(max=1000)
    weight = freq.max() / freq
    weight[~torch.isfinite(weight)] = 0.001
    return nn.CrossEntropyLoss(weight=weight.to(device))


def train_epoch_frattvae(model, optimizer, loader, config, global_step, device, frag_ecfps, freq_label, amp_dtype=None):
    model.train()
    criterion = _frattvae_criterion(freq_label, device)
    num_tokens = frag_ecfps.shape[0]
    total_loss = total_kl = total_label = 0.0

    for frag_indices, positions, _ in tqdm(loader, desc="Train FRATTVAE", leave=False):
        B, L = frag_indices.shape
        features      = frag_ecfps[frag_indices.flatten()].reshape(B, L, -1).to(device)
        positions     = positions.to(device)
        target        = torch.cat([frag_indices, torch.zeros(B, 1)], dim=1).flatten().long().to(device)
        idx_with_root = torch.cat([torch.full((B, 1), -1), frag_indices], dim=1).to(device)

        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(idx_with_root, idx_with_root, pad_idx=0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            z, mu, ln_var, output = model(features, positions, src_mask, src_pad_mask, tgt_mask, tgt_pad_mask)
            kl_weight  = min(config.kl_weight, global_step / config.kl_anneal_steps)
            kl_loss    = batched_kl_divergence(mu, ln_var)
            label_loss = criterion(output.view(-1, num_tokens), target)
            loss       = kl_weight * kl_loss + config.label_loss_weight * label_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        global_step += 1

        total_loss  += loss.item()        * B
        total_kl    += kl_loss.item()     * B
        total_label += label_loss.item()  * B

    n = len(loader.dataset)
    return total_loss / n, total_label / n, total_kl / n, global_step


@torch.no_grad()
def val_epoch_frattvae(model, loader, config, global_step, device, frag_ecfps, freq_label, amp_dtype=None):
    model.eval()
    criterion = _frattvae_criterion(freq_label, device)
    num_tokens = frag_ecfps.shape[0]
    total_loss = total_kl = total_label = 0.0

    for frag_indices, positions, _ in tqdm(loader, desc="Val FRATTVAE", leave=False):
        B, L = frag_indices.shape
        features      = frag_ecfps[frag_indices.flatten()].reshape(B, L, -1).to(device)
        positions     = positions.to(device)
        target        = torch.cat([frag_indices, torch.zeros(B, 1)], dim=1).flatten().long().to(device)
        idx_with_root = torch.cat([torch.full((B, 1), -1), frag_indices], dim=1).to(device)

        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(idx_with_root, idx_with_root, pad_idx=0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            # Use parallel (teacher-forcing) decode during validation for speed
            z, mu, ln_var, output = model(features, positions, src_mask, src_pad_mask, tgt_mask, tgt_pad_mask,
                                          sequential=False)
            kl_weight  = min(config.kl_weight, global_step / config.kl_anneal_steps)
            kl_loss    = batched_kl_divergence(mu, ln_var)
            label_loss = criterion(output.view(-1, num_tokens), target)
            loss       = kl_weight * kl_loss + config.label_loss_weight * label_loss

        total_loss  += loss.item()        * B
        total_kl    += kl_loss.item()     * B
        total_label += label_loss.item()  * B

    n = len(loader.dataset)
    return total_loss / n, total_label / n, total_kl / n


def evaluate_model(model, config, device, metadata, val_loader=None):
    model.eval()

    # Decoders needed for GVAE sampling and reconstruction
    if config.model == 'GVAE':
        atom_decoder = MOSES_ATOM_DECODER if config.dataset == 'MOSES' else ZINC_ATOM_DECODER
        charge_decoder = None if config.dataset == 'MOSES' else ZINC_CHARGE_DECODER

    # ------------------------------------------------------------------
    # 1. Reconstruction accuracy on a held-out sample
    #    Encode real molecules -> decode -> check recovery.
    #    This is the most direct measure of VAE fidelity.
    # ------------------------------------------------------------------
    if config.model == 'GVAE' and val_loader is not None:
        recon_correct = recon_total = 0
        logger.info("Evaluating reconstruction accuracy on validation set (up to 1000 molecules)...")
        with torch.no_grad():
            for data in val_loader:
                x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = gvae_prepare_batch(data, device, config.max_atoms)
                _, _, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
                z = mu  # Use mean (no noise) for reconstruction eval
                recon_smiles = model.sample_smiles(z, atom_decoder, charge_decoder)

                for i, smi in enumerate(recon_smiles):
                    recon_total += 1
                    if smi:
                        recon_correct += 1
                if recon_total >= 1000:
                    break

        logger.info(f"Reconstruction: {recon_correct}/{recon_total} valid decodes "
                    f"({100 * recon_correct / max(1, recon_total):.1f}%) from posterior mean z")

    # ------------------------------------------------------------------
    # 2. Prior sampling + benchmark suite
    # ------------------------------------------------------------------
    logger.info(f"Generating {config.num_samples} samples for benchmarking...")
    generated_smiles = []

    with torch.no_grad():
        if config.model == 'GVAE':
            z = torch.randn(config.num_samples, config.latent_dim).to(device)
            generated_smiles = model.sample_smiles(z, atom_decoder, charge_decoder)

        elif config.model == 'FRATTVAE':
            frag_ecfps = metadata['frag_ecfps'].to(device)
            ndummys = metadata['ndummys'].to(device)
            uni_fragments = metadata['uni_fragments']
            model.set_labels(uni_fragments)

            decode_batch = 256
            z_all = torch.randn(config.num_samples, config.latent_dim)
            logger.info("Sampling FRATTVAE (sequential decode)...")
            for i in tqdm(range(0, config.num_samples, decode_batch), desc="Sampling FRATTVAE"):
                z_batch = z_all[i:i + decode_batch].to(device)
                smiles_batch = model.sequential_decode(
                    z_batch, frag_ecfps, ndummys,
                    max_nfrags=config.fratt_max_nfrags,
                    asSmiles=True,
                )
                generated_smiles.extend(smiles_batch)

    valid_smiles = [s for s in generated_smiles if s]
    validity = len(valid_smiles) / max(1, len(generated_smiles))
    logger.info(f"Prior sampling validity: {validity:.4f} "
                f"({len(valid_smiles)}/{len(generated_smiles)})")

    if not valid_smiles:
        logger.error("No valid molecules generated. Skipping Benchmarks.")
        return

    if config.max_train_mols > 0:
        logger.info("Skipping benchmarks (max_train_mols cap active — use full dataset for real evaluation).")
        return

    logger.info("Running molecule-benchmarks...")
    try:
        if config.dataset == 'MOSES':
            reference_data = SmilesDataset.load_moses_dataset()
        else:
            t_smi = random.sample(get_smiles_list('ZINC', 'train'), 10000)
            v_smi = random.sample(get_smiles_list('ZINC', 'val'), 10000)
            reference_data = SmilesDataset(train_smiles=t_smi, validation_smiles=v_smi)

        benchmarker = Benchmarker(
            dataset=reference_data,
            num_samples_to_generate=config.num_samples,
            device=device.type
        )
        # Pass the full generated list (including None/invalid) so the benchmarker
        # can compute validity metrics correctly. Pre-filtering would make
        # valid_fraction always appear as ~1.0.
        metrics = benchmarker.benchmark(generated_smiles)
        logger.info(f"Benchmark Results: {metrics}")
    except KeyboardInterrupt:
        logger.warning("Benchmarking interrupted by user.")
    except Exception as e:
        logger.warning(f"Benchmarking failed: {e}")


def train(config: Config):
    if config.dataset == 'MOSES':
        config.max_atoms = 30

    if config.model == 'GVAE':
        config.kl_weight = config.kl_weight * 0.1
    elif config.model == 'FRATTVAE':
        config.kl_weight = min(config.kl_weight, 0.0005)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f'checkpoints/{config.dataset}/{config.model}', exist_ok=True)
    checkpoint_path = f'checkpoints/{config.dataset}/{config.model}/best.pth'

    logger.info(f"Loading {config.dataset} dataset for {config.model}...")
    train_loader, val_loader, metadata = get_dataloaders(config, logger)

    steps_per_epoch = len(train_loader)
    kl_anneal_epoch = -(-config.kl_anneal_steps // steps_per_epoch)
    logger.info(f"KL annealing: {config.kl_anneal_steps} steps "
                f"= {kl_anneal_epoch} epochs @ {steps_per_epoch} steps/epoch")

    if config.model == 'GVAE':
        model = GraphVAE(
            num_node_features=metadata['num_nodes'],
            num_edge_features=metadata['num_edges'],
            latent_dim=config.latent_dim,
            max_atoms=config.max_atoms
        ).to(device)
    else:
        model = FRATTVAE(
            num_tokens=metadata['num_frags'],
            depth=config.fratt_depth,
            width=config.fratt_width,
            feat_dim=config.fratt_n_bits,
            latent_dim=config.latent_dim,
            d_model=config.fratt_d_model,
            d_ff=config.fratt_d_ff,
            num_layers=config.fratt_layers,
            nhead=config.fratt_nhead,
            n_jobs=config.n_jobs,
        ).to(device)

    amp_dtype = torch.bfloat16 if device.type == 'cuda' else None
    logger.info(f"AMP: {'bfloat16' if amp_dtype else 'disabled (CPU)'}")

    if device.type == 'cuda':
        logger.info("Compiling model with torch.compile (first epoch will be slower)...")
        model = torch.compile(model, mode='reduce-overhead', dynamic=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    frag_ecfps = metadata.get('frag_ecfps')
    freq_label = metadata.get('freq_label')

    global_step, best_val_loss, counter = 0, float('inf'), 0

    logger.info(f"Starting training on {device}...")
    for epoch in range(1, config.epochs + 1):
        if config.model == 'GVAE':
            train_l, train_recon, train_kl, global_step = train_epoch_gvae(
                model, optimizer, train_loader, config, global_step, device, amp_dtype=amp_dtype)
            val_l, val_recon, val_kl = val_epoch_gvae(
                model, val_loader, config, global_step, device, amp_dtype=amp_dtype)
            logger.info(f"Epoch {epoch:03d} | Train: {train_l:.4f} | Val: {val_l:.4f} "
                        f"| Recon: {val_recon:.4f} | KL: {val_kl:.4f}")
        else:
            train_l, train_label, train_kl, global_step = train_epoch_frattvae(
                model, optimizer, train_loader, config, global_step, device, frag_ecfps, freq_label, amp_dtype=amp_dtype)
            val_l, val_label, val_kl = val_epoch_frattvae(
                model, val_loader, config, global_step, device, frag_ecfps, freq_label, amp_dtype=amp_dtype)
            logger.info(f"Epoch {epoch:03d} | Train: {train_l:.4f} | Val: {val_l:.4f} "
                        f"| Label: {val_label:.4f} | KL: {val_kl:.4f}")

        scheduler.step()

        if val_l < best_val_loss:
            best_val_loss = val_l
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("New best model saved.")
        else:
            counter += 1
            if epoch > kl_anneal_epoch and counter >= config.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    evaluate_model(model, config, device, metadata, val_loader=val_loader)

if __name__ == "__main__":
    config = parse_args()
    set_seed(config.seed)
    train(config)
