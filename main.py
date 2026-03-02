import torch

import os
import logging
import argparse

from models.gvae import GraphVAE
from models.frattvae import FRATTVAE
from utils.utils import Config, get_dataloaders, set_seed
from training import train_epoch_gvae, val_epoch_gvae, train_epoch_frattvae, val_epoch_frattvae
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train Molecular VAE")
    parser.add_argument('--model',        type=str,   default='GVAE', choices=['GVAE', 'FRATTVAE'])
    parser.add_argument('--dataset',      type=str,   default='ZINC', choices=['ZINC', 'MOSES'])
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='Override GVAE AdamW weight decay (default: 1e-4)')
    args = parser.parse_args()
    config = Config(model=args.model, dataset=args.dataset)
    if args.weight_decay is not None:
        config.gvae.weight_decay = args.weight_decay
    return config


def train(config: Config):
    if config.dataset == 'MOSES':
        config.gvae.max_atoms = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(f'checkpoints/{config.dataset}/{config.model}', exist_ok=True)
    checkpoint_path = f'checkpoints/{config.dataset}/{config.model}/best.pth'

    logger.info(f"Loading {config.dataset} dataset for {config.model}...")
    train_loader, val_loader, metadata = get_dataloaders(config, logger)

    mc = config.gvae if config.model == 'GVAE' else config.frattvae
    steps_per_epoch = len(train_loader)
    kl_anneal_epoch = -(-mc.kl_anneal_steps // steps_per_epoch)
    logger.info(f"KL annealing: {mc.kl_anneal_steps} steps "
                f"= {kl_anneal_epoch} epochs @ {steps_per_epoch} steps/epoch")

    if config.model == 'GVAE':
        model = GraphVAE(
            num_node_features=metadata['num_nodes'],
            num_edge_features=metadata['num_edges'],
            latent_dim=config.gvae.latent_dim,
            max_atoms=config.gvae.max_atoms,
        ).to(device)
    else:
        model = FRATTVAE(
            num_tokens=metadata['num_frags'],
            depth=config.frattvae.depth,
            width=config.frattvae.width,
            feat_dim=config.frattvae.n_bits,
            latent_dim=config.frattvae.latent_dim,
            d_model=config.frattvae.d_model,
            d_ff=config.frattvae.d_ff,
            num_layers=config.frattvae.num_layers,
            nhead=config.frattvae.nhead,
            n_jobs=config.frattvae.n_jobs,
        ).to(device)

    amp_dtype = torch.bfloat16 if device.type == 'cuda' else None
    logger.info(f"AMP: {'bfloat16' if amp_dtype else 'disabled (CPU)'}")

    if config.model == 'FRATTVAE':
        # Match paper: Adam with eps=1e-3 (larger eps stabilises adaptive denominator)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.frattvae.lr, eps=1e-3)
        # Paper uses constant lr throughout — no scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.gvae.lr, weight_decay=config.gvae.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.gvae.epochs)

    frag_ecfps = metadata.get('frag_ecfps')
    freq_label = metadata.get('freq_label')

    global_step, best_val_loss, counter = 0, float('inf'), 0

    logger.info(f"Starting training on {device}...")
    for epoch in range(1, mc.epochs + 1):
        if config.model == 'GVAE':
            train_l, train_recon, train_kl, global_step = train_epoch_gvae(
                model, optimizer, train_loader, config, global_step, device, amp_dtype=amp_dtype)
            val_l, val_recon, val_kl = val_epoch_gvae(
                model, val_loader, config, global_step, device, amp_dtype=amp_dtype)
            logger.info(f"Epoch {epoch:03d} | Train: {train_l:.4f} | Val: {val_l:.4f} "
                        f"| Recon: {val_recon:.4f} | KL: {val_kl:.4f}")
        else:
            train_l, train_label, train_kl, global_step = train_epoch_frattvae(
                model, optimizer, train_loader, config, global_step, device,
                frag_ecfps, freq_label, amp_dtype=amp_dtype)
            val_l, val_label, val_kl = val_epoch_frattvae(
                model, val_loader, config, global_step, device,
                frag_ecfps, freq_label, amp_dtype=amp_dtype)
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
            # GVAE: wait until KL warmup completes before allowing early stopping.
            # FRATTVAE: warmup completes in ~20 steps (<1 epoch), so no gate needed.
            early_stop_ok = (epoch > kl_anneal_epoch) if config.model == 'GVAE' else True
            if early_stop_ok and counter >= mc.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    evaluate_model(model, config, device, metadata, val_loader=val_loader)

if __name__ == "__main__":
    config = parse_args()
    set_seed(config.seed)
    train(config)
