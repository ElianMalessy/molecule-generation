import torch

import os
import logging
import argparse

from models.gvae import GraphVAE
from models.gvae_nf import GraphVAENF
from models.frattvae import FRATTVAE
from utils.utils import Config, get_dataloaders, set_seed, cyclical_beta
from training import (train_epoch_gvae, val_epoch_gvae,
                      train_epoch_gvae_nf, val_epoch_gvae_nf,
                      train_epoch_frattvae, val_epoch_frattvae)
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train Molecular VAE")
    parser.add_argument('--model',        type=str,   default='GVAE', choices=['GVAE', 'GVAE_NF', 'FRATTVAE'])
    parser.add_argument('--dataset',      type=str,   default='ZINC', choices=['ZINC', 'MOSES'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--no_valency_mask', action='store_true',
                        help='Disable valency masking during GVAE decoding')
    # Joint property prediction
    parser.add_argument('--prop_pred',          action='store_true',
                        help='Attach a property prediction head (plogP, QED, SA) to GVAE/GVAE_NF')
    parser.add_argument('--prop_weight',        type=float, default=1.0,
                        help='γ: property loss weight at full scale (default 1.0)')
    parser.add_argument('--prop_warmup_epochs', type=int,   default=15,
                        help='Epochs before property loss starts ramping up (default 15)')
    args = parser.parse_args()
    config = Config(model=args.model, dataset=args.dataset)
    config.gvae.weight_decay = args.weight_decay
    if args.no_valency_mask:
        config.gvae.valency_mask = False
        config.gvae_nf.valency_mask = False
    if args.prop_pred:
        config.gvae.prop_pred          = True
        config.gvae.prop_weight        = args.prop_weight
        config.gvae.prop_warmup_epochs = args.prop_warmup_epochs
        config.gvae_nf.prop_pred          = True
        config.gvae_nf.prop_weight        = args.prop_weight
        config.gvae_nf.prop_warmup_epochs = args.prop_warmup_epochs
    return config


def _variant_name(config: Config) -> str:
    """Unique subdirectory name for a run, e.g. 'prop+no_valency' or 'no_prop'.
    Keeps checkpoints from different flag combinations from clobbering each other."""
    gc = config.gvae if config.model == 'GVAE' else config.gvae_nf if config.model == 'GVAE_NF' else None
    if gc is None:
        return 'default'
    parts = ['prop' if gc.prop_pred else 'no_prop']
    if not gc.valency_mask:
        parts.append('no_valency')
    return '+'.join(parts)


def train(config: Config):
    if config.dataset == 'MOSES':
        config.gvae.max_atoms = 30
        config.gvae_nf.max_atoms = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = f'checkpoints/{config.dataset}/{config.model}/{_variant_name(config)}'
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = f'{ckpt_dir}/best.pth'

    logger.info(f"Loading {config.dataset} dataset for {config.model}...")
    train_loader, val_loader, metadata = get_dataloaders(config, logger)

    steps_per_epoch = len(train_loader)
    if config.model in ('GVAE', 'GVAE_NF'):
        mc = config.gvae if config.model == 'GVAE' else config.gvae_nf
        one_cycle_steps = mc.kl_anneal_steps / mc.kl_cycles
        kl_anneal_epoch = -(-int(one_cycle_steps) // steps_per_epoch)
        # Patience = 2 full cycles so the model must fail to improve across two
        # complete β ramp-up/hold sequences before stopping.  One cycle is the
        # minimum but can be unlucky; two gives the model a chance to recover
        # across a full second annealing cycle.
        mc.patience = 2 * kl_anneal_epoch
        logger.info(f"Cyclical β: {mc.kl_cycles} cycles over {mc.kl_anneal_steps} steps "
                    f"(1 cycle = {kl_anneal_epoch} epochs, ramp ratio = {mc.kl_anneal_ratio}, "
                    f"patience = {mc.patience} epochs)")
    else:
        mc = config.frattvae
        kl_anneal_epoch = 0  # not used for FRATTVAE

    if config.model == 'GVAE':
        model = GraphVAE(
            num_node_features=metadata['num_nodes'],
            num_edge_features=metadata['num_edges'],
            latent_dim=config.gvae.latent_dim,
            max_atoms=config.gvae.max_atoms,
            prop_pred=config.gvae.prop_pred,
        ).to(device)
    elif config.model == 'GVAE_NF':
        model = GraphVAENF(
            num_node_features=metadata['num_nodes'],
            num_edge_features=metadata['num_edges'],
            latent_dim=config.gvae_nf.latent_dim,
            max_atoms=config.gvae_nf.max_atoms,
            num_flows=config.gvae_nf.num_flows,
            flow_hidden_dim=config.gvae_nf.flow_hidden_dim,
            prop_pred=config.gvae_nf.prop_pred,
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
    elif config.model == 'GVAE_NF':
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.gvae_nf.lr, weight_decay=config.gvae_nf.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.gvae_nf.epochs)
    else:  # GVAE
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=config.gvae.lr, weight_decay=config.gvae.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.gvae.epochs)

    frag_ecfps = metadata.get('frag_ecfps')
    freq_label = metadata.get('freq_label')
    prop_mean  = metadata.get('prop_mean')   # None when prop_pred=False
    prop_std   = metadata.get('prop_std')

    if config.model in ('GVAE', 'GVAE_NF'):
        logger.info(f"Valency masking: {'ON' if mc.valency_mask else 'OFF'}")
    if config.model in ('GVAE', 'GVAE_NF') and mc.prop_pred:
        logger.info(f"Joint property prediction: γ={mc.prop_weight}, "
                    f"warmup={mc.prop_warmup_epochs} epochs (plogP / QED / SA)")

    global_step, best_val_loss, counter = 0, float('inf'), 0

    logger.info(f"Starting training on {device}...")
    for epoch in range(1, mc.epochs + 1):
        ep_kw = dict(epoch=epoch, prop_mean=prop_mean, prop_std=prop_std)
        if config.model == 'GVAE':
            train_l, train_recon, train_kl, train_prop, train_raw_prop, global_step = train_epoch_gvae(
                model, optimizer, train_loader, config, global_step, device,
                amp_dtype=amp_dtype, **ep_kw)
            val_l, val_recon, val_kl, val_prop, val_raw_prop = val_epoch_gvae(
                model, val_loader, config, global_step, device,
                amp_dtype=amp_dtype, **ep_kw)
            train_prop_str = (f" | prop: mse={train_raw_prop:.4f}  r²={max(0.0, 1 - train_raw_prop):.4f}"
                              if mc.prop_pred else "")
            val_prop_str   = (f" | prop: mse={val_raw_prop:.4f}  r²={max(0.0, 1 - val_raw_prop):.4f}"
                              if mc.prop_pred else "")
            logger.info(
                f"Epoch {epoch:03d} "
                f"| train: {train_l:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f})"
                f"{train_prop_str}")
            logger.info(
                f"Epoch {epoch:03d} "
                f"| val:   {val_l:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f})"
                f"{val_prop_str}")
        elif config.model == 'GVAE_NF':
            train_l, train_recon, train_kl, train_prop, train_raw_prop, global_step = train_epoch_gvae_nf(
                model, optimizer, train_loader, config, global_step, device,
                amp_dtype=amp_dtype, **ep_kw)
            val_l, val_recon, val_kl, val_prop, val_raw_prop = val_epoch_gvae_nf(
                model, val_loader, config, global_step, device,
                amp_dtype=amp_dtype, **ep_kw)
            train_prop_str = (f" | prop: mse={train_raw_prop:.4f}  r²={max(0.0, 1 - train_raw_prop):.4f}"
                              if mc.prop_pred else "")
            val_prop_str   = (f" | prop: mse={val_raw_prop:.4f}  r²={max(0.0, 1 - val_raw_prop):.4f}"
                              if mc.prop_pred else "")
            logger.info(
                f"Epoch {epoch:03d} "
                f"| train: {train_l:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f})"
                f"{train_prop_str}")
            logger.info(
                f"Epoch {epoch:03d} "
                f"| val:   {val_l:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f})"
                f"{val_prop_str}")
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

        # Only save checkpoints when β is at (or near) its maximum within the
        # current cycle.  Saving at β≈0 (cycle reset) produces checkpoints with
        # near-zero KL pressure — the posterior is poorly aligned with N(0,I),
        # which tankes FCD even if total val_loss looks low.
        if config.model in ('GVAE', 'GVAE_NF'):
            current_beta = cyclical_beta(global_step, mc.kl_anneal_steps,
                                         mc.kl_weight, mc.kl_cycles, mc.kl_anneal_ratio)
            beta_mature = current_beta >= mc.kl_weight * 0.9
        else:
            beta_mature = True  # FRATTVAE has a fixed small kl_weight, no cycling

        if beta_mature and val_l < best_val_loss:
            best_val_loss = val_l
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("New best model saved.")
        else:
            # Only count patience during beta-mature epochs: during the ramp-up
            # phase the model cannot save regardless, so those epochs shouldn't
            # consume patience budget.
            if beta_mature or config.model == 'FRATTVAE':
                counter += 1
            early_stop_ok = (epoch > kl_anneal_epoch) if config.model in ('GVAE', 'GVAE_NF') else True
            if early_stop_ok and counter >= mc.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    evaluate_model(model, config, device, metadata)

if __name__ == "__main__":
    config = parse_args()
    set_seed(config.seed)
    train(config)
