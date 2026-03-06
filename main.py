import torch

import os
import logging
import argparse

from models.gvae import GraphVAE, GraphVAENF
from models.gvae_ar import GraphVAEAR, GraphVAEARNF
from models.frattvae import FRATTVAE
from utils.utils import Config, get_dataloaders, set_seed
from training import (train_epoch_gvae, val_epoch_gvae,
                      train_epoch_gvae_ar, val_epoch_gvae_ar,
                      train_epoch_frattvae, val_epoch_frattvae)
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Train Molecular VAE")
    parser.add_argument('--model',        type=str,   default='GVAE', choices=['GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF', 'FRATTVAE'])
    parser.add_argument('--dataset',      type=str,   default='ZINC', choices=['ZINC', 'MOSES'])
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--valency_mask', action='store_true',
                        help='Enable valency masking during GVAE decoding')
    # Joint property prediction
    parser.add_argument('--prop_pred', action='store_true',
                        help='Attach a property prediction head (plogP, QED, SA) to GVAE/GVAE_NF')
    parser.add_argument('--prop_weight', type=float, default=None, help='property loss weight')
    args = parser.parse_args()
    config = Config(model=args.model, dataset=args.dataset)
    config.gvae.weight_decay = args.weight_decay
    if args.valency_mask:
        config.gvae.valency_mask = True
        config.gvae_nf.valency_mask = True
        config.gvae_ar.valency_mask = True
        config.gvae_ar_nf.valency_mask = True
    if args.prop_pred:
        config.gvae.prop_pred          = True
        config.gvae_nf.prop_pred          = True
        config.gvae_ar.prop_pred          = True
        config.gvae_ar_nf.prop_pred          = True
        # Override per-model weight defaults only when the flag is explicitly passed.
        # GVAE/NF: 5.0  |  GVAE_AR: 1.0  |  GVAE_AR_NF: 0.1
        if args.prop_weight is not None:
            config.gvae.prop_weight        = args.prop_weight
            config.gvae_nf.prop_weight     = args.prop_weight
            config.gvae_ar.prop_weight     = args.prop_weight
            config.gvae_ar_nf.prop_weight  = args.prop_weight
    return config


def _variant_name(config: Config) -> str:
    """Unique subdirectory name for a run, e.g. 'prop+no_valency' or 'no_prop'.
    Keeps checkpoints from different flag combinations from clobbering each other."""
    gc = config.gvae if config.model == 'GVAE' else config.gvae_nf if config.model == 'GVAE_NF' else config.gvae_ar if config.model == 'GVAE_AR' else config.gvae_ar_nf if config.model == 'GVAE_AR_NF' else None
    if gc is None:
        return 'default'
    parts = ['prop' if gc.prop_pred else 'no_prop']
    if gc.valency_mask:
        parts.append('valency')
    return '+'.join(parts)


def train(config: Config):
    if config.dataset == 'MOSES':
        config.gvae.max_atoms = 30
        config.gvae_nf.max_atoms = 30
        config.gvae_ar.max_atoms = 30
        config.gvae_ar_nf.max_atoms = 30

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_dir = f'checkpoints/{config.dataset}/{config.model}/{_variant_name(config)}'
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_path = f'{ckpt_dir}/best.pth'

    logger.info(f"Loading {config.dataset} dataset for {config.model}...")
    train_loader, val_loader, metadata = get_dataloaders(config, logger)

    steps_per_epoch = len(train_loader)
    if config.model in ('GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF'):
        mc = config.gvae if config.model == 'GVAE' else config.gvae_nf if config.model == 'GVAE_NF' else config.gvae_ar if config.model == 'GVAE_AR' else config.gvae_ar_nf
        kl_anneal_epoch = -(-mc.kl_anneal_steps // steps_per_epoch)  # ceil
        fb_info  = (f", free_bits_per_dim={mc.free_bits_per_dim}"
                    if getattr(mc, 'free_bits_per_dim', 0.0) > 0 else "")
        logger.info(f"KL: β={mc.kl_weight} (constant), capacity 0→{mc.kl_capacity_max} "
                    f"over {mc.kl_anneal_steps} steps ({kl_anneal_epoch} epochs)"
                    f"{fb_info} | patience = {mc.patience} epochs after ramp")
    else:
        mc = config.frattvae

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
    elif config.model == 'GVAE_AR':
        model = GraphVAEAR(
            num_node_features=metadata['num_nodes'],
            num_edge_features=metadata['num_edges'],
            latent_dim=config.gvae_ar.latent_dim,
            max_atoms=config.gvae_ar.max_atoms,
            ar_d_model=config.gvae_ar.ar_d_model,
            ar_n_heads=config.gvae_ar.ar_n_heads,
            ar_n_layers=config.gvae_ar.ar_n_layers,
            ar_d_ff=config.gvae_ar.ar_d_ff,
            ar_dropout=config.gvae_ar.ar_dropout,
            prop_pred=config.gvae_ar.prop_pred,
            context_dropout=config.gvae_ar.context_dropout,
        ).to(device)
    elif config.model == 'GVAE_AR_NF':
        model = GraphVAEARNF(
            num_node_features=metadata['num_nodes'],
            num_edge_features=metadata['num_edges'],
            latent_dim=config.gvae_ar_nf.latent_dim,
            max_atoms=config.gvae_ar_nf.max_atoms,
            num_flows=config.gvae_ar_nf.num_flows,
            flow_hidden_dim=config.gvae_ar_nf.flow_hidden_dim,
            ar_d_model=config.gvae_ar_nf.ar_d_model,
            ar_n_heads=config.gvae_ar_nf.ar_n_heads,
            ar_n_layers=config.gvae_ar_nf.ar_n_layers,
            ar_d_ff=config.gvae_ar_nf.ar_d_ff,
            ar_dropout=config.gvae_ar_nf.ar_dropout,
            prop_pred=config.gvae_ar_nf.prop_pred,
            context_dropout=config.gvae_ar_nf.context_dropout,
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
        # All GVAE variants: AdamW + cosine schedule.
        # The prop head gets its own higher LR (3× backbone) to counteract the
        # Adam second-moment asymmetry: the backbone's v_t is built up from strong
        # reconstruction + KL gradients, so the prop-to-encoder signal (rank-3)
        # is suppressed by sqrt(v_backbone) unless we raise its effective LR.
        # A separate param group gives the head a fresh v_t and its own LR.
        prop_params    = (list(model.prop_head.parameters())
                          if mc.prop_pred and getattr(model, 'prop_head', None) is not None
                          else [])
        prop_param_ids = {id(p) for p in prop_params}
        backbone_params = [p for p in model.parameters() if id(p) not in prop_param_ids]
        param_groups = [{'params': backbone_params, 'lr': mc.lr}]
        if prop_params:
            param_groups.append({'params': prop_params, 'lr': mc.lr * 3})
        optimizer = torch.optim.AdamW(param_groups,
                                      weight_decay=mc.weight_decay,
                                      fused=device.type == 'cuda')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=mc.epochs)

    frag_ecfps = metadata.get('frag_ecfps')
    freq_label = metadata.get('freq_label')
    prop_mean  = metadata.get('prop_mean')   # None when prop_pred=False
    prop_std   = metadata.get('prop_std')
    node_class_weights = metadata.get('node_class_weights')  # None for AR / FRATTVAE / MOSES
    edge_class_weights = metadata.get('edge_class_weights')  # None for AR / FRATTVAE / MOSES

    if config.model in ('GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF'):
        logger.info(f"Valency masking: {'ON' if mc.valency_mask else 'OFF'}")
    if config.model in ('GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF') and mc.prop_pred:
        logger.info(f"Joint property prediction: γ={mc.prop_weight} (constant, plogP / QED / SA)")

    global_step, best_val_loss, counter = 0, float('inf'), 0

    logger.info(f"Starting training on {device}...")
    for epoch in range(1, mc.epochs + 1):
        ep_kw = dict(epoch=epoch, prop_mean=prop_mean, prop_std=prop_std,
                     node_class_weights=node_class_weights,
                     edge_class_weights=edge_class_weights)
        if config.model in ('GVAE', 'GVAE_NF'):
            train_l, train_recon, train_kl, train_prop, train_raw_prop, train_prop_gnorm, global_step = train_epoch_gvae(
                model, optimizer, train_loader, config, global_step, device,
                amp_dtype=amp_dtype, **ep_kw)
            val_l, val_recon, val_kl, val_prop, val_raw_prop = val_epoch_gvae(
                model, val_loader, config, global_step, device,
                amp_dtype=amp_dtype, **ep_kw)
            # ckpt_loss includes prop (gamma is constant, already baked into val_l)
            ckpt_loss = val_l
            train_prop_str = (f" | prop: mse={train_raw_prop:.4f}  r\u00b2={max(0.0, 1 - train_raw_prop):.4f}  grad={train_prop_gnorm:.2e}"
                              if mc.prop_pred else "")
            val_prop_str   = (f" | prop: mse={val_raw_prop:.4f}  r\u00b2={max(0.0, 1 - val_raw_prop):.4f}"
                              if mc.prop_pred else "")
            logger.info(
                f"Epoch {epoch:03d} "
                f"| train: {train_l:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f})"
                f"{train_prop_str}")
            logger.info(
                f"Epoch {epoch:03d} "
                f"| val:   {val_l:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f})"
                f"{val_prop_str}  [ckpt={ckpt_loss:.4f}]")
        elif config.model in ('GVAE_AR', 'GVAE_AR_NF'):
            train_l, train_recon, train_kl, train_prop, train_raw_prop, train_prop_gnorm, global_step = train_epoch_gvae_ar(
                model, optimizer, train_loader, config, global_step, device,
                amp_dtype=amp_dtype, **ep_kw)
            val_l, val_recon, val_kl, val_prop, val_raw_prop = val_epoch_gvae_ar(
                model, val_loader, config, global_step, device,
                amp_dtype=amp_dtype, **ep_kw)
            # ckpt_loss includes prop (gamma is constant, already baked into val_l)
            ckpt_loss = val_l
            train_prop_str = (f" | prop: mse={train_raw_prop:.4f}  r\u00b2={max(0.0, 1 - train_raw_prop):.4f}  grad={train_prop_gnorm:.2e}"
                              if mc.prop_pred else "")
            val_prop_str   = (f" | prop: mse={val_raw_prop:.4f}  r\u00b2={max(0.0, 1 - val_raw_prop):.4f}"
                              if mc.prop_pred else "")
            logger.info(
                f"Epoch {epoch:03d} "
                f"| train: {train_l:.4f} (recon={train_recon:.4f}, kl={train_kl:.4f})"
                f"{train_prop_str}")
            logger.info(
                f"Epoch {epoch:03d} "
                f"| val:   {val_l:.4f} (recon={val_recon:.4f}, kl={val_kl:.4f})"
                f"{val_prop_str}  [ckpt={ckpt_loss:.4f}]")
        else:
            train_l, train_label, train_kl, global_step = train_epoch_frattvae(
                model, optimizer, train_loader, config, global_step, device,
                frag_ecfps, freq_label, amp_dtype=amp_dtype)
            val_l, val_label, val_kl = val_epoch_frattvae(
                model, val_loader, config, global_step, device,
                frag_ecfps, freq_label, amp_dtype=amp_dtype)
            ckpt_loss = val_l
            logger.info(f"Epoch {epoch:03d} | Train: {train_l:.4f} | Val: {val_l:.4f} "
                        f"| Label: {val_label:.4f} | KL: {val_kl:.4f}")

        scheduler.step()

        # Only count patience and allow early stopping after KL annealing ends.
        # During the ramp the model is still under-regularised and val_loss
        # improvements are not meaningful for convergence decisions.
        if config.model in ('GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF'):
            annealing_done = global_step >= mc.kl_anneal_steps
        else:
            annealing_done = True

        if ckpt_loss < best_val_loss:
            best_val_loss = ckpt_loss
            counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info("New best model saved.")
        else:
            if annealing_done:
                counter += 1
            if annealing_done and counter >= mc.patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

    model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
    evaluate_model(model, config, device, metadata, val_loader=val_loader)

def main():
    config = parse_args()
    set_seed(config.seed)
    train(config)


if __name__ == "__main__":
    main()
