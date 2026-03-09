import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.gvae_ar import GraphVAEARNF, gvae_ar_loss, gvae_ar_nf_loss
from utils.utils import Config
from utils.properties import normalise_props

logger = logging.getLogger(__name__)


def train_epoch_gvae_ar(model, optimizer, loader, config: Config, global_step: int,
                        device, amp_dtype=None, epoch: int = 1,
                        prop_mean=None, prop_std=None, node_class_weights=None,
                        edge_class_weights=None):
    model.train()
    total_loss = total_recon = total_kl = total_true_kl = total_raw_prop = 0.0
    n_skipped = 0
    use_nf = isinstance(model, GraphVAEARNF)
    mc     = config.gvae_ar_nf if use_nf else config.gvae_ar
    gamma  = mc.prop_weight
    desc   = "Train GVAE_AR_NF" if use_nf else "Train GVAE_AR"

    for batch_data in tqdm(loader, desc=desc, leave=False):
        optimizer.zero_grad()
        pyg_batch, input_tokens, target_tokens, target_types, seq_lens = batch_data
        pyg_batch     = pyg_batch.to(device)
        x_in          = pyg_batch.x.squeeze(-1) + 1
        edge_attr_in  = pyg_batch.edge_attr.squeeze(-1) + 1
        input_tokens  = input_tokens.to(device)
        target_tokens = target_tokens.to(device)
        target_types  = target_types.to(device)
        seq_lens      = seq_lens.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            # β-annealing: ramp β from 0 → kl_weight over kl_anneal_steps.
            # This forces the encoder to learn an informative posterior (by minimising
            # reconstruction at β=0) before the KL penalty regularises it toward N(0,I).
            # The capacity hinge (β·|KL−C|) is NOT used for AR models because it allows
            # Mutual Information Collapse: the encoder can satisfy |KL−C|=0 by outputting
            # a constant μ for every input — zero MI — which is exactly what we observed
            # (all reconstructions identical despite diverse prior samples).
            beta = mc.kl_weight * min(1.0, global_step / max(1, mc.kl_anneal_steps))
            if use_nf:
                recon, mu, logvar, z0, zK, sum_log_det = model(
                    x_in, pyg_batch.edge_index, edge_attr_in, pyg_batch.batch,
                    input_tokens, target_tokens, target_types, seq_lens,
                    node_class_weights=node_class_weights,
                    edge_class_weights=edge_class_weights)
                loss, _, kl, true_kl = gvae_ar_nf_loss(recon, mu, logvar, z0, zK, sum_log_det,
                                              beta, free_bits=mc.free_bits_per_dim)
            else:
                recon, mu, logvar = model(
                    x_in, pyg_batch.edge_index, edge_attr_in, pyg_batch.batch,
                    input_tokens, target_tokens, target_types, seq_lens,
                    node_class_weights=node_class_weights,
                    edge_class_weights=edge_class_weights)
                loss, _, kl, true_kl = gvae_ar_loss(recon, mu, logvar, beta,
                                           free_bits=mc.free_bits_per_dim)

            raw_prop_loss = torch.tensor(0.0, device=device)
            if mc.prop_pred and hasattr(pyg_batch, 'props'):
                true_z = normalise_props(pyg_batch.props.to(device, dtype=torch.float32),
                                         prop_mean, prop_std)
                raw_prop_loss = F.mse_loss(model.predict_props(mu), true_z)
                prop_loss = gamma * raw_prop_loss
                loss += prop_loss

        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad()
            global_step += 1
            continue

        loss.backward()
        # Clip encoder+decoder only.  The prop head has a tiny gradient norm relative
        # to the AR Transformer decoder (many more tokens → much larger total norm).
        # Clipping all parameters together scales prop head gradients to near-zero,
        # preventing the property head from learning.  Clip just the main backbone.
        prop_param_ids = (
            {id(p) for p in model.prop_head.parameters()}
            if (mc.prop_pred and getattr(model, 'prop_head', None) is not None)
            else set()
        )

        main_params = [p for p in model.parameters() if id(p) not in prop_param_ids]
        grad_norm = torch.nn.utils.clip_grad_norm_(main_params, 5.0)

        # Check if ANY parameter has a non-finite gradient
        has_nan_grad = any(
            p.grad is not None and not torch.isfinite(p.grad).all() 
            for p in model.parameters()
        )

        if not torch.isfinite(grad_norm) or has_nan_grad:
            n_skipped += 1
            optimizer.zero_grad()
            global_step += 1
            continue

        optimizer.step()
        global_step += 1

        total_loss     += loss.item()                    * pyg_batch.num_graphs
        total_recon    += recon.item()                   * pyg_batch.num_graphs
        total_kl       += kl.item()                       * pyg_batch.num_graphs
        total_true_kl  += true_kl.item()                  * pyg_batch.num_graphs
        total_raw_prop += raw_prop_loss.item()                * pyg_batch.num_graphs

    if n_skipped > 0:
        logger.warning(f"Skipped {n_skipped}/{len(loader)} batches due to non-finite loss/gradients.")
    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n, total_true_kl / n, total_raw_prop / n, global_step


@torch.no_grad()
def val_epoch_gvae_ar(model, loader, config: Config, global_step: int,
                      device, amp_dtype=None, epoch: int = 1,
                      prop_mean=None, prop_std=None, node_class_weights=None,
                      edge_class_weights=None):
    model.eval()
    total_loss = total_recon = total_kl = total_true_kl = total_raw_prop = 0.0
    use_nf = isinstance(model, GraphVAEARNF)
    mc     = config.gvae_ar_nf if use_nf else config.gvae_ar
    gamma  = mc.prop_weight
    desc   = "Val GVAE_AR_NF" if use_nf else "Val GVAE_AR"

    for batch_data in tqdm(loader, desc=desc, leave=False):
        pyg_batch, input_tokens, target_tokens, target_types, seq_lens = batch_data
        pyg_batch     = pyg_batch.to(device)
        x_in          = pyg_batch.x.squeeze(-1) + 1
        edge_attr_in  = pyg_batch.edge_attr.squeeze(-1) + 1
        input_tokens  = input_tokens.to(device)
        target_tokens = target_tokens.to(device)
        target_types  = target_types.to(device)
        seq_lens      = seq_lens.to(device)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            beta = mc.kl_weight * min(1.0, global_step / max(1, mc.kl_anneal_steps))
            if use_nf:
                recon, mu, logvar, z0, zK, sum_log_det = model(
                    x_in, pyg_batch.edge_index, edge_attr_in, pyg_batch.batch,
                    input_tokens, target_tokens, target_types, seq_lens,
                    node_class_weights=node_class_weights,
                    edge_class_weights=edge_class_weights)
                loss, _, kl, true_kl = gvae_ar_nf_loss(recon, mu, logvar, z0, zK, sum_log_det,
                                              beta, free_bits=mc.free_bits_per_dim)
            else:
                recon, mu, logvar = model(
                    x_in, pyg_batch.edge_index, edge_attr_in, pyg_batch.batch,
                    input_tokens, target_tokens, target_types, seq_lens,
                    node_class_weights=node_class_weights,
                    edge_class_weights=edge_class_weights)
                loss, _, kl, true_kl = gvae_ar_loss(recon, mu, logvar, beta,
                                           free_bits=mc.free_bits_per_dim)

            raw_prop_loss = torch.tensor(0.0, device=device)
            if mc.prop_pred and hasattr(pyg_batch, 'props'):
                true_z = normalise_props(pyg_batch.props.to(device, dtype=torch.float32),
                                         prop_mean, prop_std)
                raw_prop_loss = F.mse_loss(model.predict_props(mu), true_z)
                prop_loss = gamma * raw_prop_loss
                loss += prop_loss

        total_loss     += loss.item()                    * pyg_batch.num_graphs
        total_recon    += recon.item()                   * pyg_batch.num_graphs
        total_kl       += kl.item()                       * pyg_batch.num_graphs
        total_true_kl  += true_kl.item()                  * pyg_batch.num_graphs
        total_raw_prop += raw_prop_loss.item()                * pyg_batch.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n, total_true_kl / n, total_raw_prop / n
