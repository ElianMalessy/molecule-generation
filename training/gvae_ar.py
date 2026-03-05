"""Training epoch functions for GraphVAEAR and GraphVAEARNF.

A single pair of train/val functions handles both AR variants by dispatching
on isinstance(model, GraphVAEARNF).  The correct config section (config.gvae_ar
or config.gvae_ar_nf) is selected automatically.

The DataLoader uses ar_collate_fn (see utils/utils.py) so BFS serialization
runs in worker processes.  Each iteration yields:
    (pyg_batch, input_tokens, target_tokens, target_types, seq_lens)
all as CPU tensors; we move them to device here.
"""
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.gvae_ar import GraphVAEARNF, gvae_ar_loss, gvae_ar_nf_loss
from utils.utils import Config, kl_capacity
from utils.properties import prop_gamma, normalise_props


def train_epoch_gvae_ar(model, optimizer, loader, config: Config, global_step: int,
                        device, amp_dtype=None, epoch: int = 1,
                        prop_mean=None, prop_std=None):
    model.train()
    total_loss = total_recon = total_kl = total_prop = total_raw_prop = 0.0
    total_prop_gnorm = 0.0
    n_batches = 0
    use_nf = isinstance(model, GraphVAEARNF)
    mc     = config.gvae_ar_nf if use_nf else config.gvae_ar
    gamma  = prop_gamma(epoch, mc.prop_warmup_epochs, mc.prop_weight)
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
            kl_weight = mc.kl_weight
            capacity  = kl_capacity(global_step, mc.kl_capacity_max, mc.kl_anneal_steps)
            if use_nf:
                recon, mu, logvar, z0, zK, sum_log_det = model(
                    x_in, pyg_batch.edge_index, edge_attr_in, pyg_batch.batch,
                    input_tokens, target_tokens, target_types, seq_lens)
                loss, _, kl = gvae_ar_nf_loss(recon, mu, logvar, z0, zK, sum_log_det,
                                              kl_weight, free_bits=mc.free_bits_per_dim,
                                              capacity=capacity)
            else:
                recon, mu, logvar = model(
                    x_in, pyg_batch.edge_index, edge_attr_in, pyg_batch.batch,
                    input_tokens, target_tokens, target_types, seq_lens)
                loss, _, kl = gvae_ar_loss(recon, mu, logvar, kl_weight,
                                           free_bits=mc.free_bits_per_dim, capacity=capacity)

            raw_prop_loss = torch.tensor(0.0, device=device)
            if mc.prop_pred and hasattr(pyg_batch, 'props'):
                true_z = normalise_props(pyg_batch.props.to(device, dtype=torch.float32),
                                         prop_mean, prop_std)
                raw_prop_loss = F.mse_loss(model.predict_props(mu), true_z)
                if gamma > 0:
                    loss = loss + gamma * raw_prop_loss

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
        torch.nn.utils.clip_grad_norm_(main_params, 5.0)

        # Prop head gradient norm (0.0 during warmup when head not in loss)
        prop_gnorm = 0.0
        if prop_param_ids and gamma > 0:
            sq = [p.grad.detach().norm().item() ** 2
                  for p in model.prop_head.parameters() if p.grad is not None]
            prop_gnorm = sum(sq) ** 0.5 if sq else 0.0
        total_prop_gnorm += prop_gnorm
        n_batches += 1

        optimizer.step()
        global_step += 1

        total_loss     += loss.item()                    * pyg_batch.num_graphs
        total_recon    += recon.item()                   * pyg_batch.num_graphs
        total_kl       += kl.item()                       * pyg_batch.num_graphs
        total_prop     += (gamma * raw_prop_loss).item() * pyg_batch.num_graphs
        total_raw_prop += raw_prop_loss.item()           * pyg_batch.num_graphs

    n = len(loader.dataset)
    return (total_loss / n, total_recon / n, total_kl / n,
            total_prop / n, total_raw_prop / n,
            total_prop_gnorm / max(1, n_batches), global_step)


@torch.no_grad()
def val_epoch_gvae_ar(model, loader, config: Config, global_step: int,
                      device, amp_dtype=None, epoch: int = 1,
                      prop_mean=None, prop_std=None):
    model.eval()
    total_loss = total_recon = total_kl = total_prop = total_raw_prop = 0.0
    use_nf = isinstance(model, GraphVAEARNF)
    mc     = config.gvae_ar_nf if use_nf else config.gvae_ar
    gamma  = prop_gamma(epoch, mc.prop_warmup_epochs, mc.prop_weight)
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
            beta     = mc.kl_weight
            capacity = kl_capacity(global_step, mc.kl_capacity_max, mc.kl_anneal_steps)
            if use_nf:
                recon, mu, logvar, z0, zK, sum_log_det = model(
                    x_in, pyg_batch.edge_index, edge_attr_in, pyg_batch.batch,
                    input_tokens, target_tokens, target_types, seq_lens)
                loss, _, kl = gvae_ar_nf_loss(recon, mu, logvar, z0, zK, sum_log_det,
                                              beta, free_bits=mc.free_bits_per_dim,
                                              capacity=capacity)
            else:
                recon, mu, logvar = model(
                    x_in, pyg_batch.edge_index, edge_attr_in, pyg_batch.batch,
                    input_tokens, target_tokens, target_types, seq_lens)
                loss, _, kl = gvae_ar_loss(recon, mu, logvar, beta,
                                           free_bits=mc.free_bits_per_dim, capacity=capacity)

            raw_prop_loss = torch.tensor(0.0, device=device)
            if mc.prop_pred and hasattr(pyg_batch, 'props'):
                true_z = normalise_props(pyg_batch.props.to(device, dtype=torch.float32),
                                         prop_mean, prop_std)
                raw_prop_loss = F.mse_loss(model.predict_props(mu), true_z)
                if gamma > 0:
                    loss = loss + gamma * raw_prop_loss

        total_loss     += loss.item()                    * pyg_batch.num_graphs
        total_recon    += recon.item()                   * pyg_batch.num_graphs
        total_kl       += kl.item()                       * pyg_batch.num_graphs
        total_prop     += (gamma * raw_prop_loss).item() * pyg_batch.num_graphs
        total_raw_prop += raw_prop_loss.item()           * pyg_batch.num_graphs

    n = len(loader.dataset)
    return (total_loss / n, total_recon / n, total_kl / n,
            total_prop / n, total_raw_prop / n)
