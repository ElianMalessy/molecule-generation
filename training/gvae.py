"""
Training epoch functions for GraphVAE and GraphVAENF.

A single pair of train/val functions handles both variants by dispatching
on isinstance(model, GraphVAENF).  The correct config section (config.gvae
or config.gvae_nf) is selected automatically.
"""
import logging
import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.gvae import GraphVAENF, gvae_loss, gvae_nf_loss, gvae_prepare_batch
from utils.utils import Config, kl_capacity
from utils.properties import normalise_props

logger = logging.getLogger(__name__)


def train_epoch_gvae(model, optimizer, loader, config: Config, global_step: int,
                     device, amp_dtype=None, epoch: int = 1,
                     prop_mean=None, prop_std=None, node_class_weights=None,
                     edge_class_weights=None):
    model.train()
    total_loss = total_recon = total_kl = total_true_kl = total_raw_prop = 0.0
    total_prop_gnorm = 0.0
    n_batches = 0
    use_nf = isinstance(model, GraphVAENF)
    mc     = config.gvae_nf if use_nf else config.gvae
    gamma  = mc.prop_weight
    desc   = "Train GVAE-NF" if use_nf else "Train GVAE"

    for data in tqdm(loader, desc=desc, leave=False):
        optimizer.zero_grad()
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = \
            gvae_prepare_batch(data, device, mc.max_atoms)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            kl_weight = mc.kl_weight
            capacity  = kl_capacity(global_step, mc.kl_capacity_max, mc.kl_anneal_steps)
            if use_nf:
                node_logits, edge_logits, mu, logvar, z0, zK, sum_log_det = \
                    model(x_in, edge_index, edge_attr_in, batch)
                loss, recon, kl, true_kl = gvae_nf_loss(
                    node_logits, edge_logits, target_nodes, target_edges,
                    mu, logvar, z0, zK, sum_log_det, kl_weight,
                    free_bits=mc.free_bits_per_dim, capacity=capacity,
                    node_class_weights=node_class_weights,
                    edge_class_weights=edge_class_weights,
                )
            else:
                node_logits, edge_logits, mu, logvar = \
                    model(x_in, edge_index, edge_attr_in, batch)
                loss, recon, kl, true_kl = gvae_loss(
                    node_logits, edge_logits, target_nodes, target_edges,
                    mu, logvar, kl_weight, free_bits=mc.free_bits_per_dim, capacity=capacity,
                    node_class_weights=node_class_weights,
                    edge_class_weights=edge_class_weights,
                )

            raw_prop_loss = torch.tensor(0.0, device=device)
            if mc.prop_pred and hasattr(data, 'props'):
                true_z = normalise_props(data.props.to(device, dtype=torch.float32),
                                         prop_mean, prop_std)
                raw_prop_loss = F.mse_loss(model.predict_props(mu), true_z)
                prop_loss = gamma * raw_prop_loss
                loss += prop_loss

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss ({loss.item():.4g}) at step {global_step} — skipping batch.")
            optimizer.zero_grad()
            global_step += 1
            continue

        loss.backward()

        prop_param_ids = ({id(p) for p in model.prop_head.parameters()}
                         if mc.prop_pred and getattr(model, 'prop_head', None) else set())
        main_params = [p for p in model.parameters() if id(p) not in prop_param_ids]
        grad_norm = torch.nn.utils.clip_grad_norm_(main_params, 5.0)

        # Guard against NaN/inf gradients.  Adam moment corruption from a single
        # NaN step would permanently poison all future updates.
        if not torch.isfinite(grad_norm):
            logger.warning(f"Non-finite grad norm at step {global_step} — skipping optimizer step.")
            optimizer.zero_grad()
            global_step += 1
            continue

        optimizer.step()
        global_step += 1

        total_loss     += loss.item()                    * data.num_graphs
        total_recon    += recon.item()                   * data.num_graphs
        total_kl       += kl.item()                       * data.num_graphs
        total_true_kl  += true_kl.item()                  * data.num_graphs
        total_raw_prop     += raw_prop_loss.item()                * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n, total_true_kl / n, total_raw_prop / n, global_step



@torch.no_grad()
def val_epoch_gvae(model, loader, config: Config, global_step: int, device,
                   amp_dtype=None, epoch: int = 1, prop_mean=None, prop_std=None,
                   node_class_weights=None, edge_class_weights=None):
    model.eval()
    total_loss = total_recon = total_kl = total_true_kl = total_raw_prop = 0.0
    use_nf = isinstance(model, GraphVAENF)
    mc     = config.gvae_nf if use_nf else config.gvae
    gamma  = mc.prop_weight
    desc   = "Val GVAE-NF" if use_nf else "Val GVAE"

    for data in tqdm(loader, desc=desc, leave=False):
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = \
            gvae_prepare_batch(data, device, mc.max_atoms)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            beta     = mc.kl_weight
            capacity = kl_capacity(global_step, mc.kl_capacity_max, mc.kl_anneal_steps)
            if use_nf:
                node_logits, edge_logits, mu, logvar, z0, zK, sum_log_det = \
                    model(x_in, edge_index, edge_attr_in, batch)
                loss, recon, kl, true_kl = gvae_nf_loss(
                    node_logits, edge_logits, target_nodes, target_edges,
                    mu, logvar, z0, zK, sum_log_det, beta,
                    free_bits=mc.free_bits_per_dim, capacity=capacity,
                    node_class_weights=node_class_weights,
                    edge_class_weights=edge_class_weights,
                )
            else:
                node_logits, edge_logits, mu, logvar = \
                    model(x_in, edge_index, edge_attr_in, batch)
                loss, recon, kl, true_kl = gvae_loss(
                    node_logits, edge_logits, target_nodes, target_edges,
                    mu, logvar, beta, free_bits=mc.free_bits_per_dim, capacity=capacity,
                    node_class_weights=node_class_weights,
                    edge_class_weights=edge_class_weights,
                )

            raw_prop_loss = torch.tensor(0.0, device=device)
            if mc.prop_pred and hasattr(data, 'props'):
                true_z = normalise_props(data.props.to(device, dtype=torch.float32),
                                         prop_mean, prop_std)
                raw_prop_loss = F.mse_loss(model.predict_props(mu), true_z)
                prop_loss = gamma * raw_prop_loss
                loss += prop_loss

        total_loss     += loss.item()                    * data.num_graphs
        total_recon    += recon.item()                   * data.num_graphs
        total_kl       += kl.item()                       * data.num_graphs
        total_true_kl  += true_kl.item()                  * data.num_graphs
        total_raw_prop     += raw_prop_loss.item()                * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n, total_true_kl / n, total_raw_prop / n
