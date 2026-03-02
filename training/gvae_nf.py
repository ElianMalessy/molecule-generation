import torch
from tqdm import tqdm

from models.gvae_nf import gvae_nf_loss
from models.gvae import gvae_prepare_batch
from utils.utils import Config, cyclical_beta


def train_epoch_gvae_nf(model, optimizer, loader, config: Config, global_step: int,
                        device, amp_dtype=None):
    model.train()
    total_loss = total_recon = total_kl = 0.0

    for data in tqdm(loader, desc="Train GVAE-NF", leave=False):
        optimizer.zero_grad()
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = \
            gvae_prepare_batch(data, device, config.gvae_nf.max_atoms)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            node_logits, edge_logits, mu, logvar, z0, zK, sum_log_det = \
                model(x_in, edge_index, edge_attr_in, batch)
            kl_weight = cyclical_beta(global_step, config.gvae_nf.kl_anneal_steps,
                                      config.gvae_nf.kl_weight, config.gvae_nf.kl_cycles,
                                      config.gvae_nf.kl_anneal_ratio)
            loss, recon, kl = gvae_nf_loss(
                node_logits, edge_logits, target_nodes, target_edges,
                mu, logvar, z0, zK, sum_log_det, kl_weight,
            )

        loss.backward()
        optimizer.step()
        global_step += 1

        total_loss  += loss.item()  * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl    += kl.item()   * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n, global_step


@torch.no_grad()
def val_epoch_gvae_nf(model, loader, config: Config, global_step: int,
                      device, amp_dtype=None):
    model.eval()
    total_loss = total_recon = total_kl = 0.0

    for data in tqdm(loader, desc="Val GVAE-NF", leave=False):
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = \
            gvae_prepare_batch(data, device, config.gvae_nf.max_atoms)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            node_logits, edge_logits, mu, logvar, z0, zK, sum_log_det = \
                model(x_in, edge_index, edge_attr_in, batch)
            beta = cyclical_beta(global_step, config.gvae_nf.kl_anneal_steps,
                                 config.gvae_nf.kl_weight, config.gvae_nf.kl_cycles,
                                 config.gvae_nf.kl_anneal_ratio)
            loss, recon, kl = gvae_nf_loss(
                node_logits, edge_logits, target_nodes, target_edges,
                mu, logvar, z0, zK, sum_log_det, beta,
            )

        total_loss  += loss.item()  * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl    += kl.item()   * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n
