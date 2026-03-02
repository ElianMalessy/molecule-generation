import torch
from tqdm import tqdm

from models.gvae import gvae_loss, gvae_prepare_batch
from utils.utils import Config


def train_epoch_gvae(model, optimizer, loader, config: Config, global_step: int,
                     device, amp_dtype=None):
    model.train()
    total_loss = total_recon = total_kl = 0.0

    for data in tqdm(loader, desc="Train GVAE", leave=False):
        optimizer.zero_grad()
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = \
            gvae_prepare_batch(data, device, config.gvae.max_atoms)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            node_logits, edge_logits, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
            kl_weight = min(config.gvae.kl_weight, global_step / config.gvae.kl_anneal_steps)
            loss, recon, kl = gvae_loss(node_logits, edge_logits, target_nodes, target_edges,
                                        mu, logvar, kl_weight)

        loss.backward()
        optimizer.step()
        global_step += 1

        total_loss  += loss.item()  * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl    += kl.item()   * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n, global_step


@torch.no_grad()
def val_epoch_gvae(model, loader, config: Config, global_step: int, device, amp_dtype=None):
    model.eval()
    total_loss = total_recon = total_kl = 0.0

    for data in tqdm(loader, desc="Val GVAE", leave=False):
        x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = \
            gvae_prepare_batch(data, device, config.gvae.max_atoms)

        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            node_logits, edge_logits, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
            beta = min(config.gvae.kl_weight, global_step / config.gvae.kl_anneal_steps)
            loss, recon, kl = gvae_loss(node_logits, edge_logits, target_nodes, target_edges,
                                        mu, logvar, beta)

        total_loss  += loss.item()  * data.num_graphs
        total_recon += recon.item() * data.num_graphs
        total_kl    += kl.item()   * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n
