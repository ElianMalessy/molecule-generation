import torch
import torch.nn.functional as F
from tqdm import tqdm

from models.gvae_nf import gvae_nf_loss
from models.gvae import gvae_prepare_batch
from utils.utils import Config, cyclical_beta
from utils.properties import prop_gamma, normalise_props


def train_epoch_gvae_nf(model, optimizer, loader, config: Config, global_step: int,
                        device, amp_dtype=None, epoch: int = 1,
                        prop_mean=None, prop_std=None):
    model.train()
    total_loss = total_recon = total_kl = total_prop = total_raw_prop = 0.0
    gamma = prop_gamma(epoch, config.gvae_nf.prop_warmup_epochs, config.gvae_nf.prop_weight)

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

            raw_prop_loss = torch.tensor(0.0, device=device)
            if config.gvae_nf.prop_pred and hasattr(data, 'props'):
                true_z = normalise_props(data.props.to(device, dtype=torch.float32),
                                         prop_mean, prop_std)
                raw_prop_loss = F.mse_loss(model.predict_props(mu), true_z)
                if gamma > 0:
                    loss = loss + gamma * raw_prop_loss

        loss.backward()
        optimizer.step()
        global_step += 1

        total_loss      += loss.item()                       * data.num_graphs
        total_recon     += recon.item()                      * data.num_graphs
        total_kl        += (kl_weight * kl).item()           * data.num_graphs
        total_prop      += (gamma * raw_prop_loss).item()    * data.num_graphs
        total_raw_prop  += raw_prop_loss.item()              * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n, total_prop / n, total_raw_prop / n, global_step


@torch.no_grad()
def val_epoch_gvae_nf(model, loader, config: Config, global_step: int,
                      device, amp_dtype=None, epoch: int = 1,
                      prop_mean=None, prop_std=None):
    model.eval()
    total_loss = total_recon = total_kl = total_prop = total_raw_prop = 0.0
    gamma = prop_gamma(epoch, config.gvae_nf.prop_warmup_epochs, config.gvae_nf.prop_weight)

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

            raw_prop_loss = torch.tensor(0.0, device=device)
            if config.gvae_nf.prop_pred and hasattr(data, 'props'):
                true_z = normalise_props(data.props.to(device, dtype=torch.float32),
                                         prop_mean, prop_std)
                raw_prop_loss = F.mse_loss(model.predict_props(mu), true_z)
                if gamma > 0:
                    loss = loss + gamma * raw_prop_loss

        total_loss      += loss.item()                       * data.num_graphs
        total_recon     += recon.item()                      * data.num_graphs
        total_kl        += (beta * kl).item()                * data.num_graphs
        total_prop      += (gamma * raw_prop_loss).item()    * data.num_graphs
        total_raw_prop  += raw_prop_loss.item()              * data.num_graphs

    n = len(loader.dataset)
    return total_loss / n, total_recon / n, total_kl / n, total_prop / n, total_raw_prop / n
