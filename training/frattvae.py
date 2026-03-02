import torch
import torch.nn as nn
import logging
from tqdm import tqdm

from models.frattvae import batched_kl_divergence
from models.frattvae.utils.mask import create_mask
from utils.utils import Config

logger = logging.getLogger(__name__)


def _frattvae_criterion(freq_label: torch.Tensor, device) -> nn.CrossEntropyLoss:
    """Frequency-weighted CrossEntropyLoss for fragment token prediction."""
    freq = freq_label.clone().clamp(max=1000)
    weight = freq.max() / freq
    weight[~torch.isfinite(weight)] = 0.001
    return nn.CrossEntropyLoss(weight=weight.to(device))


def train_epoch_frattvae(model, optimizer, loader, config: Config, global_step: int,
                         device, frag_ecfps, freq_label, amp_dtype=None):
    model.train()
    criterion  = _frattvae_criterion(freq_label, device)
    num_tokens = frag_ecfps.shape[0]
    total_loss = total_kl = total_label = 0.0

    for frag_indices, positions, _ in tqdm(loader, desc="Train FRATTVAE", leave=False):
        B, L = frag_indices.shape
        features      = frag_ecfps[frag_indices.flatten()].reshape(B, L, -1).to(device)
        positions     = positions.to(device=device, dtype=torch.float32)
        target        = torch.cat([frag_indices, torch.zeros(B, 1)], dim=1).flatten().long().to(device)
        idx_with_root = torch.cat([torch.full((B, 1), -1), frag_indices], dim=1).to(device)

        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(idx_with_root, idx_with_root, pad_idx=0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            z, mu, ln_var, output = model(features, positions, src_mask, src_pad_mask, tgt_mask, tgt_pad_mask)

        # Compute both losses in float32: label_loss uses softmax over many tokens (overflow risk
        # in bfloat16); KL uses exp(ln_var) which also overflows in bfloat16.
        kl_weight  = config.frattvae.kl_weight
        kl_loss    = batched_kl_divergence(mu.float(), ln_var.float())
        label_loss = criterion(output.float().view(-1, num_tokens), target)
        loss       = kl_weight * kl_loss + config.frattvae.label_loss_weight * label_loss

        if not torch.isfinite(loss):
            logger.warning(f"Non-finite loss ({loss.item():.4g}) at step {global_step} — skipping batch.")
            optimizer.zero_grad()
            global_step += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        # Guard against NaN/inf gradients (e.g. from compiled backward CUDA graph shape mismatches).
        # clip_grad_norm_ returns the total grad norm — if NaN, skip the step to prevent
        # Adam moment corruption, which would permanently poison all future updates.
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        if not torch.isfinite(grad_norm):
            logger.warning(f"Non-finite grad norm at step {global_step} — skipping optimizer step.")
            optimizer.zero_grad()
            global_step += 1
            continue

        optimizer.step()
        global_step += 1

        total_loss  += loss.item()        * B
        total_kl    += kl_loss.item()     * B
        total_label += label_loss.item()  * B

    n = len(loader.dataset)
    return total_loss / n, total_label / n, total_kl / n, global_step


@torch.no_grad()
def val_epoch_frattvae(model, loader, config: Config, global_step: int,
                       device, frag_ecfps, freq_label, amp_dtype=None):
    model.eval()
    criterion  = _frattvae_criterion(freq_label, device)
    num_tokens = frag_ecfps.shape[0]
    total_loss = total_kl = total_label = 0.0

    for frag_indices, positions, _ in tqdm(loader, desc="Val FRATTVAE", leave=False):
        B, L = frag_indices.shape
        features      = frag_ecfps[frag_indices.flatten()].reshape(B, L, -1).to(device)
        positions     = positions.to(device=device, dtype=torch.float32)
        target        = torch.cat([frag_indices, torch.zeros(B, 1)], dim=1).flatten().long().to(device)
        idx_with_root = torch.cat([torch.full((B, 1), -1), frag_indices], dim=1).to(device)

        src_mask, tgt_mask, src_pad_mask, tgt_pad_mask = create_mask(idx_with_root, idx_with_root, pad_idx=0)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            # Use parallel (teacher-forcing) decode during validation for speed
            z, mu, ln_var, output = model(features, positions, src_mask, src_pad_mask, tgt_mask, tgt_pad_mask,
                                          sequential=False)

        # Compute both losses in float32 (same reason as train).
        kl_weight  = config.frattvae.kl_weight
        kl_loss    = batched_kl_divergence(mu.float(), ln_var.float())
        label_loss = criterion(output.float().view(-1, num_tokens), target)
        loss       = kl_weight * kl_loss + config.frattvae.label_loss_weight * label_loss

        total_loss  += loss.item()        * B
        total_kl    += kl_loss.item()     * B
        total_label += label_loss.item()  * B

    n = len(loader.dataset)
    return total_loss / n, total_label / n, total_kl / n
