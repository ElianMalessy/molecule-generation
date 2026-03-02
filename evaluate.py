"""
evaluate.py — evaluate a trained molecular VAE checkpoint.

Importable:
    from evaluate import evaluate_model

Standalone:
    python evaluate.py                  # all checkpoints under checkpoints/
    python evaluate.py --model FRATTVAE --dataset ZINC
    python evaluate.py --checkpoint checkpoints/ZINC/FRATTVAE/best.pth
"""

import torch
import random
import logging
import argparse
import os
from tqdm import tqdm
from molecule_benchmarks import Benchmarker, SmilesDataset

from models.gvae import GraphVAE, gvae_prepare_batch
from models.frattvae import FRATTVAE, build_frattvae_dataset, collate_pad_fn
from models.frattvae.utils.mask import create_mask
from utils.utils import Config, get_dataloaders, get_smiles_list
from utils.constants import MOSES_ATOM_DECODER, ZINC_ATOM_DECODER, ZINC_CHARGE_DECODER

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core evaluation function
# ---------------------------------------------------------------------------

def evaluate_model(model, config: Config, device, metadata, val_loader=None):
    """
    Run reconstruction accuracy + prior-sampling benchmarks for a trained model.

    Args:
        model:       Trained GraphVAE or FRATTVAE (already on `device`).
        config:      Config object (model, dataset, latent dims, etc.).
        device:      torch.device.
        metadata:    Dict returned by get_dataloaders (keys depend on model type).
        val_loader:  Validation DataLoader used for reconstruction eval; optional.
    """
    model.eval()

    # Decoders needed for GVAE sampling and reconstruction
    if config.model == 'GVAE':
        atom_decoder   = MOSES_ATOM_DECODER if config.dataset == 'MOSES' else ZINC_ATOM_DECODER
        charge_decoder = None               if config.dataset == 'MOSES' else ZINC_CHARGE_DECODER

    # ------------------------------------------------------------------
    # 1. Reconstruction accuracy on a held-out sample
    #    Encode real molecules -> decode -> check recovery.
    # ------------------------------------------------------------------
    if config.model == 'GVAE' and val_loader is not None:
        recon_correct = recon_total = 0
        logger.info("Evaluating reconstruction accuracy on validation set (up to 1000 molecules)...")
        with torch.no_grad():
            for data in val_loader:
                x_in, edge_index, edge_attr_in, batch, target_nodes, target_edges = \
                    gvae_prepare_batch(data, device, config.gvae.max_atoms)
                _, _, mu, logvar = model(x_in, edge_index, edge_attr_in, batch)
                z = mu  # Use mean (no noise) for reconstruction eval
                recon_smiles = model.sample_smiles(z, atom_decoder, charge_decoder)

                for smi in recon_smiles:
                    recon_total += 1
                    if smi:
                        recon_correct += 1
                if recon_total >= 1000:
                    break

        logger.info(f"Reconstruction: {recon_correct}/{recon_total} valid decodes "
                    f"({100 * recon_correct / max(1, recon_total):.1f}%) from posterior mean z")

    if config.model == 'FRATTVAE' and val_loader is not None:
        frag_ecfps_r = metadata['frag_ecfps'].to(device)
        ndummys_r    = metadata['ndummys'].to(device)
        model.set_labels(metadata['uni_fragments'])

        recon_correct = recon_total = 0
        logger.info("Evaluating reconstruction accuracy on validation set (up to 1000 molecules)...")
        with torch.no_grad():
            for frag_indices, positions, _ in val_loader:
                B, L = frag_indices.shape
                features_r    = frag_ecfps_r[frag_indices.flatten()].reshape(B, L, -1).to(device)
                positions_r   = positions.to(device=device, dtype=torch.float32)
                idx_with_root = torch.cat([torch.full((B, 1), -1), frag_indices], dim=1).to(device)
                _, _, src_pad_mask, _ = create_mask(idx_with_root, idx_with_root, pad_idx=0)

                _, mu, _ = model.encode(features_r, positions_r, src_pad_mask=src_pad_mask)
                recon_smiles = model.sequential_decode(
                    mu, frag_ecfps_r, ndummys_r,
                    max_nfrags=config.frattvae.max_nfrags, asSmiles=True,
                )

                for smi in recon_smiles:
                    recon_total += 1
                    if smi:
                        recon_correct += 1
                if recon_total >= 1000:
                    break

        logger.info(f"Reconstruction: {recon_correct}/{recon_total} valid decodes "
                    f"({100 * recon_correct / max(1, recon_total):.1f}%) from posterior mean z")

    # ------------------------------------------------------------------
    # 2. Prior sampling + benchmark suite
    # ------------------------------------------------------------------
    logger.info(f"Generating {config.num_samples} samples for benchmarking...")
    generated_smiles = []

    with torch.no_grad():
        if config.model == 'GVAE':
            z = torch.randn(config.num_samples, config.gvae.latent_dim).to(device)
            generated_smiles = model.sample_smiles(z, atom_decoder, charge_decoder)

        elif config.model == 'FRATTVAE':
            frag_ecfps = metadata['frag_ecfps'].to(device)
            ndummys    = metadata['ndummys'].to(device)
            model.set_labels(metadata['uni_fragments'])

            decode_batch = 256
            z_all = torch.randn(config.num_samples, config.frattvae.latent_dim)
            logger.info("Sampling FRATTVAE (sequential decode)...")
            for i in tqdm(range(0, config.num_samples, decode_batch), desc="Sampling FRATTVAE"):
                z_batch = z_all[i:i + decode_batch].to(device)
                smiles_batch = model.sequential_decode(
                    z_batch, frag_ecfps, ndummys,
                    max_nfrags=config.frattvae.max_nfrags,
                    asSmiles=True,
                )
                generated_smiles.extend(smiles_batch)

    valid_smiles  = [s for s in generated_smiles if s]
    validity      = len(valid_smiles) / max(1, len(generated_smiles))
    unique_smiles = set(valid_smiles)
    uniqueness    = len(unique_smiles) / max(1, len(valid_smiles))
    logger.info(f"Prior sampling validity:   {validity:.4f} "
                f"({len(valid_smiles)}/{len(generated_smiles)})")
    logger.info(f"Prior sampling uniqueness: {uniqueness:.4f} "
                f"({len(unique_smiles)} unique / {len(valid_smiles)} valid)")

    if not valid_smiles:
        logger.error("No valid molecules generated. Skipping benchmarks.")
        return

    if len(unique_smiles) < 100:
        logger.error(f"Too few unique molecules ({len(unique_smiles)}) — "
                     "likely posterior collapse. Skipping benchmarks.")
        return

    if config.max_train_mols > 0:
        logger.info("Skipping benchmarks (max_train_mols cap active — use full dataset for real evaluation).")
        return

    logger.info("Running molecule-benchmarks...")
    try:
        if config.dataset == 'MOSES':
            reference_data = SmilesDataset.load_moses_dataset()
        else:
            t_smi = random.sample(get_smiles_list('ZINC', 'train'), 10000)
            v_smi = random.sample(get_smiles_list('ZINC', 'val'), 10000)
            reference_data = SmilesDataset(train_smiles=t_smi, validation_smiles=v_smi)

        benchmarker = Benchmarker(
            dataset=reference_data,
            num_samples_to_generate=config.num_samples,
            device=device.type,
        )
        # Pass the full generated list (including None/invalid) so the benchmarker
        # can compute validity metrics correctly.  Pre-filtering would inflate valid_fraction.
        metrics = benchmarker.benchmark(generated_smiles)
        logger.info(f"Benchmark Results: {metrics}")
    except KeyboardInterrupt:
        logger.warning("Benchmarking interrupted by user.")
    except Exception as e:
        logger.warning(f"Benchmarking failed: {e}")


# ---------------------------------------------------------------------------
# Helpers for the standalone runner
# ---------------------------------------------------------------------------

def _build_model(config: Config, metadata, device):
    """Instantiate the right model architecture from config + metadata."""
    if config.model == 'GVAE':
        model = GraphVAE(
            num_node_features=metadata['num_nodes'],
            num_edge_features=metadata['num_edges'],
            latent_dim=config.gvae.latent_dim,
            max_atoms=config.gvae.max_atoms,
        )
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
        )
    return model.to(device)


def _discover_checkpoints(root: str = 'checkpoints') -> list[tuple[str, str, str]]:
    """
    Walk checkpoints/ and return (dataset, model_name, path) for every best.pth found.
    Expected layout: checkpoints/{DATASET}/{MODEL}/best.pth
    """
    found = []
    if not os.path.isdir(root):
        return found
    for dataset in sorted(os.listdir(root)):
        d_path = os.path.join(root, dataset)
        if not os.path.isdir(d_path):
            continue
        for model_name in sorted(os.listdir(d_path)):
            ckpt = os.path.join(d_path, model_name, 'best.pth')
            if os.path.isfile(ckpt):
                found.append((dataset, model_name, ckpt))
    return found


def run_validation(dataset: str, model_name: str, checkpoint: str,
                   num_samples: int = 10000, num_workers: int = 4,
                   device_str: str | None = None):
    """
    Load a checkpoint and run the full evaluation suite.

    Args:
        dataset:     'ZINC' or 'MOSES'
        model_name:  'GVAE' or 'FRATTVAE'
        checkpoint:  Path to best.pth
        num_samples: Number of molecules to sample for benchmarks
        num_workers: DataLoader worker count
        device_str:  e.g. 'cuda', 'cpu'; auto-detected if None
    """
    logger.info(f"=== Evaluating {model_name} on {dataset} from {checkpoint} ===")

    device = torch.device(device_str if device_str else ('cuda' if torch.cuda.is_available() else 'cpu'))

    config = Config(model=model_name, dataset=dataset, num_samples=num_samples, num_workers=num_workers)
    if dataset == 'MOSES':
        config.gvae.max_atoms = 30

    _, val_loader, metadata = get_dataloaders(config, logger)

    model = _build_model(config, metadata, device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    logger.info(f"Loaded weights from {checkpoint}")

    evaluate_model(model, config, device, metadata, val_loader=val_loader)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained molecular VAE checkpoints")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--checkpoint', type=str, default=None,
                       help='Path to a specific best.pth to evaluate')
    group.add_argument('--all', action='store_true', default=True,
                       help='Evaluate all checkpoints found under checkpoints/ (default)')
    parser.add_argument('--model',   type=str, choices=['GVAE', 'FRATTVAE'], default=None,
                        help='Filter to a specific model (used with --all)')
    parser.add_argument('--dataset', type=str, choices=['ZINC', 'MOSES'],    default=None,
                        help='Filter to a specific dataset (used with --all or with --checkpoint)')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    common = dict(
        num_samples=args.num_samples,
        num_workers=args.num_workers,
        device_str=args.device,
    )

    if args.checkpoint:
        # Single explicit checkpoint; infer dataset/model from path if not supplied
        parts = args.checkpoint.replace('\\', '/').split('/')
        # Expect …/{DATASET}/{MODEL}/best.pth
        try:
            model_name = args.model   or parts[-2]
            dataset    = args.dataset or parts[-3]
        except IndexError:
            raise ValueError(
                "Could not infer dataset/model from checkpoint path. "
                "Pass --model and --dataset explicitly."
            )
        run_validation(dataset, model_name, args.checkpoint, **common)
    else:
        checkpoints = _discover_checkpoints()
        if not checkpoints:
            logger.error("No checkpoints found under checkpoints/. "
                         "Train a model first or pass --checkpoint explicitly.")
        for dataset, model_name, ckpt in checkpoints:
            if args.model   and model_name != args.model:
                continue
            if args.dataset and dataset    != args.dataset:
                continue
            run_validation(dataset, model_name, ckpt, **common)
