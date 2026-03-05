import torch
import random
import logging
import argparse
import os
import json
from datetime import datetime, timezone
from tqdm import tqdm
from molecule_benchmarks import Benchmarker, SmilesDataset

from models.gvae import GraphVAE, GraphVAENF, gvae_prepare_batch
from models.gvae_ar import GraphVAEAR, GraphVAEARNF
from models.frattvae import FRATTVAE, build_frattvae_dataset, collate_pad_fn
from models.frattvae.utils.mask import create_mask
from utils.utils import Config, get_dataloaders, get_smiles_list
from utils.constants import MOSES_ATOM_DECODER, ZINC_ATOM_DECODER, ZINC_CHARGE_DECODER

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def evaluate_model(model, config: Config, device, metadata, val_loader=None):
    """
    Run prior-sampling benchmarks for a trained model.

    Args:
        model:      Trained GraphVAE, GraphVAENF, or FRATTVAE (already on `device`).
        config:     Config object (model, dataset, latent dims, etc.).
        device:     torch.device.
        metadata:   Dict returned by get_dataloaders.
        val_loader: Optional validation DataLoader used to compute reconstruction rate
                    (only used for GVAE and GVAE_NF; ignored for AR/FRATTVAE).
    """
    model.eval()

    # Decoders needed for GVAE/GVAE_NF/GVAE_AR sampling
    if config.model in ('GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF'):
        atom_decoder   = MOSES_ATOM_DECODER if config.dataset == 'MOSES' else ZINC_ATOM_DECODER
        charge_decoder = None               if config.dataset == 'MOSES' else ZINC_CHARGE_DECODER

    # ------------------------------------------------------------------
    # Reconstruction rate (encode val set → decode from μ → compare)
    # ------------------------------------------------------------------
    recon_rate = None
    if val_loader is not None:
        recon_rate = _compute_recon_rate(model, val_loader, config, device, metadata)
        if recon_rate is not None:
            logger.info(f"Reconstruction rate (μ-decode on val set): {recon_rate:.4f}")

    # ------------------------------------------------------------------
    # Prior sampling + benchmark suite
    # ------------------------------------------------------------------
    logger.info(f"Generating {config.num_samples} samples for benchmarking...")
    generated_smiles = []

    with torch.no_grad():
        if config.model in ('GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF'):
            gc = (config.gvae if config.model == 'GVAE' else
                  config.gvae_nf if config.model == 'GVAE_NF' else
                  config.gvae_ar if config.model == 'GVAE_AR' else
                  config.gvae_ar_nf)
            # AR models pre-allocate KV buffers of shape (B, max_tf_len, d_model) per
            # layer — at B=10 000 that is ~60 GB for ZINC defaults.  Chunk into smaller
            # batches to keep peak memory bounded.  GVAE/GVAE_NF use a flat MLP decoder
            # so a larger chunk is fine; AR models need a much smaller one.
            decode_batch = 256 if config.model in ('GVAE_AR', 'GVAE_AR_NF') else 2048
            z_all = torch.randn(config.num_samples, gc.latent_dim)
            for i in tqdm(range(0, config.num_samples, decode_batch), desc=f"Sampling {config.model}"):
                z_batch = z_all[i:i + decode_batch].to(device)
                generated_smiles.extend(
                    model.sample_smiles(z_batch, atom_decoder, charge_decoder,
                                        valency_mask=gc.valency_mask)
                )

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
            t_smi = get_smiles_list('MOSES', 'train')
            v_smi = get_smiles_list('MOSES', 'test')
            reference_data = SmilesDataset(train_smiles=t_smi, validation_smiles=v_smi)
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
        metrics    = benchmarker.benchmark(generated_smiles)
        prop_stats = _compute_property_stats(valid_smiles)
        _log_metrics_table(metrics, logger, recon_rate=recon_rate, prop_stats=prop_stats)
        _save_metrics(metrics, config, recon_rate=recon_rate, prop_stats=prop_stats)
    except KeyboardInterrupt:
        logger.warning("Benchmarking interrupted by user.")
    except Exception as e:
        logger.warning(f"Benchmarking failed: {e}")


def _run_key(config: Config) -> str:
    """
    Build a human-readable unique key for this run, e.g.
      'ZINC/GVAE_NF+prop+no_valency'
    Used as the key in checkpoints/scores.json.
    """
    gc = (config.gvae if config.model == 'GVAE' else
          config.gvae_nf if config.model == 'GVAE_NF' else
          config.gvae_ar if config.model == 'GVAE_AR' else
          config.gvae_ar_nf if config.model == 'GVAE_AR_NF' else None)
    parts = [config.dataset, config.model]
    if gc is not None:
        parts.append('prop' if gc.prop_pred else 'no_prop')
        if gc.valency_mask:
            parts.append('valency')
    return '/'.join(parts[:2]) + ('+' + '+'.join(parts[2:]) if len(parts) > 2 else '')


def _log_metrics_table(metrics: dict, log,
                       recon_rate=None, prop_stats: dict | None = None):
    """Pretty-print benchmark metrics as an aligned table."""
    v    = metrics.get('validity', {})
    fcd  = metrics.get('fcd', {})
    mos  = metrics.get('moses', {})
    ps   = prop_stats or {}

    rows = [
        # (label, value, fmt)
        # ── Primary academic metrics ──────────────────────────────────
        ('Recon',       recon_rate,                                         '.4f'),
        ('Similar',     mos.get('snn_score'),                               '.4f'),
        ('Valid',       v.get('valid_fraction'),                            '.4f'),
        ('Unique',      v.get('unique_fraction_of_valids'),                 '.4f'),
        ('Novelty',     v.get('unique_and_novel_fraction_of_valids'),       '.4f'),
        ('FCD',         fcd.get('fcd'),                                     '.4f'),
        ('KL-Div.',     metrics.get('kl_score'),                            '.4f'),
        ('logP',        ps.get('logP'),                                     '.4f'),
        ('QED',         ps.get('QED'),                                      '.4f'),
        ('NP',          ps.get('NP'),                                       '.4f'),
        ('SA',          ps.get('SA'),                                       '.4f'),
        # ── Additional distribution / diversity metrics ───────────────
        ('IntDiv',      mos.get('IntDiv'),                                  '.4f'),
        ('IntDiv2',     mos.get('IntDiv2'),                                 '.4f'),
        ('MOSES filt.', mos.get('fraction_passing_moses_filters'),          '.4f'),
        ('Fragment sim.',mos.get('fragment_similarity'),                    '.4f'),
        ('Scaffold sim.',mos.get('scaffolds_similarity'),                   '.4f'),
    ]

    col_w = max(len(r[0]) for r in rows)
    sep   = '+' + '-' * (col_w + 2) + '+' + '-' * 12 + '+'
    log.info(sep)
    log.info(f"| {'Metric':<{col_w}} | {'Value':>10} |")
    log.info(sep)
    for label, val, fmt in rows:
        if val is None:
            formatted = '       N/A'
        else:
            formatted = format(val, fmt).rjust(10)
        log.info(f"| {label:<{col_w}} | {formatted} |")
    log.info(sep)


def _save_metrics(metrics: dict, config: Config,
                  recon_rate=None, prop_stats: dict | None = None,
                  scores_path: str = 'checkpoints/scores.json'):
    """Append / update results in a shared scores.json keyed by run identifier."""
    os.makedirs(os.path.dirname(scores_path), exist_ok=True)

    if os.path.exists(scores_path):
        with open(scores_path) as f:
            scores = json.load(f)
    else:
        scores = {}

    key = _run_key(config)
    v   = metrics.get('validity', {})
    fcd = metrics.get('fcd', {})
    mos = metrics.get('moses', {})
    ps  = prop_stats or {}

    scores[key] = {
        'timestamp':     datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
        # Primary metrics (table order)
        'recon':         recon_rate,
        'snn':           mos.get('snn_score'),
        'validity':      v.get('valid_fraction'),
        'uniqueness':    v.get('unique_fraction_of_valids'),
        'novelty':       v.get('unique_and_novel_fraction_of_valids'),
        'fcd':           fcd.get('fcd'),
        'kl_score':      metrics.get('kl_score'),
        'logP':          ps.get('logP'),
        'QED':           ps.get('QED'),
        'NP':            ps.get('NP'),
        'SA':            ps.get('SA'),
        # Distribution / diversity extras
        'intdiv':        mos.get('IntDiv'),
        'intdiv2':       mos.get('IntDiv2'),
        'moses_filters': mos.get('fraction_passing_moses_filters'),
        'fragment_sim':  mos.get('fragment_similarity'),
        'scaffold_sim':  mos.get('scaffolds_similarity'),
    }

    with open(scores_path, 'w') as f:
        json.dump(scores, f, indent=2)
    logger.info(f"Scores saved → {scores_path}  (key: '{key}')")


def _compute_property_stats(valid_smiles: list[str]) -> dict:
    """
    Compute mean molecular properties over the valid generated molecules.

    Returns a dict with keys: logP, QED, SA, NP.
    All are means over the valid generated set (comparable to reference
    dataset means for a quick distribution sanity-check).
    """
    from rdkit.Chem import Descriptors
    from rdkit.Chem import QED as rdkit_QED
    from rdkit.Contrib.SA_Score import sascorer
    from rdkit.Contrib.NP_Score import npscorer

    np_model = npscorer.readNPModel()

    logPs, qeds, sas, nps = [], [], [], []
    for smi in valid_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        logPs.append(Descriptors.MolLogP(mol))
        qeds.append(rdkit_QED.qed(mol))
        sas.append(sascorer.calculateScore(mol))
        nps.append(npscorer.scoreMol(mol, np_model))

    def _mean(lst):
        return float(sum(lst) / len(lst)) if lst else None

    return {'logP': _mean(logPs), 'QED': _mean(qeds), 'SA': _mean(sas), 'NP': _mean(nps)}


def _compute_recon_rate(model, val_loader, config: Config, device, metadata,
                        n_recon: int = 1000):
    """
    Reconstruction accuracy on the validation set.

    For each model: encode validation molecules to μ (no noise), decode back,
    then compare non-isomeric canonical SMILES to the original.

    GVAE / GVAE_NF / GVAE_AR / GVAE_AR_NF
        Ground-truth SMILES come from batch.smiles (stored in each PyG Data
        object at dataset-construction time).  This is the only reliable
        approach: an external list indexed by n_total would break for MOSES
        (max_atoms filtering drops molecules → misalignment) and is fragile
        for ZINC (PyG cache vs. smiles_*.txt may differ in ordering).
        Comparison uses isomericSmiles=False because the graph VAE encodes
        atom type and bond order but not chirality or double-bond geometry;
        the decoder can never reproduce @/@@ or /\\ markers.
    FRATTVAE
        References from metadata['val_smiles'] (stored during preprocessing).
        Returns None when the metadata key is absent (delete cache to regen).
    """
    model.eval()

    if config.model in ('GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF'):
        gc           = {'GVAE':   config.gvae,
                        'GVAE_NF': config.gvae_nf,
                        'GVAE_AR': config.gvae_ar,
                        'GVAE_AR_NF': config.gvae_ar_nf}[config.model]
        atom_decoder = MOSES_ATOM_DECODER if config.dataset == 'MOSES' else ZINC_ATOM_DECODER
        charge_dec   = None if config.dataset == 'MOSES' else ZINC_CHARGE_DECODER

        n_correct = 0
        n_total   = 0
        with torch.no_grad():
            for batch_raw in val_loader:
                # AR loader yields (PyGBatch, input_tok, target_tok, types, lens);
                # flat loaders yield the PyGBatch directly.
                batch = batch_raw[0] if isinstance(batch_raw, (tuple, list)) else batch_raw
                x_in, edge_index, edge_attr, batch_idx, _, _ = gvae_prepare_batch(
                    batch, device, gc.max_atoms)
                mu, _logvar = model.encode(x_in, edge_index, edge_attr, batch_idx)
                # T=0.0 → greedy argmax for AR models (see _sample_batch).
                # Flat decoders ignore temperature; argmax is always used.
                recon_T = 0.0 if config.model in ('GVAE_AR', 'GVAE_AR_NF') else 1.0
                decoded = model.sample_smiles(mu, atom_decoder, charge_dec,
                                              valency_mask=gc.valency_mask,
                                              temperature=recon_T)

                # Ground-truth SMILES travel with the batch (batch.smiles is a list[str]
                # set at dataset construction time, same molecules the encoder just saw).
                # This avoids any index-alignment assumption between the DataLoader and
                # an external SMILES list (which breaks for MOSES where max_atoms
                # filtering drops molecules, and is fragile for ZINC if the smiles
                # cache and PyG cache were built from different sources).
                ref_smiles_batch = batch.smiles  # list[str], len == batch.num_graphs

                for pred, ref in zip(decoded, ref_smiles_batch):
                    ref_mol  = Chem.MolFromSmiles(ref)  if ref  else None
                    pred_mol = Chem.MolFromSmiles(pred) if pred else None
                    if ref_mol and pred_mol:
                        # Strip isomeric / stereo markers: the graph VAE encodes atom
                        # type and bond order but not chirality or double-bond geometry,
                        # so the decoder can never reproduce @/@@ or /\ markers.
                        # Comparing non-isomeric SMILES gives a fair structural match.
                        ref_smi  = Chem.MolToSmiles(ref_mol,  isomericSmiles=False)
                        pred_smi = Chem.MolToSmiles(pred_mol, isomericSmiles=False)
                        if ref_smi == pred_smi:
                            n_correct += 1
                n_total += batch.num_graphs
                if n_total >= n_recon:
                    break
        return n_correct / max(1, n_total)

    elif config.model == 'FRATTVAE':
        ref_smiles = metadata.get('val_smiles')
        if not ref_smiles:
            logger.warning(
                "metadata['val_smiles'] not available — delete the FRATTVAE cache "
                "and re-run to enable reconstruction evaluation."
            )
            return None

        from models.frattvae.utils.mask import create_mask
        frag_ecfps = metadata['frag_ecfps'].to(device)
        ndummys    = metadata['ndummys'].to(device)
        model.set_labels(metadata['uni_fragments'])

        n_correct = 0
        n_total   = 0
        with torch.no_grad():
            for frag_indices, positions, _ in val_loader:
                B, L    = frag_indices.shape
                fi      = frag_indices.to(device)
                pos     = positions.to(device=device, dtype=torch.float32)
                features = frag_ecfps[fi.flatten()].reshape(B, L, -1)

                # Build encoder padding mask (identical to training)
                idx_with_root = torch.cat([
                    torch.full((B, 1), -1, device=device), fi
                ], dim=1)
                _, _, src_pad_mask, _ = create_mask(idx_with_root, idx_with_root, pad_idx=0)

                # encode() uses μ in eval mode (reparameterize returns μ when not training)
                z, _mu, _lv = model.encode(features, pos, src_pad_mask=src_pad_mask)
                decoded = model.sequential_decode(
                    z, frag_ecfps, ndummys,
                    max_nfrags=config.frattvae.max_nfrags,
                    asSmiles=True,
                )
                refs = ref_smiles[n_total: n_total + B]

                for pred, ref in zip(decoded, refs):
                    ref_mol  = Chem.MolFromSmiles(ref)  if ref  else None
                    pred_mol = Chem.MolFromSmiles(pred) if pred else None
                    if ref_mol and pred_mol:
                        if Chem.MolToSmiles(ref_mol) == Chem.MolToSmiles(pred_mol):
                            n_correct += 1
                n_total += B
                if n_total >= n_recon:
                    break
        return n_correct / max(1, n_total)

    return None


def _build_model(config: Config, metadata, device):
    """Instantiate the right model architecture from config + metadata."""
    if config.model == 'GVAE':
        model = GraphVAE(
            num_node_features=metadata['num_nodes'],
            num_edge_features=metadata['num_edges'],
            latent_dim=config.gvae.latent_dim,
            max_atoms=config.gvae.max_atoms,
            prop_pred=config.gvae.prop_pred,
        )
    elif config.model == 'GVAE_NF':
        model = GraphVAENF(
            num_node_features=metadata['num_nodes'],
            num_edge_features=metadata['num_edges'],
            latent_dim=config.gvae_nf.latent_dim,
            max_atoms=config.gvae_nf.max_atoms,
            num_flows=config.gvae_nf.num_flows,
            flow_hidden_dim=config.gvae_nf.flow_hidden_dim,
            prop_pred=config.gvae_nf.prop_pred,
        )
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
        )
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
    Expected layout: checkpoints/{DATASET}/{MODEL}/{VARIANT}/best.pth
    """
    found = []
    if not os.path.isdir(root):
        return found
    for dataset in sorted(os.listdir(root)):
        d_path = os.path.join(root, dataset)
        if not os.path.isdir(d_path):
            continue
        for model_name in sorted(os.listdir(d_path)):
            m_path = os.path.join(d_path, model_name)
            if not os.path.isdir(m_path):
                continue
            for variant in sorted(os.listdir(m_path)):
                ckpt = os.path.join(m_path, variant, 'best.pth')
                if os.path.isfile(ckpt):
                    found.append((dataset, model_name, ckpt))
    return found


def run_validation(dataset: str, model_name: str, checkpoint: str,
                   num_samples: int = 10000, num_workers: int = 4):
    """
    Load a checkpoint and run the full evaluation suite.

    Args:
        dataset:     'ZINC' or 'MOSES'
        model_name:  'GVAE', 'GVAE_NF', or 'FRATTVAE'
        checkpoint:  Path to best.pth  (…/{DATASET}/{MODEL}/{VARIANT}/best.pth)
        num_samples: Number of molecules to sample for benchmarks
        num_workers: DataLoader worker count
        valency_mask: Fallback if variant cannot be inferred from path
    """
    logger.info(f"=== Evaluating {model_name} on {dataset} from {checkpoint} ===")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Infer flags from the variant directory name embedded in the checkpoint path.
    # Layout: …/{DATASET}/{MODEL}/{VARIANT}/best.pth
    variant = os.path.basename(os.path.dirname(checkpoint))   # e.g. 'prop+no_valency'
    variant_parts = set(variant.split('+'))
    prop_pred = 'prop'    in variant_parts
    valency   = 'valency' in variant_parts

    config = Config(model=model_name, dataset=dataset, num_samples=num_samples, num_workers=num_workers)
    if dataset == 'MOSES':
        config.gvae.max_atoms = 30
        config.gvae_nf.max_atoms = 30
        config.gvae_ar.max_atoms = 30
        config.gvae_ar_nf.max_atoms = 30
    config.gvae.prop_pred     = prop_pred
    config.gvae_nf.prop_pred  = prop_pred
    config.gvae_ar.prop_pred  = prop_pred
    config.gvae_ar_nf.prop_pred = prop_pred
    config.gvae.valency_mask    = valency
    config.gvae_nf.valency_mask = valency
    config.gvae_ar.valency_mask = valency
    config.gvae_ar_nf.valency_mask = valency

    _, val_loader, metadata = get_dataloaders(config, logger)

    model = _build_model(config, metadata, device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)
    logger.info(f"Loaded weights from {checkpoint}")

    evaluate_model(model, config, device, metadata, val_loader)


def _parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained molecular VAE checkpoints")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--checkpoint', type=str, default=None,
                       help='Path to a specific best.pth to evaluate')
    group.add_argument('--all', action='store_true', default=False,
                       help='Evaluate all checkpoints found under checkpoints/')
    parser.add_argument('--model',   type=str, choices=['GVAE', 'GVAE_NF', 'GVAE_AR', 'GVAE_AR_NF', 'FRATTVAE'], default=None,
                        help='Filter to a specific model (used with --all)')
    parser.add_argument('--dataset', type=str, choices=['ZINC', 'MOSES'],    default=None,
                        help='Filter to a specific dataset (used with --all or with --checkpoint)')
    parser.add_argument('--num_samples', type=int, default=10000)
    parser.add_argument('--num_workers', type=int, default=4)

    return parser.parse_args()


def main():
    args = _parse_args()

    common = dict(
        num_samples=args.num_samples,
        num_workers=args.num_workers,
    )

    if args.checkpoint:
        # Single explicit checkpoint; infer dataset/model from path if not supplied.
        # Expected layout: …/{DATASET}/{MODEL}/{VARIANT}/best.pth
        parts = args.checkpoint.replace('\\', '/').split('/')
        try:
            model_name = args.model   or parts[-3]
            dataset    = args.dataset or parts[-4]
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


if __name__ == '__main__':
    main()
