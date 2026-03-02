import numpy as np
from joblib import Parallel, delayed

import torch
import torch.nn as nn

from .utils.tree import FragmentTree, get_pad_features
from .utils.mask import generate_square_subsequent_mask
from .utils.construct import constructMol, constructMolwithTimeout


class TreePositionalEncoding(nn.Module):
    """
    Reference: https://github.com/microsoft/icecaps/blob/master/icecaps/estimators/abstract_transformer_estimator.py
    """
    def __init__(self, d_model: int, d_pos: int, depth: int, width: int) -> None:
        super().__init__()
        self.d_params = d_pos // (depth * width)
        self.d_model = d_model
        self.depth = depth
        self.width = width
        self.params = nn.Parameter(torch.randn(self.d_params), requires_grad=True)
        self.fc = nn.Linear(self.d_params * self.depth * self.width, d_model)

    def forward(self, positions: torch.Tensor):
        """positions: shape= (Batch_size, Length, depth * width)"""
        tree_weights = self._compute_weights(positions.device)
        treeified = positions.unsqueeze(-1) * tree_weights  # (B, L, depth*width, d_param)
        treeified = treeified.flatten(start_dim=2)           # (B, L, depth*width*d_param = d_pos)
        if treeified.shape[-1] != self.d_model:
            treeified = self.fc(treeified)
        return treeified

    def _compute_weights(self, device):
        params = torch.tanh(self.params)
        tiled_tree_params = params.view(1, 1, -1).repeat(self.depth, self.width, 1)
        tiled_depths = torch.arange(self.depth, dtype=torch.float32, device=device).view(-1, 1, 1).repeat(1, self.width, self.d_params)
        tree_norm = torch.sqrt((1 - params.square()) * self.d_model / 2)
        weights = (tiled_tree_params ** tiled_depths) * tree_norm
        return weights.view(self.depth * self.width, self.d_params)


class FRATTVAE(nn.Module):
    """
    Fragment Tree Transformer VAE.

    Encodes molecules represented as BRICS fragment trees into a continuous latent space
    using a Transformer encoder, and decodes back fragment-by-fragment with a Transformer decoder.

    Args:
        num_tokens:  Total number of unique fragments (vocabulary size including padding at index 0).
        depth:       Maximum tree depth (max_depth hyperparameter).
        width:       Maximum tree degree (max_degree hyperparameter).
        feat_dim:    Dimension of the fragment fingerprint (ECFP bit length).
        latent_dim:  Dimension of the VAE latent space.
        d_model:     Transformer model dimension.
        d_ff:        Transformer feed-forward dimension.
        num_layers:  Number of Transformer encoder/decoder layers.
        nhead:       Number of attention heads.
        activation:  Activation function ('relu' or 'gelu').
        dropout:     Dropout probability.
        n_jobs:      Parallelism for molecule construction during sampling.
    """
    def __init__(self, num_tokens: int, depth: int, width: int,
                 feat_dim: int = 2048, latent_dim: int = 256,
                 d_model: int = 512, d_ff: int = 2048, num_layers: int = 6, nhead: int = 8,
                 activation: str = 'gelu', dropout: float = 0.1, n_jobs: int = 1) -> None:
        super().__init__()
        assert activation in ['relu', 'gelu']
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.dropout = dropout
        self.depth = depth
        self.width = width
        self.n_jobs = n_jobs

        self.embed = nn.Embedding(num_embeddings=1, embedding_dim=d_model)  # <root>
        self.fc_ecfp = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Linear(feat_dim // 2, d_model)
        )
        self.PE = TreePositionalEncoding(d_model=d_model, d_pos=max(d_model, depth * width), depth=depth, width=width)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=self.dropout, activation=activation, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc_vae = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.Linear(latent_dim, 2 * latent_dim)
        )

        self.fc_memory = nn.Linear(latent_dim, d_model)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_ff,
            dropout=self.dropout, activation=activation, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.fc_dec = nn.Linear(d_model, num_tokens)

        self.labels = None  # fragment SMILES strings (set via set_labels before sampling)

    def forward(self, features: torch.Tensor, positions: torch.Tensor,
                src_mask: torch.Tensor = None, src_pad_mask: torch.Tensor = None,
                tgt_mask: torch.Tensor = None, tgt_pad_mask: torch.Tensor = None,
                frag_ecfps: torch.Tensor = None, ndummys: torch.Tensor = None,
                max_nfrags: int = 30, free_n: bool = False, sequential: bool = None):
        """
        Encode and decode.  Uses parallel teacher-forcing during training,
        sequential autoregressive decoding during inference.

        Returns: (z, mu, ln_var, output)
            z:      Latent sample  (B, latent_dim)
            mu:     Posterior mean  (B, latent_dim)
            ln_var: Posterior log-variance  (B, latent_dim)
            output: Token logits  (B, L+1, num_tokens)  [parallel] or list of FragmentTree [sequential]
        """
        sequential = not self.training if sequential is None else sequential

        z, mu, ln_var = self.encode(features, positions, src_mask, src_pad_mask)
        if sequential:
            output = self.sequential_decode(z, frag_ecfps, ndummys, max_nfrags=max_nfrags, free_n=free_n)
        else:
            output = self.decode(z, features, positions, tgt_mask, tgt_pad_mask)

        return z, mu, ln_var, output

    def encode(self, features: torch.Tensor, positions: torch.Tensor,
               src_mask: torch.Tensor = None, src_pad_mask: torch.Tensor = None):
        """
        features:  (B, L, feat_dim)
        positions: (B, L, depth * width)
        Returns:   z (B, latent_dim), mu (B, latent_dim), ln_var (B, latent_dim)
        """
        src = self.fc_ecfp(features) + self.PE(positions)           # (B, L, d_model)
        root_embed = self.embed(src.new_zeros(src.shape[0], 1).long())
        src = torch.cat([root_embed, src], dim=1)                    # (B, L+1, d_model)

        out = self.encoder(src, mask=src_mask, src_key_padding_mask=src_pad_mask)
        out = out[:, 0, :].squeeze(1)                                # take super-root token

        mu, ln_var = self.fc_vae(out).chunk(2, dim=-1)
        # Clamp ln_var to prevent exp() overflow (both float32 and bfloat16 overflow
        # at ln_var > ~88). exp(10) ≈ 22000 is already a very large variance.
        ln_var = ln_var.clamp(-10, 10)
        z = self.reparameterization_trick(mu, ln_var)

        return z, mu, ln_var

    def decode(self, z: torch.Tensor, features: torch.Tensor, positions: torch.Tensor,
               tgt_mask: torch.Tensor = None, tgt_pad_mask: torch.Tensor = None):
        """
        Parallel teacher-forcing decode (used during training).
        Returns logits: (B, L+1, num_tokens)
        """
        memory = self.fc_memory(z).unsqueeze(1)                      # (B, 1, d_model)

        tgt = self.fc_ecfp(features) + self.PE(positions)            # (B, L, d_model)
        root_embed = self.embed(tgt.new_zeros(tgt.shape[0], 1).long())
        tgt = torch.cat([root_embed, tgt], dim=1)                    # (B, L+1, d_model)

        out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
        out = self.fc_dec(out)                                        # (B, L+1, num_tokens)
        return out

    def sequential_decode(self, z: torch.Tensor, frag_ecfps: torch.Tensor, ndummys: torch.Tensor,
                          max_nfrags: int = 30, free_n: bool = False, asSmiles: bool = False) -> list:
        """
        Autoregressive decoding from a latent vector z.

        z:          (B, latent_dim)
        frag_ecfps: (num_tokens, feat_dim)  – ECFPs for each vocabulary fragment
        ndummys:    (num_tokens,)            – degree (# dummy atoms) for each fragment
        asSmiles:   if True, returns SMILES strings; else returns FragmentTree objects.
        """
        batch_size = z.shape[0]
        device = z.device

        memory = self.fc_memory(z).unsqueeze(1)

        # Root prediction
        root_embed = self.embed(torch.zeros(batch_size, 1, device=device).long())
        tgt_pad_mask = torch.all(root_embed == 0, dim=-1).to(device)
        out = self.decoder(root_embed, memory, tgt_key_padding_mask=tgt_pad_mask)
        out = self.fc_dec(out)
        root_idxs = out.argmax(dim=-1).flatten()

        continues = []
        target_ids = [0] * batch_size
        target_ids_list = [[0] for _ in range(batch_size)]
        tree_list = [FragmentTree() for _ in range(batch_size)]
        for i, idx in enumerate(root_idxs):
            parent_id = tree_list[i].add_node(parent_id=None, feature=frag_ecfps[idx], fid=idx.item(), bondtype=0)
            assert parent_id == 0
            tree_list[i].set_positional_encoding(parent_id, d_pos=self.depth * self.width)
            continues.append(ndummys[idx].item() > 0)

        nfrags = 1
        while (nfrags < max_nfrags) & (sum(continues) > 0):
            tgt_mask = generate_square_subsequent_mask(length=nfrags + 1).to(device)
            tgt_pad_mask = torch.hstack([tgt_pad_mask, tgt_pad_mask.new_full(size=(batch_size, 1), fill_value=False)])
            features = get_pad_features(tree_list, key='x', max_nodes_num=nfrags).to(device)
            positions = get_pad_features(tree_list, key='pos', max_nodes_num=nfrags).to(device)

            tgt = self.fc_ecfp(features) + self.PE(positions)
            tgt = torch.cat([root_embed, tgt], dim=1)

            out = self.decoder(tgt, memory, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_pad_mask)
            out = self.fc_dec(out)
            new_idxs = out[:, -1, :].argmax(dim=-1).flatten()

            for i, idx in enumerate(new_idxs):
                if continues[i]:
                    if ndummys[idx] == 0:
                        idx = torch.tensor(0)
                    if idx != 0:
                        parent_id = target_ids[i]
                        add_node_id = tree_list[i].add_node(parent_id=parent_id, feature=frag_ecfps[idx], fid=idx.item(), bondtype=1)
                        parent_fid = tree_list[i].dgl_graph.ndata['fid'][parent_id].item()
                        num_sibling = ndummys[parent_fid].item() - 1 if parent_id > 0 else ndummys[parent_fid].item()
                        if free_n:
                            tree_list[i].set_positional_encoding(add_node_id, num_sibling=num_sibling, d_pos=self.depth * self.width)
                        else:
                            tree_list[i].set_positional_encoding(add_node_id, num_sibling=self.width, d_pos=self.depth * self.width)
                        level = tree_list[i].dgl_graph.ndata['level'][add_node_id].item()

                        if (len(tree_list[i].dgl_graph.predecessors(parent_id)) >= num_sibling):
                            target_ids_list[i].pop(-1)
                        if (ndummys[idx] > 1) & (self.depth > level):
                            target_ids_list[i].append(add_node_id)

                    continues[i] = bool(target_ids_list[i]) if (idx != 0) else False
                    target_ids[i] = target_ids_list[i][-1] if continues[i] else 0
            nfrags += 1

        if asSmiles:
            if self.labels is not None:
                outputs = Parallel(n_jobs=self.n_jobs)(
                    delayed(constructMol)(
                        self.labels[tree.dgl_graph.ndata['fid'].squeeze(-1).tolist()].tolist(),
                        tree.adjacency_matrix().tolist()
                    )
                    for tree in tree_list
                )
            else:
                raise ValueError('Set fragment SMILES labels first via model.set_labels(labels).')
        else:
            outputs = tree_list

        return outputs

    def reparameterization_trick(self, mu, ln_var):
        eps = torch.randn_like(mu)
        z = mu + torch.exp(ln_var / 2) * eps if self.training else mu
        return z

    def set_labels(self, labels):
        """Set the fragment SMILES vocabulary so sequential_decode can output SMILES strings."""
        if isinstance(labels, np.ndarray):
            self.labels = labels
        else:
            self.labels = np.array(labels)
