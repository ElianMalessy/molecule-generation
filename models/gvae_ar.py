"""
GraphVAE with a latent-conditioned autoregressive Transformer decoder (GVAE_AR).

Decoder architecture
--------------------
Molecules are serialized into a single 1-D sequence via BFS linearization:

    S = [v₁,  v₂, e₂₁,  v₃, e₃₁, e₃₂,  v₄, …,  vₙ, eₙ₁…eₙₙ₋₁,  <EOS>]

where vᵢ is the atom type (BFS step i) and eᵢⱼ is the bond type from atom i
to the previously-generated atom j.

For max_atoms=38 the maximum sequence length is 38 + 703 + 1 = 742 tokens.

A GPT-style (decoder-only) causal Transformer is conditioned on the latent z by
prepending z as a "prefix token" (projected to d_model).  The model sees:

    Transformer input:  [z_proj,  v₁, v₂, e₂₁, …, eₙₙ₋₁]     length L
    Transformer output: [h₀, h₁, h₂, h₃,  …, hₗ₋₁]            length L
    Targets:            [v₁,  v₂, e₂₁,  …, eₙₙ₋₁,  <EOS>]     length L

Two separate linear heads route each hidden state to either the atom vocabulary
or the bond vocabulary depending on the deterministic token type at that position.
"""
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem

from models.encoder import GINEConvEncoder
from models.flows import InverseAutoregressiveFlow
from models.gvae import PropertyHead, decode_to_smiles, _MAX_VALENCE, _BOND_ORDER, _get_rdkit_bond


# ---------------------------------------------------------------------------
# BFS utilities
# ---------------------------------------------------------------------------

def _bfs_order(adj_bin, n: int) -> list:
    """Return atom indices in BFS order starting from the highest-degree atom.
    Ties are broken by atom index (deterministic).  Disconnected components are
    visited after the main component (edge case in valid molecules)."""
    if n == 0:
        return []
    degrees = adj_bin.sum(1)
    start = int(degrees.argmax())

    visited = [False] * n
    order = []
    queue = collections.deque([start])
    visited[start] = True

    while queue:
        node = queue.popleft()
        order.append(node)
        for nb in adj_bin[node].nonzero()[0].tolist():
            if not visited[nb]:
                visited[nb] = True
                queue.append(nb)

    # Handle disconnected atoms (rare, but possible in corrupt data)
    for i in range(n):
        if not visited[i]:
            visited[i] = True
            q2 = collections.deque([i])
            while q2:
                node = q2.popleft()
                order.append(node)
                for nb in adj_bin[node].nonzero()[0].tolist():
                    if not visited[nb]:
                        visited[nb] = True
                        q2.append(nb)
    return order


def _mol_to_sequence(node_row, adj_row, n: int, eos_id: int):
    """Convert one molecule (numpy arrays) to a BFS-linearised token sequence.

    Returns
    -------
    tokens : list[int]  – [v₁, v₂, e₂₁, v₃, …, eₙₙ₋₁, EOS]
    types  : list[int]  – 1=node, 2=edge  (same length as tokens)
    """
    adj_bin = (adj_row[:n, :n] > 0)
    order = _bfs_order(adj_bin, n)

    tokens, types = [], []
    for i, orig in enumerate(order):
        tokens.append(int(node_row[orig]))
        types.append(1)
        for j in range(i):
            prev = order[j]
            tokens.append(int(adj_row[orig, prev]))
            types.append(2)

    tokens.append(eos_id)
    types.append(1)  # EOS is a "node-type" token
    return tokens, types


def build_ar_batch(target_nodes, target_edges, eos_id: int, abs_max_len: int):
    """Serialize a batch of dense graph tensors into padded AR sequences.

    Parameters
    ----------
    target_nodes : (B, N) long  – 0=padding, 1..K=atom class
    target_edges : (B, N, N) long – 0=no-bond, 1..M=bond type
    eos_id       : int – EOS token index  (= num_node_features)
    abs_max_len  : int – maximum allowed sequence length (= max_atoms*(max_atoms+1)//2+1)

    Returns
    -------
    input_tokens  : (B, max_L-1) long – seq[:-1] per molecule, padded with 1
    target_tokens : (B, max_L)   long – seq      per molecule, padded with -1
    target_types  : (B, max_L)   long – 1/2 at valid positions, 0 at padding
    seq_lens      : (B,)         long – actual sequence length L_b per molecule
    """
    B = target_nodes.size(0)
    device = target_nodes.device

    nodes_cpu = target_nodes.cpu().numpy()
    edges_cpu = target_edges.cpu().numpy()

    all_tokens, all_types = [], []
    for b in range(B):
        n = int((target_nodes[b] > 0).sum().item())
        tok, typ = _mol_to_sequence(nodes_cpu[b], edges_cpu[b], n, eos_id)
        all_tokens.append(tok)
        all_types.append(typ)

    max_L = min(max(len(t) for t in all_tokens), abs_max_len)

    # Build on CPU, then move in one H2D copy
    input_tokens  = torch.ones (B, max_L - 1, dtype=torch.long)
    target_tokens = torch.full ((B, max_L),   -1, dtype=torch.long)
    target_types  = torch.zeros(B, max_L,         dtype=torch.long)
    seq_lens      = torch.zeros(B,                dtype=torch.long)

    for b, (tok, typ) in enumerate(zip(all_tokens, all_types)):
        # Guard: if truncated, ensure the last token is EOS so the model
        # always learns when to stop
        tok_b = tok[:max_L]
        typ_b = typ[:max_L]
        if tok_b[-1] != eos_id:          # always ensure termination token present
            tok_b = tok_b[:max_L - 1] + [eos_id]
            typ_b = typ_b[:max_L - 1] + [1]
        L = len(tok_b)
        seq_lens[b] = L
        if L > 1:
            input_tokens [b, :L - 1] = torch.tensor(tok_b[:L - 1], dtype=torch.long)
        target_tokens[b, :L] = torch.tensor(tok_b, dtype=torch.long)
        target_types [b, :L] = torch.tensor(typ_b, dtype=torch.long)

    return (input_tokens.to(device), target_tokens.to(device),
            target_types.to(device), seq_lens.to(device))


# ---------------------------------------------------------------------------
# Causal Transformer with first-class KV-cache support
# ---------------------------------------------------------------------------

class _CausalLayer(nn.Module):
    """Pre-LN causal transformer layer.

    Explicit Q/K/V projections (vs. fused in_proj) give a clean public API
    for KV-cached inference — no private PyTorch internals required.
    Activation is GELU (standard for GPT-style decoders; original used ReLU).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.norm1    = nn.LayerNorm(d_model)
        self.norm2    = nn.LayerNorm(d_model)
        self.q_proj   = nn.Linear(d_model, d_model)
        self.k_proj   = nn.Linear(d_model, d_model)
        self.v_proj   = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.attn_drop = dropout

    def _to_heads(self, t: torch.Tensor) -> torch.Tensor:
        """(B, L, d) → (B, n_heads, L, head_dim).  reshape handles non-contiguous slices."""
        B, L, _ = t.shape
        return t.reshape(B, L, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Full-sequence causal forward (teacher-forced training).

        attn_mask: optional boolean mask (B, 1, L, L) — True = attend, False = mask.
          When None (the common path), is_causal=True is passed to SDPA so
          FlashAttention can be selected automatically by PyTorch.
          Note: PyTorch SDPA disallows is_causal=True alongside attn_mask, so
          these two paths are mutually exclusive.
        """
        h = self.norm1(x)
        q, k, v = self._to_heads(self.q_proj(h)), self._to_heads(self.k_proj(h)), self._to_heads(self.v_proj(h))
        drop = self.attn_drop if self.training else 0.0
        if attn_mask is None:
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=drop)
        else:
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=drop)
        x    = x + self.out_proj(a.transpose(1, 2).contiguous().view_as(x))
        x    = x + self.ff(self.norm2(x))
        return x

    def step(self, x: torch.Tensor,
             K_buf: torch.Tensor, V_buf: torch.Tensor, t: int) -> torch.Tensor:
        """Single-token causal step with in-place KV-cache writes.

        Parameters
        ----------
        x     : (B, 1, d_model) – embedding of the current token
        K_buf : (B, max_tf_len, d_model) – pre-allocated; mutated at column t
        V_buf : (B, max_tf_len, d_model) – pre-allocated; mutated at column t
        t     : absolute position of this token in the sequence

        No causal mask is needed: the query is always the newest position and
        may legally attend to all t+1 entries in the cache.
        """
        h = self.norm1(x)
        K_buf[:, t : t + 1, :] = self.k_proj(h)          # write in-place — no allocation
        V_buf[:, t : t + 1, :] = self.v_proj(h)
        q = self._to_heads(self.q_proj(h))                 # (B, nh, 1, hd)
        k = self._to_heads(K_buf[:, : t + 1, :])          # (B, nh, t+1, hd)
        v = self._to_heads(V_buf[:, : t + 1, :])
        a = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        x = x + self.out_proj(a.transpose(1, 2).contiguous().view_as(x))
        x = x + self.ff(self.norm2(x))
        return x                                           # (B, 1, d_model)


class _CausalTransformer(nn.Module):
    """Stack of _CausalLayer with a training forward and a KV-cached inference step."""

    def __init__(self, n_layers: int, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList(
            [_CausalLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, attn_mask)
        return x

    def step(self, x: torch.Tensor,
             K_bufs: list[torch.Tensor], V_bufs: list[torch.Tensor], t: int) -> torch.Tensor:
        """Run one token through every layer using the shared KV buffers.
        Returns (B, d_model) — the hidden state of the new token.
        """
        for i, layer in enumerate(self.layers):
            x = layer.step(x, K_bufs[i], V_bufs[i], t)
        return x[:, 0, :]

# ---------------------------------------------------------------------------
# Autoregressive Transformer decoder
# ---------------------------------------------------------------------------

class ARDecoder(nn.Module):
    """Latent-conditioned causal Transformer for sequential graph generation.

    The latent vector z is projected to d_model and prepended as a "prefix
    token" at position 0.  All subsequent positions attend to it through the
    standard lower-triangular causal mask — achieving full z-conditioning
    without cross-attention.
    """

    def __init__(self, latent_dim: int, num_node_features: int, num_edge_features: int,
                 d_model: int = 256, n_heads: int = 8, n_layers: int = 4,
                 d_ff: int = 512, dropout: float = 0.1, max_atoms: int = 38):
        super().__init__()

        # +1 for EOS token appended to node vocabulary
        self.node_vocab      = num_node_features + 1
        self.edge_vocab      = num_edge_features
        self.eos_id          = num_node_features
        self.num_node_feat   = num_node_features
        self.num_edge_feat   = num_edge_features
        self.d_model         = d_model
        self.max_atoms       = max_atoms
        # Maximum sequence length = L_max = N*(N+1)/2 + 1  (atoms + edges + EOS)
        self.max_seq_len     = max_atoms * (max_atoms + 1) // 2 + 1
        # Training uses positions 0..max_seq_len-1 (z prefix + seq[:-1]).
        # Sampling needs one extra slot: z at 0, then up to max_seq_len graph
        # tokens at positions 1..max_seq_len.  Allocate max_seq_len + 1 (B1).
        self.max_tf_len      = self.max_seq_len + 1

        # Latent prefix projection
        self.z_proj = nn.Linear(latent_dim, d_model)

        # Separate embedding tables: node vocab and edge vocab
        self.node_emb = nn.Embedding(self.node_vocab, d_model)
        self.edge_emb = nn.Embedding(self.edge_vocab, d_model)

        # 3-way token-type embedding: 0=z-prefix, 1=node, 2=edge
        # This disambiguates overlapping integer values across the two vocabularies.
        self.type_emb = nn.Embedding(3, d_model)

        # 1-D absolute positional encoding (learnable, better than sinusoidal
        # for this highly structured sequence where position encodes atom index)
        self.pos_emb = nn.Embedding(self.max_tf_len, d_model)

        # Dual output heads
        self.node_head = nn.Linear(d_model, self.node_vocab)
        self.edge_head = nn.Linear(d_model, self.edge_vocab)

        # Replace nn.TransformerEncoderLayer / nn.TransformerEncoder with _CausalTransformer.
        # dynamic=True handles the varying sequence lengths seen both at training time
        # (max_L differs per batch) and the growing L during the sampling loop (P1).
        self.transformer = _CausalTransformer(n_layers, d_model, n_heads, d_ff, dropout)
        if hasattr(torch, 'compile'):
            self.transformer = torch.compile(self.transformer, dynamic=True, fullgraph=False)
        # Note: self.transformer.step(...) is accessed via OptimizedModule.__getattr__ and
        # runs uncompiled, which is correct and safe — no device/state issues.

    # ------------------------------------------------------------------
    # Internal: build transformer input embeddings
    # ------------------------------------------------------------------

    def _embed(self, z: torch.Tensor, input_tokens: torch.Tensor,
               input_types: torch.Tensor) -> torch.Tensor:
        """Build full embedding sequence [z_prefix | graph_tokens].

        node_emb / edge_emb are kept in the computation graph via weighted sum.
        The previous scatter-write approach (tok_emb[mask] = emb(...)) severed
        their gradient path because __setitem__ on a requires_grad=False tensor
        does not create a grad_fn.
        """
        B    = z.size(0)
        device = z.device

        # Clamp to valid ranges; per-position weights zero out irrelevant lookups
        # so out-of-vocabulary indices for the "wrong" table are harmless.
        node_idx = input_tokens.clamp(0, self.node_vocab - 1)
        edge_idx = input_tokens.clamp(0, self.edge_vocab - 1)
        n_w = (input_types == 1).to(dtype=z.dtype).unsqueeze(-1)  # (B, L, 1)
        e_w = (input_types == 2).to(dtype=z.dtype).unsqueeze(-1)

        tok_emb = (n_w * self.node_emb(node_idx)
                   + e_w * self.edge_emb(edge_idx)
                   + self.type_emb(input_types))                        # (B, L-1, d)

        z_vec = self.z_proj(z)                                         # (B, d_model)
        z_emb = z_vec.unsqueeze(1) + self.type_emb.weight[0]

        # Inject z as a residual at every token position so the decoder
        # cannot ignore z (prevents posterior collapse / latent variable dropout).
        tok_emb = tok_emb + z_vec.unsqueeze(1)

        full = torch.cat([z_emb, tok_emb], dim=1)                      # (B, L, d)
        full = full + self.pos_emb(torch.arange(full.size(1), device=device).unsqueeze(0))
        return full

    # ------------------------------------------------------------------
    # Training forward (teacher-forced, fully parallel)
    # ------------------------------------------------------------------

    def forward(self, z, input_tokens, target_tokens, target_types, seq_lens,
                context_dropout: float = 0.0):
        max_L  = target_tokens.size(1)
        device = z.device

        input_types = target_types[:, :max_L - 1]

        # Context dropout: randomly replace a fraction of input tokens with 0
        # (the padding/mask index).  This prevents the decoder from relying
        # purely on sequential context and forces it to use z.
        inp = input_tokens
        if context_dropout > 0.0 and self.training:
            drop_mask = torch.rand_like(inp, dtype=torch.float) < context_dropout
            inp = inp.masked_fill(drop_mask, 0)
        full_emb    = self._embed(z, inp, input_types)

        # No key-padding mask needed: padding always appears after EOS so the
        # causal mask already prevents real tokens from attending to pad positions.
        # Pad-token hidden states are excluded from the loss by token_mask below,
        # so they never generate gradients.  Passing no mask lets SDPA use
        # is_causal=True internally, which enables FlashAttention.
        h = self.transformer(full_emb)  # (B, max_L, d)

        token_mask = (target_types > 0)
        if not token_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        h_valid    = h[token_mask]               # (N_tokens, d)
        tgt_valid  = target_tokens[token_mask]   # (N_tokens,)
        type_valid = target_types[token_mask]    # (N_tokens,)

        n_mask = (type_valid == 1)
        e_mask = (type_valid == 2)

        # Compute each head's CE with reduction='sum', then normalise by total tokens.
        # This gives correct per-token loss with full gradient flow — no scatter-writes.
        total_tokens = token_mask.sum()
        loss = torch.tensor(0.0, device=device)
        if n_mask.any():
            loss = loss + F.cross_entropy(
                self.node_head(h_valid[n_mask]), tgt_valid[n_mask], reduction='sum')
        if e_mask.any():
            loss = loss + F.cross_entropy(
                self.edge_head(h_valid[e_mask]), tgt_valid[e_mask], reduction='sum')
        return loss / total_tokens.float()

    # ------------------------------------------------------------------
    # Inference: autoregressive sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_smiles(self, z: torch.Tensor, atom_decoder: dict,
                      charge_decoder, valency_mask: bool = True,
                      temperature: float = 1.0) -> list:
        """Autoregressively sample B molecules in parallel using KV caching.

        KV caching avoids re-processing past tokens: each step runs one
        (B, 1, d) token through all transformer layers, writing K/V into
        pre-allocated buffers.  Total attention work is O(L) per step.

        Always runs in eval mode (dropout disabled).
        """
        training = self.training
        self.eval()          # ensure dropout is off during sampling (issue 2.6)
        try:
            result = self._sample_batch(z, atom_decoder, charge_decoder,
                                        valency_mask, temperature)
        finally:
            self.train(training)  # restore original mode
        return result

    @torch.no_grad()
    def _sample_batch(self, z: torch.Tensor, atom_decoder: dict, charge_decoder,
                      valency_mask: bool, temperature: float) -> list:
        """KV-cached parallel AR sampling — fully vectorised per-step logic.

        The original per-molecule Python loop (O(B) Python iters per step) is
        replaced by batched tensor operations — two multinomial calls and a
        single (B, edge_vocab) broadcast for valency masking.  Python overhead
        per step is now O(1) rather than O(B), giving a large speedup for the
        ~300–700 sequential steps required per batch.
        """
        B        = z.size(0)
        device   = z.device
        T        = max(temperature, 1e-6)
        d        = self.d_model
        n_layers = len(self.transformer.layers)
        M        = self.max_atoms

        # ── Precompute per-class lookup tables (CPU → device, done once) ──
        # cls_idx = token − 1,  range [0, eos_id − 1]
        n_cls = self.eos_id
        anum_arr   = torch.zeros(n_cls, dtype=torch.long)
        fc_arr     = torch.zeros(n_cls, dtype=torch.long)
        maxval_arr = torch.zeros(n_cls, dtype=torch.long)
        for cls_idx in range(n_cls):
            an = atom_decoder.get(cls_idx, 6)
            fc = charge_decoder.get(cls_idx, 0) if charge_decoder else 0
            bv = _MAX_VALENCE.get(an, 4)
            if fc > 0:    bv += 1
            elif fc < 0:  bv = max(bv - 1, 0)
            anum_arr[cls_idx]   = an
            fc_arr[cls_idx]     = fc
            maxval_arr[cls_idx] = bv
        anum_lut   = anum_arr.to(device)     # (n_cls,)
        fc_lut     = fc_arr.to(device)       # (n_cls,)
        maxval_lut = maxval_arr.to(device)   # (n_cls,)

        # bond_order_lut[bt] = valence units consumed by bond type bt
        bo_arr = torch.ones(self.edge_vocab, dtype=torch.long)
        for k, v in _BOND_ORDER.items():
            if k < self.edge_vocab:
                bo_arr[k] = v
        bond_order_lut = bo_arr.to(device)   # (edge_vocab,)

        # ── Per-molecule state tensors ─────────────────────────────────────
        edges_remaining = torch.zeros(B, dtype=torch.long, device=device)
        current_atom_i  = torch.zeros(B, dtype=torch.long, device=device)
        n_atoms         = torch.zeros(B, dtype=torch.long, device=device)
        finished        = torch.zeros(B, dtype=torch.bool, device=device)
        val_used        = torch.zeros(B, M, dtype=torch.long, device=device)
        base_val_t      = torch.zeros(B, M, dtype=torch.long, device=device)
        atom_anum_t     = torch.zeros(B, M, dtype=torch.long, device=device)
        atom_fc_t       = torch.zeros(B, M, dtype=torch.long, device=device)
        bonds_t         = torch.zeros(B, M, M, dtype=torch.long, device=device)
        bidx            = torch.arange(B, device=device)

        # ── Pre-allocate K/V buffers ───────────────────────────────────────
        K_bufs = [torch.zeros(B, self.max_tf_len, d, device=device) for _ in range(n_layers)]
        V_bufs = [torch.zeros(B, self.max_tf_len, d, device=device) for _ in range(n_layers)]

        # ── Position 0: z prefix ──────────────────────────────────────────
        z_vec  = self.z_proj(z)                                        # (B, d_model)
        z_emb  = (z_vec.unsqueeze(1)
                  + self.type_emb(torch.zeros(B, 1, dtype=torch.long, device=device))
                  + self.pos_emb(torch.zeros(1, 1, dtype=torch.long, device=device)))
        last_h = self.transformer.step(z_emb, K_bufs, V_bufs, t=0)  # (B, d)

        for step in range(self.max_seq_len - 1):
            if finished.all():
                break

            node_logits = self.node_head(last_h) / T   # (B, node_vocab)
            edge_logits = self.edge_head(last_h) / T   # (B, edge_vocab)
            node_logits[:, 0] = float('-inf')           # never predict padding index

            active    = ~finished
            edge_mode = active & (edges_remaining > 0)
            node_mode = active & (edges_remaining == 0)

            new_toks  = torch.full((B,), self.eos_id, dtype=torch.long, device=device)
            new_types = torch.ones(B, dtype=torch.long, device=device)

            # ── Edge sampling ────────────────────────────────────────────
            if edge_mode.any():
                el = edge_logits.clone()
                if valency_mask:
                    ci        = current_atom_i.clamp(0, M - 1)
                    tj        = (current_atom_i - edges_remaining).clamp(0, M - 1)
                    ci_used   = val_used.gather(1, ci.unsqueeze(1)).squeeze(1)    # (B,)
                    ci_maxval = base_val_t.gather(1, ci.unsqueeze(1)).squeeze(1)  # (B,)
                    tj_used   = val_used.gather(1, tj.unsqueeze(1)).squeeze(1)    # (B,)
                    tj_maxval = base_val_t.gather(1, tj.unsqueeze(1)).squeeze(1)  # (B,)
                    # Broadcast once: (B, 1) op (1, edge_vocab) → (B, edge_vocab)
                    orders    = bond_order_lut.unsqueeze(0)                        # (1, V)
                    ci_exceed = (ci_used.unsqueeze(1) + orders) > ci_maxval.unsqueeze(1)
                    tj_exceed = (tj_used.unsqueeze(1) + orders) > tj_maxval.unsqueeze(1)
                    invalid   = edge_mode.unsqueeze(1) & (ci_exceed | tj_exceed)  # (B, V)
                    invalid[:, 0] = False   # no-bond (0) is always structurally valid
                    el.masked_fill_(invalid, float('-inf'))
                # Rows where every bond type is masked → force no-bond
                all_inf      = ~el.isfinite().any(dim=-1)                         # (B,)
                el_safe      = el.clone()
                el_safe[all_inf & edge_mode, 0] = 0.0
                probs_e      = F.softmax(el_safe, dim=-1).clamp(min=0)
                sampled_e    = torch.multinomial(probs_e, 1).squeeze(1)
                sampled_e    = torch.where(all_inf & edge_mode,
                                           torch.zeros_like(sampled_e), sampled_e)
                new_toks     = torch.where(edge_mode, sampled_e, new_toks)
                new_types    = torch.where(edge_mode,
                                           torch.full_like(new_types, 2), new_types)

            # ── Node / EOS sampling ──────────────────────────────────────
            if node_mode.any():
                probs_n   = F.softmax(node_logits, dim=-1).clamp(min=0)
                sampled_n = torch.multinomial(probs_n, 1).squeeze(1)
                new_toks  = torch.where(node_mode, sampled_n, new_toks)
                new_types = torch.where(node_mode, torch.ones_like(new_types), new_types)

            # ── State updates ────────────────────────────────────────────
            just_finished = node_mode & (new_toks == self.eos_id)
            finished      = finished | just_finished

            placed_atom = node_mode & ~just_finished
            if placed_atom.any():
                b_pa     = bidx[placed_atom]
                na_safe  = n_atoms[placed_atom].clamp(0, M - 1)
                cls_safe = (new_toks[placed_atom] - 1).clamp(0)
                atom_anum_t[b_pa, na_safe] = anum_lut[cls_safe]
                atom_fc_t  [b_pa, na_safe] = fc_lut  [cls_safe]
                base_val_t [b_pa, na_safe] = maxval_lut[cls_safe]
                n_atoms         = n_atoms + placed_atom.long()
                current_atom_i  = torch.where(placed_atom, n_atoms - 1, current_atom_i)
                edges_remaining = torch.where(placed_atom, n_atoms - 1, edges_remaining)

            placed_bond = edge_mode & (new_toks > 0)
            if placed_bond.any():
                b_pb  = bidx[placed_bond]
                ci_pb = current_atom_i[placed_bond].clamp(0, M - 1)
                tj_pb = (current_atom_i[placed_bond] - edges_remaining[placed_bond]).clamp(0, M - 1)
                bonds_t[b_pb, ci_pb, tj_pb] = new_toks[placed_bond]
                orders = bond_order_lut[new_toks[placed_bond]]
                val_used[b_pb, ci_pb] += orders
                val_used[b_pb, tj_pb] += orders

            edges_remaining = torch.where(edge_mode, edges_remaining - 1, edges_remaining)

            # ── KV cache advance ─────────────────────────────────────────
            active_l  = (~finished).long()
            tok_safe  = new_toks  * active_l
            type_safe = new_types * active_l
            n_w = (type_safe == 1).to(dtype=z.dtype).unsqueeze(-1)
            e_w = (type_safe == 2).to(dtype=z.dtype).unsqueeze(-1)
            pos     = step + 1
            tok_emb = (
                n_w * self.node_emb(tok_safe.clamp(0, self.node_vocab - 1))
                + e_w * self.edge_emb(tok_safe.clamp(0, self.edge_vocab - 1))
                + self.type_emb(type_safe)
                + self.pos_emb(torch.full((B,), pos, dtype=torch.long, device=device))
                + z_vec  # residual z injection — mirrors training path
            ).unsqueeze(1)
            last_h = self.transformer.step(tok_emb, K_bufs, V_bufs, t=pos)

        # ── Build RDKit molecules from tensor state (CPU) ─────────────────
        n_atoms_list  = n_atoms.cpu().tolist()
        anum_list     = atom_anum_t.cpu().tolist()
        fc_list       = atom_fc_t.cpu().tolist()
        bonds_list    = bonds_t.cpu().tolist()
        results = []
        for b in range(B):
            na    = int(n_atoms_list[b])
            atoms = [(int(anum_list[b][i]), int(fc_list[b][i]), None) for i in range(na)]
            bonds = {}
            for i in range(na):
                for j in range(i):
                    bt = int(bonds_list[b][i][j])
                    if bt > 0:
                        bonds[(i, j)] = _get_rdkit_bond(bt)
            results.append(self._build_mol(atoms, bonds))
        return results

    def _build_mol(self, atoms: list, bonds: dict):
        """Build and sanitize an RDKit molecule from decoded atoms and bonds."""
        if not atoms:
            return None
        mol = Chem.RWMol()
        for atomic_num, fc, _ in atoms:
            at = Chem.Atom(atomic_num)
            if fc != 0:
                at.SetFormalCharge(fc)
            mol.AddAtom(at)
        for (i, j), bond_type in bonds.items():
            if not mol.GetBondBetweenAtoms(i, j):
                mol.AddBond(i, j, bond_type)
        try:
            result = Chem.SanitizeMol(mol, catchErrors=True)
            if result != Chem.rdmolops.SanitizeFlags.SANITIZE_NONE:
                return None
            smi = Chem.MolToSmiles(mol)
            return smi if (smi and Chem.MolFromSmiles(smi)) else None
        except Exception:
            return None


# ---------------------------------------------------------------------------
# GraphVAEAR: GINEConv encoder + ARDecoder
# ---------------------------------------------------------------------------

class GraphVAEAR(nn.Module):
    """GraphVAE with an autoregressive Transformer decoder.

    Encoder:  identical to GraphVAE (4-layer GINEConv, global_add_pool).
    Decoder:  latent-conditioned causal Transformer (ARDecoder above).
    """

    def __init__(self, num_node_features: int, num_edge_features: int,
                 latent_dim: int = 128, max_atoms: int = 38,
                 ar_d_model: int = 256, ar_n_heads: int = 8,
                 ar_n_layers: int = 4, ar_d_ff: int = 512,
                 ar_dropout: float = 0.1,
                 prop_pred: bool = False,
                 context_dropout: float = 0.0):
        super().__init__()

        self.max_atoms         = max_atoms
        self.latent_dim        = latent_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.context_dropout   = context_dropout

        # ---- Encoder ----
        self.encoder = GINEConvEncoder(num_node_features, num_edge_features,
                                       hidden_dim=256, latent_dim=latent_dim)

        # ---- Autoregressive decoder ----
        self.decoder = ARDecoder(
            latent_dim=latent_dim,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            d_model=ar_d_model, n_heads=ar_n_heads,
            n_layers=ar_n_layers, d_ff=ar_d_ff,
            dropout=ar_dropout, max_atoms=max_atoms,
        )

        # Optional property prediction head (predicts from μ, same as GVAE)
        self.prop_head = PropertyHead(latent_dim) if prop_pred else None

    # ------------------------------------------------------------------

    def encode(self, x, edge_index, edge_attr, batch):
        return self.encoder(x, edge_index, edge_attr, batch)

    def reparameterize(self, mu, logvar):
        return GINEConvEncoder.reparameterize(mu, logvar)

    def predict_props(self, mu: torch.Tensor) -> torch.Tensor:
        if self.prop_head is None:
            raise RuntimeError("GraphVAEAR built without prop_pred=True")
        return self.prop_head(mu)

    def forward(self, x, edge_index, edge_attr, batch,
                input_tokens, target_tokens, target_types, seq_lens):
        """Teacher-forced training forward pass.

        BFS sequences are pre-built by ar_collate_fn in DataLoader workers.

        Returns
        -------
        recon_loss : scalar
        mu         : (B, latent_dim)
        logvar     : (B, latent_dim)
        """
        mu, logvar = self.encode(x, edge_index, edge_attr, batch)
        z = self.reparameterize(mu, logvar)
        recon_loss = self.decoder(z, input_tokens, target_tokens, target_types, seq_lens,
                                  context_dropout=self.context_dropout)
        return recon_loss, mu, logvar

    def sample_smiles(self, z, atom_decoder_dict=None, charge_decoder=None,
                      valency_mask: bool = True, temperature: float = 1.0) -> list:
        if atom_decoder_dict is None:
            atom_decoder_dict = {}
        return self.decoder.sample_smiles(
            z, atom_decoder_dict, charge_decoder,
            valency_mask=valency_mask, temperature=temperature,
        )


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def gvae_ar_loss(recon_loss: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                 kl_weight: float, free_bits: float = 0.0, capacity: float = 0.0):
    """Combine AR reconstruction loss with KL divergence.

    free_bits: minimum KL per latent dimension (nats).
    capacity:  target KL (nats); loss = kl_weight * |KL - capacity|.
    Returns (total, recon, kl) where kl is the raw (unweighted) divergence.
    """
    kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())  # (B, D)
    if free_bits > 0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl = kl_per_dim.sum(dim=1).mean()
    total = recon_loss + kl_weight * (kl - capacity).abs()
    return total, recon_loss, kl


# ---------------------------------------------------------------------------
# GraphVAEARNF: same AR decoder, IAF flow in the encoder path
# ---------------------------------------------------------------------------

class GraphVAEARNF(nn.Module):
    """GraphVAE with IAF normalizing flow encoder + autoregressive Transformer decoder.

    Architecture mirrors GraphVAENF (encoder + flow) but replaces the flat MLP
    decoder with the causal Transformer ARDecoder.

    The prior is p(zK) = N(0, I): during sampling, draw z ~ N(0, I) and
    decode directly — do NOT pass it through the flow.
    """

    def __init__(self, num_node_features: int, num_edge_features: int,
                 latent_dim: int = 128, max_atoms: int = 38,
                 num_flows: int = 4, flow_hidden_dim: int = 256,
                 ar_d_model: int = 256, ar_n_heads: int = 8,
                 ar_n_layers: int = 4, ar_d_ff: int = 512,
                 ar_dropout: float = 0.1,
                 prop_pred: bool = False,
                 context_dropout: float = 0.0):
        super().__init__()

        self.max_atoms        = max_atoms
        self.latent_dim        = latent_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features
        self.context_dropout   = context_dropout

        # ---- Encoder ----
        self.encoder = GINEConvEncoder(num_node_features, num_edge_features,
                                       hidden_dim=256, latent_dim=latent_dim)

        # ---- IAF flow ----
        self.flow = InverseAutoregressiveFlow(latent_dim, num_flows=num_flows,
                                              hidden_dim=flow_hidden_dim)

        # ---- Autoregressive decoder ----
        self.decoder = ARDecoder(
            latent_dim=latent_dim,
            num_node_features=num_node_features,
            num_edge_features=num_edge_features,
            d_model=ar_d_model, n_heads=ar_n_heads,
            n_layers=ar_n_layers, d_ff=ar_d_ff,
            dropout=ar_dropout, max_atoms=max_atoms,
        )

        # Optional property head predicts from μ₀ (pre-flow), same reasoning as GVAE_NF
        self.prop_head = PropertyHead(latent_dim) if prop_pred else None

    # ------------------------------------------------------------------

    def encode(self, x, edge_index, edge_attr, batch):
        return self.encoder(x, edge_index, edge_attr, batch)

    def reparameterize(self, mu, logvar):
        return GINEConvEncoder.reparameterize(mu, logvar)

    def predict_props(self, mu: torch.Tensor) -> torch.Tensor:
        """Predict from μ₀ (pre-flow) — same rationale as GraphVAENF."""
        if self.prop_head is None:
            raise RuntimeError("GraphVAEARNF built without prop_pred=True")
        return self.prop_head(mu)

    def forward(self, x, edge_index, edge_attr, batch,
                input_tokens, target_tokens, target_types, seq_lens):
        """Teacher-forced training forward pass.

        BFS sequences are pre-built by ar_collate_fn in DataLoader workers.

        Returns
        -------
        recon_loss  : scalar
        mu          : (B, latent_dim)
        logvar      : (B, latent_dim)
        z0          : (B, latent_dim)
        zK          : (B, latent_dim)
        sum_log_det : (B,)
        """
        mu, logvar = self.encode(x, edge_index, edge_attr, batch)
        z0 = self.reparameterize(mu, logvar)
        zK, sum_log_det = self.flow(z0)
        recon_loss = self.decoder(zK, input_tokens, target_tokens, target_types, seq_lens,
                                  context_dropout=self.context_dropout)
        return recon_loss, mu, logvar, z0, zK, sum_log_det

    def sample_smiles(self, z, atom_decoder_dict=None, charge_decoder=None,
                      valency_mask: bool = True, temperature: float = 1.0) -> list:
        # z ~ N(0, I) is already zK — the prior is p(zK) = N(0, I).
        # Decode directly; do NOT pass through the flow.
        if atom_decoder_dict is None:
            atom_decoder_dict = {}
        return self.decoder.sample_smiles(
            z, atom_decoder_dict, charge_decoder,
            valency_mask=valency_mask, temperature=temperature,
        )


def gvae_ar_nf_loss(recon_loss: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor,
                    z0: torch.Tensor, zK: torch.Tensor,
                    sum_log_det: torch.Tensor, kl_weight: float,
                    free_bits: float = 0.0, capacity: float = 0.0):
    """AR + NF combined loss (same KL formulation as gvae_nf_loss).

    free_bits: minimum KL per latent dimension applied to the base q0 term.
    capacity: target KL (nats); loss = kl_weight * |KL - capacity|.
    Returns (total, recon, kl_flow) where kl_flow is the raw divergence.
    """
    std = (0.5 * logvar).exp()
    # Per-dim contribution: log q_0(z_0) - log p(z_K) before the flow correction
    kl_per_dim = -0.5 * (logvar + ((z0 - mu) / (std + 1e-8)).pow(2)) + 0.5 * zK.pow(2)  # (B, D)
    if free_bits > 0:
        kl_per_dim = kl_per_dim.clamp(min=free_bits)
    kl_flow = kl_per_dim.sum(dim=1).mean() - sum_log_det.mean()
    total   = recon_loss + kl_weight * (kl_flow - capacity).abs()
    return total, recon_loss, kl_flow
