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
        for nb in range(n):
            if adj_bin[node, nb] and not visited[nb]:
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
                for nb in range(n):
                    if adj_bin[node, nb] and not visited[nb]:
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

    # input_tokens: seq[:-1], padded with 1 (a safe non-padding atom index)
    # target_tokens: seq,      padded with -1 (ignored by cross_entropy)
    input_tokens  = torch.ones (B, max_L - 1, dtype=torch.long, device=device)
    target_tokens = torch.full ((B, max_L),   -1, dtype=torch.long, device=device)
    target_types  = torch.zeros(B, max_L,         dtype=torch.long, device=device)
    seq_lens      = torch.zeros(B,                dtype=torch.long, device=device)

    for b, (tok, typ) in enumerate(zip(all_tokens, all_types)):
        L = min(len(tok), max_L)
        seq_lens[b] = L
        if L > 1:
            input_tokens [b, :L - 1] = torch.tensor(tok[:L - 1], dtype=torch.long, device=device)
        target_tokens[b, :L]     = torch.tensor(tok[:L],     dtype=torch.long, device=device)
        target_types [b, :L]     = torch.tensor(typ[:L],     dtype=torch.long, device=device)

    return input_tokens, target_tokens, target_types, seq_lens


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
        # Transformer input length = 1 (z) + max_seq_len - 1 (graph tokens) = max_seq_len
        self.max_tf_len      = self.max_seq_len

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

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,  # Pre-LN for stability
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=n_layers, enable_nested_tensor=False,
        )

        # Dual output heads
        self.node_head = nn.Linear(d_model, self.node_vocab)
        self.edge_head = nn.Linear(d_model, self.edge_vocab)

    # ------------------------------------------------------------------
    # Internal: build transformer input embeddings
    # ------------------------------------------------------------------

    def _embed(self, z: torch.Tensor, input_tokens: torch.Tensor,
               input_types: torch.Tensor) -> torch.Tensor:
        """Build full embedding sequence [z_prefix | graph_tokens].

        Parameters
        ----------
        z            : (B, latent_dim)
        input_tokens : (B, L-1) – seq[:-1], padded
        input_types  : (B, L-1) – 1=node/2=edge at each position, 0=padding

        Returns
        -------
        emb : (B, L, d_model)
        """
        B = z.size(0)
        L_in = input_tokens.size(1)
        device = z.device

        # Embed graph tokens with separate tables + type signal
        tok_emb = torch.zeros(B, L_in, self.d_model, device=device)
        n_mask = (input_types == 1)
        e_mask = (input_types == 2)
        if n_mask.any():
            tok_emb[n_mask] = self.node_emb(input_tokens[n_mask])
        if e_mask.any():
            tok_emb[e_mask] = self.edge_emb(input_tokens[e_mask])
        tok_emb = tok_emb + self.type_emb(input_types)

        # Z prefix: linear projection + type-0 embedding
        z_emb = self.z_proj(z).unsqueeze(1)                                    # (B, 1, d)
        z_emb = z_emb + self.type_emb(torch.zeros(B, 1, dtype=torch.long, device=device))

        full = torch.cat([z_emb, tok_emb], dim=1)                              # (B, L, d)

        # Positional encoding
        L = full.size(1)
        full = full + self.pos_emb(torch.arange(L, device=device).unsqueeze(0))
        return full

    # ------------------------------------------------------------------
    # Training forward (teacher-forced, fully parallel)
    # ------------------------------------------------------------------

    def forward(self, z: torch.Tensor, input_tokens: torch.Tensor,
                target_tokens: torch.Tensor, target_types: torch.Tensor,
                seq_lens: torch.Tensor) -> torch.Tensor:
        """Teacher-forced training pass.

        The entire batch is processed in one parallel Transformer forward pass.

        Parameters
        ----------
        z             : (B, latent_dim)
        input_tokens  : (B, max_L-1) – seq[:-1] per molecule
        target_tokens : (B, max_L)   – seq per molecule (includes EOS)
        target_types  : (B, max_L)   – 1=node, 2=edge, 0=padding
        seq_lens      : (B,)         – actual sequence length per molecule

        Returns
        -------
        recon_loss : scalar – cross-entropy per token (nodes + edges)
        """
        B = z.size(0)
        max_L = target_tokens.size(1)
        device = z.device

        # Input types equal target types shifted right by 1 position:
        # input at transformer-pos t is seq[t-1], whose type is target_types[:, t-1].
        # For position 0 (z prefix), we handle separately in _embed.
        # So input_types = target_types[:, :max_L-1]
        input_types = target_types[:, :max_L - 1]   # (B, max_L-1)

        full_emb = self._embed(z, input_tokens, input_types)  # (B, max_L, d)

        # Causal mask (upper triangular → attend to self and past only)
        causal_mask = torch.triu(
            torch.ones(max_L, max_L, device=device, dtype=torch.bool), diagonal=1
        )

        # Key-padding mask: True at positions t ≥ seq_lens[b]
        # (position 0 is z—never padded; input at pos t corresponds to seq[t-1])
        pos  = torch.arange(max_L, device=device).unsqueeze(0)     # (1, max_L)
        kpm  = pos >= seq_lens.unsqueeze(1)                         # (B, max_L)

        h = self.transformer(full_emb, mask=causal_mask,
                             src_key_padding_mask=kpm,
                             is_causal=False)                       # (B, max_L, d)

        # Compute loss: select positions by type, apply appropriate head
        node_mask = (target_types == 1)  # (B, max_L)
        edge_mask = (target_types == 2)

        recon_loss = torch.tensor(0.0, device=device)
        n_terms = 0

        if node_mask.any():
            node_logits = self.node_head(h[node_mask])       # (N_node, node_vocab)
            node_tgts   = target_tokens[node_mask]           # (N_node,)
            recon_loss  = recon_loss + F.cross_entropy(node_logits, node_tgts, ignore_index=-1)
            n_terms += 1

        if edge_mask.any():
            edge_logits = self.edge_head(h[edge_mask])       # (N_edge, edge_vocab)
            edge_tgts   = target_tokens[edge_mask]           # (N_edge,)
            recon_loss  = recon_loss + F.cross_entropy(edge_logits, edge_tgts, ignore_index=-1)
            n_terms += 1

        return recon_loss / max(n_terms, 1)

    # ------------------------------------------------------------------
    # Inference: autoregressive sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _step(self, z: torch.Tensor, seq_tokens: list, seq_types: list):
        """Single autoregressive step.

        Runs a full forward pass on [z, seq_tokens] and returns the hidden
        state at the last position, which predicts the next token.

        Parameters
        ----------
        z          : (1, latent_dim)
        seq_tokens : current generated tokens (list of int)
        seq_types  : corresponding types      (list of int: 1 or 2)

        Returns
        -------
        last_h : (1, d_model)
        """
        device = z.device
        L_in = len(seq_tokens)

        if L_in == 0:
            tok_t  = torch.zeros(1, 0, dtype=torch.long, device=device)
            type_t = torch.zeros(1, 0, dtype=torch.long, device=device)
        else:
            tok_t  = torch.tensor(seq_tokens, dtype=torch.long, device=device).unsqueeze(0)
            type_t = torch.tensor(seq_types,  dtype=torch.long, device=device).unsqueeze(0)

        full_emb = self._embed(z, tok_t, type_t)   # (1, L_in+1, d)
        seq_L = full_emb.size(1)
        causal = torch.triu(
            torch.ones(seq_L, seq_L, device=device, dtype=torch.bool), diagonal=1
        )
        h = self.transformer(full_emb, mask=causal, is_causal=False)  # (1, L_in+1, d)
        return h[:, -1, :]                                              # (1, d)

    @torch.no_grad()
    def sample_smiles(self, z: torch.Tensor, atom_decoder: dict,
                      charge_decoder, valency_mask: bool = True,
                      temperature: float = 1.0) -> list:
        """Autoregressively sample molecules from latent vectors.

        Parameters
        ----------
        z              : (B, latent_dim) – sampled from N(0, I)
        atom_decoder   : {class_idx: atomic_num}
        charge_decoder : {class_idx: formal_charge} or None
        valency_mask   : if True, masks chemically invalid bond choices
        temperature    : sampling temperature (1.0 = no rescaling)

        Returns
        -------
        list of B SMILES strings (None where sanitization failed)
        """
        B = z.size(0)
        results = []
        for b in range(B):
            smi = self._sample_one(z[b:b + 1], atom_decoder, charge_decoder,
                                   valency_mask, temperature)
            results.append(smi)
        return results

    def _sample_one(self, z1, atom_decoder, charge_decoder,
                    valency_mask: bool, temperature: float):
        """Sample a single molecule.  Returns SMILES or None."""
        seq_tokens, seq_types = [], []
        atoms      = []   # list of (atomic_num, fc, max_val)
        val_used   = []   # current valence per atom
        bonds      = {}   # (i, j) → rdkit bond type (i > j)

        # State machine: after placing atom i we need i edge tokens
        edges_remaining = 0   # edges left for the current new atom
        current_atom_i  = 0   # index (0-based) of atom currently getting edges

        max_steps = self.max_seq_len
        for _ in range(max_steps):
            last_h = self._step(z1, seq_tokens, seq_types)   # (1, d)

            if edges_remaining == 0:
                # ---- NEXT TOKEN IS A NODE ----
                logits = self.node_head(last_h)[0] / max(temperature, 1e-6)  # (V_node,)
                logits[0] = float('-inf')  # never predict padding class (index 0)
                token = int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())

                if token == self.eos_id:
                    break

                seq_tokens.append(token)
                seq_types.append(1)

                cls_idx   = token - 1                               # 0-indexed class
                atomic_num = atom_decoder.get(cls_idx, 6)
                fc = 0
                if charge_decoder is not None:
                    fc = charge_decoder.get(cls_idx, 0)
                base_val  = _MAX_VALENCE.get(atomic_num, 4)
                if fc > 0:
                    base_val += 1
                elif fc < 0:
                    base_val = max(base_val - 1, 0)

                atoms.append((atomic_num, fc, base_val))
                val_used.append(0)
                current_atom_i  = len(atoms) - 1
                edges_remaining = current_atom_i   # need edges to atoms 0..i-1

            else:
                # ---- NEXT TOKEN IS AN EDGE ----
                target_j = current_atom_i - edges_remaining  # which prior atom
                logits = self.edge_head(last_h)[0] / max(temperature, 1e-6)  # (V_edge,)

                if valency_mask:
                    for b_type in range(1, self.edge_vocab):
                        order = _BOND_ORDER.get(b_type, 1)
                        if (val_used[current_atom_i] + order > atoms[current_atom_i][2] or
                                val_used[target_j] + order > atoms[target_j][2]):
                            logits[b_type] = float('-inf')

                token = int(torch.multinomial(F.softmax(logits, dim=-1), 1).item())
                seq_tokens.append(token)
                seq_types.append(2)

                if token > 0:
                    order = _BOND_ORDER.get(token, 1)
                    val_used[current_atom_i] += order
                    val_used[target_j]       += order
                    bonds[(current_atom_i, target_j)] = _get_rdkit_bond(token)

                edges_remaining -= 1

        # ---- Build RDKit molecule ----
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
                 prop_pred: bool = False):
        super().__init__()

        self.max_atoms         = max_atoms
        self.latent_dim        = latent_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

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

    def forward(self, x, edge_index, edge_attr, batch, target_nodes, target_edges):
        """Teacher-forced training forward pass.

        Returns
        -------
        recon_loss : scalar
        mu         : (B, latent_dim)
        logvar     : (B, latent_dim)
        """
        mu, logvar = self.encode(x, edge_index, edge_attr, batch)
        z = self.reparameterize(mu, logvar)

        abs_max_len = self.max_atoms * (self.max_atoms + 1) // 2 + 1
        eos_id = self.decoder.eos_id

        input_tokens, target_tokens, target_types, seq_lens = build_ar_batch(
            target_nodes, target_edges, eos_id, abs_max_len,
        )

        recon_loss = self.decoder(z, input_tokens, target_tokens, target_types, seq_lens)
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
                 kl_weight: float):
    """Combine AR reconstruction loss with KL divergence.

    Returns (total, recon, kl) where kl is the raw (unweighted) divergence.
    """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
    total = recon_loss + kl_weight * kl
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
                 prop_pred: bool = False):
        super().__init__()

        self.max_atoms        = max_atoms
        self.latent_dim        = latent_dim
        self.num_node_features = num_node_features
        self.num_edge_features = num_edge_features

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

    def forward(self, x, edge_index, edge_attr, batch, target_nodes, target_edges):
        """Teacher-forced training forward pass.

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

        abs_max_len = self.max_atoms * (self.max_atoms + 1) // 2 + 1
        eos_id = self.decoder.eos_id

        input_tokens, target_tokens, target_types, seq_lens = build_ar_batch(
            target_nodes, target_edges, eos_id, abs_max_len,
        )

        recon_loss = self.decoder(zK, input_tokens, target_tokens, target_types, seq_lens)
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
                    sum_log_det: torch.Tensor, kl_weight: float):
    """AR + NF combined loss (same KL formulation as gvae_nf_loss).

    Returns (total, recon, kl_flow) where kl_flow is the raw divergence.
    """
    std = (0.5 * logvar).exp()
    log_q0  = -0.5 * (logvar + ((z0 - mu) / (std + 1e-8)).pow(2))
    log_pK  = -0.5 * zK.pow(2)
    kl_flow = (log_q0 - log_pK).sum(dim=1).mean() - sum_log_det.mean()
    total   = recon_loss + kl_weight * kl_flow
    return total, recon_loss, kl_flow
