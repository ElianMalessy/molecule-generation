import copy
from typing import List, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import rdkit.Chem as Chem

from models.fast_jtnn.mol_tree import MolTree
from models.fast_jtnn.jtnn_enc import JTNNEncoder
from models.fast_jtnn.jtnn_dec import JTNNDecoder
from models.fast_jtnn.mpn import MPN
from models.fast_jtnn.jtmpn import JTMPN
from models.fast_jtnn.datautils import tensorize
from models.fast_jtnn.chemutils import enum_assemble, set_atommap, copy_edit_mol, attach_mols


class JTNNVAE(nn.Module):
    def __init__(self, vocab, hidden_size: int, latent_size: int, depthT: int, depthG: int):
        super().__init__()
        self.vocab = vocab
        self.hidden_size = hidden_size
        self.latent_size = latent_size // 2  # Tree and Mol each get half the latent space

        # Encoders & Decoders
        self.jtnn = JTNNEncoder(hidden_size, depthT, nn.Embedding(vocab.size(), hidden_size))
        self.decoder = JTNNDecoder(vocab, hidden_size, self.latent_size, nn.Embedding(vocab.size(), hidden_size))

        # Message Passing Networks
        self.jtmpn = JTMPN(hidden_size, depthG)
        self.mpn = MPN(hidden_size, depthG)

        # Assembly & VAE Layers
        self.A_assm = nn.Linear(self.latent_size, hidden_size, bias=False)
        self.assm_loss = nn.CrossEntropyLoss(reduction='sum')

        self.T_mean = nn.Linear(hidden_size, self.latent_size)
        self.T_var = nn.Linear(hidden_size, self.latent_size)
        self.G_mean = nn.Linear(hidden_size, self.latent_size)
        self.G_var = nn.Linear(hidden_size, self.latent_size)

    @property
    def device(self) -> torch.device:
        """Dynamically infer the device from the model's parameters."""
        return next(self.parameters()).device

    def encode(self, jtenc_holder: tuple, mpn_holder: tuple) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tree_vecs, tree_mess = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        return tree_vecs, tree_mess, mol_vecs

    def encode_from_smiles(self, smiles_list: List[str]) -> torch.Tensor:
        tree_batch = [MolTree(s) for s in smiles_list]
        _, jtenc_holder, mpn_holder = tensorize(tree_batch, self.vocab, assm=False)
        tree_vecs, _, mol_vecs = self.encode(jtenc_holder, mpn_holder)
        return torch.cat([tree_vecs, mol_vecs], dim=-1)

    def encode_latent(self, jtenc_holder: tuple, mpn_holder: tuple) -> Tuple[torch.Tensor, torch.Tensor]:
        tree_vecs, _ = self.jtnn(*jtenc_holder)
        mol_vecs = self.mpn(*mpn_holder)
        
        tree_mean = self.T_mean(tree_vecs)
        mol_mean = self.G_mean(mol_vecs)
        
        # Following Mueller et al. stabilization trick
        tree_var = -torch.abs(self.T_var(tree_vecs))
        mol_var = -torch.abs(self.G_var(mol_vecs))
        
        return torch.cat([tree_mean, mol_mean], dim=1), torch.cat([tree_var, mol_var], dim=1)

    def rsample(self, z_vecs: torch.Tensor, W_mean: nn.Linear, W_var: nn.Linear) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reparameterization trick with sum-scaled KL Divergence."""
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs(W_var(z_vecs)) 
        
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
        
        epsilon = torch.randn_like(z_mean)
        z_sampled = z_mean + torch.exp(z_log_var / 2) * epsilon
        
        return z_sampled, kl_loss

    def sample_prior(self, prob_decode: bool = False) -> Optional[str]:
        z_tree = torch.randn(1, self.latent_size, device=self.device)
        z_mol = torch.randn(1, self.latent_size, device=self.device)
        return self.decode(z_tree, z_mol, prob_decode)

    def forward(self, x_batch: tuple, beta: float) -> Tuple[torch.Tensor, float, float, float, float]:
        x_batch_data, x_jtenc_holder, x_mpn_holder, x_jtmpn_holder = x_batch
        
        x_tree_vecs, x_tree_mess, x_mol_vecs = self.encode(x_jtenc_holder, x_mpn_holder)
        
        z_tree_vecs, tree_kl = self.rsample(x_tree_vecs, self.T_mean, self.T_var)
        z_mol_vecs, mol_kl = self.rsample(x_mol_vecs, self.G_mean, self.G_var)
        
        kl_div = tree_kl + mol_kl
        word_loss, topo_loss, word_acc, topo_acc = self.decoder(x_batch_data, z_tree_vecs)
        assm_loss, assm_acc = self.assm(x_batch_data, x_jtmpn_holder, z_mol_vecs, x_tree_mess)
        
        total_loss = word_loss + topo_loss + assm_loss + beta * kl_div
        return total_loss, kl_div.item(), word_acc, topo_acc, assm_acc

    def assm(self, mol_batch: List[MolTree], jtmpn_holder: tuple, x_mol_vecs: torch.Tensor, x_tree_mess: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Fully vectorized assembly loss and accuracy calculation."""
        jtmpn_data, batch_idx = jtmpn_holder
        fatoms, fbonds, agraph, bgraph, scope = jtmpn_data
        
        # Move inputs to device
        batch_idx = torch.tensor(batch_idx, dtype=torch.long, device=self.device)
        cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, x_tree_mess)
        
        x_mol_vecs = self.A_assm(x_mol_vecs.index_select(0, batch_idx))
        
        # [Total_Candidates]
        scores = torch.bmm(x_mol_vecs.unsqueeze(1), cand_vecs.unsqueeze(-1)).squeeze(-1)
        
        # 1. Extract dimensions and labels in purely Python (fast, no CUDA syncs)
        cand_sizes = []
        labels = []
        for mol_tree in mol_batch:
            for node in mol_tree.nodes:
                if len(node.cands) > 1 and not node.is_leaf:
                    cand_sizes.append(len(node.cands))
                    labels.append(node.cands.index(node.label))
        
        if not cand_sizes:
            return torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

        # 2. Vectorize the loss computation using pad_sequence
        # Split the 1D scores tensor into individual chunks per node
        score_chunks = torch.split(scores, cand_sizes)
        
        # Pad chunks to [Num_Nodes, Max_Candidates]. 
        # Padding with -1e9 ensures padded elements have 0 probability after softmax in CrossEntropy.
        padded_scores = pad_sequence(score_chunks, batch_first=True, padding_value=-1e9)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=self.device)
        
        # Compute loss in a single batched operation
        total_loss = self.assm_loss(padded_scores, labels_tensor)
        
        # Compute accuracy in a single batched operation
        preds = padded_scores.argmax(dim=1)
        final_acc = (preds == labels_tensor).float().mean().item()
        
        return total_loss, final_acc

    def decode(self, x_tree_vecs: torch.Tensor, x_mol_vecs: torch.Tensor, prob_decode: bool) -> Optional[str]:
        assert x_tree_vecs.size(0) == 1 and x_mol_vecs.size(0) == 1

        decode_result = self.decoder.decode(x_tree_vecs, prob_decode)
        if decode_result is None:
            return None
        pred_root, pred_nodes = decode_result
        
        if len(pred_nodes) == 0: return None
        if len(pred_nodes) == 1: return pred_root.smiles

        # Mark nid & is_leaf & atommap
        for i, node in enumerate(pred_nodes):
            node.nid = i + 1
            node.is_leaf = (len(node.neighbors) == 1)
            if len(node.neighbors) > 1:
                set_atommap(node.mol, node.nid)

        scope = [(0, len(pred_nodes))]
        jtenc_holder, mess_dict = JTNNEncoder.tensorize_nodes(pred_nodes, scope)
        _, tree_mess = self.jtnn(*jtenc_holder)
        tree_mess = (tree_mess, mess_dict)

        x_mol_vecs = self.A_assm(x_mol_vecs).squeeze(0)

        cur_mol = copy_edit_mol(pred_root.mol)
        global_amap = [{}] + [{} for _ in pred_nodes]
        global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}

        cur_mol, _ = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=True)
        
        if cur_mol is None: 
            cur_mol = copy_edit_mol(pred_root.mol)
            global_amap = [{}] + [{} for _ in pred_nodes]
            global_amap[1] = {atom.GetIdx(): atom.GetIdx() for atom in cur_mol.GetAtoms()}
            cur_mol, pre_mol = self.dfs_assemble(tree_mess, x_mol_vecs, pred_nodes, cur_mol, global_amap, [], pred_root, None, prob_decode, check_aroma=False)
            if cur_mol is None: cur_mol = pre_mol

        if cur_mol is None: return None

        cur_mol = cur_mol.GetMol()
        set_atommap(cur_mol)
        cur_mol = Chem.MolFromSmiles(Chem.MolToSmiles(cur_mol))
        
        return Chem.MolToSmiles(cur_mol) if cur_mol is not None else None
        
    def dfs_assemble(self, y_tree_mess: tuple, x_mol_vecs: torch.Tensor, all_nodes: list, cur_mol: Any, global_amap: list, fa_amap: list, cur_node: Any, fa_node: Any, prob_decode: bool, check_aroma: bool):
        fa_nid = fa_node.nid if fa_node is not None else -1
        prev_nodes = [fa_node] if fa_node is not None else []

        children = [nei for nei in cur_node.neighbors if nei.nid != fa_nid]
        neighbors = [nei for nei in children if nei.mol.GetNumAtoms() > 1]
        neighbors = sorted(neighbors, key=lambda x: x.mol.GetNumAtoms(), reverse=True)
        singletons = [nei for nei in children if nei.mol.GetNumAtoms() == 1]
        neighbors = singletons + neighbors

        cur_amap = [(fa_nid, a2, a1) for nid, a1, a2 in fa_amap if nid == cur_node.nid]
        cands, aroma_score = enum_assemble(cur_node, neighbors, prev_nodes, cur_amap)
        
        if len(cands) == 0 or (sum(aroma_score) < 0 and check_aroma):
            return None, cur_mol

        cand_smiles, cand_amap = zip(*cands)
        aroma_tensor = torch.tensor(aroma_score, dtype=torch.float32, device=self.device)
        cands = [(smiles, all_nodes, cur_node) for smiles in cand_smiles]

        if len(cands) > 1:
            jtmpn_holder = JTMPN.tensorize(cands, y_tree_mess[1])
            fatoms, fbonds, agraph, bgraph, scope = jtmpn_holder
            cand_vecs = self.jtmpn(fatoms, fbonds, agraph, bgraph, scope, y_tree_mess[0])
            scores = torch.mv(cand_vecs, x_mol_vecs) + aroma_tensor
        else:
            scores = torch.tensor([1.0], device=self.device)

        if prob_decode:
            probs = F.softmax(scores.unsqueeze(0), dim=1).squeeze(0)
            probs = probs.clamp(min=1e-7)
            probs = probs / probs.sum() # Strictly re-normalize
            cand_idx = torch.multinomial(probs, probs.numel())
        else:
            _, cand_idx = torch.sort(scores, descending=True)

        backup_mol = Chem.RWMol(cur_mol)
        pre_mol = cur_mol
        
        for i in range(cand_idx.numel()):
            cur_mol = Chem.RWMol(backup_mol)
            pred_amap = cand_amap[cand_idx[i].item()]
            new_global_amap = copy.deepcopy(global_amap)

            for nei_id, ctr_atom, nei_atom in pred_amap:
                if nei_id == fa_nid: continue
                new_global_amap[nei_id][nei_atom] = new_global_amap[cur_node.nid][ctr_atom]

            cur_mol = attach_mols(cur_mol, children, [], new_global_amap)
            new_mol = cur_mol.GetMol()
            new_mol = Chem.MolFromSmiles(Chem.MolToSmiles(new_mol))

            if new_mol is None: continue
            
            has_error = False
            for nei_node in children:
                if nei_node.is_leaf: continue
                
                tmp_mol, tmp_mol2 = self.dfs_assemble(
                    y_tree_mess, x_mol_vecs, all_nodes, cur_mol, 
                    new_global_amap, pred_amap, nei_node, cur_node, prob_decode, check_aroma
                )
                
                if tmp_mol is None: 
                    has_error = True
                    if i == 0: pre_mol = tmp_mol2
                    break
                
                cur_mol = tmp_mol

            if not has_error: return cur_mol, cur_mol

        return None, pre_mol
