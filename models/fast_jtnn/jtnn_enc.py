import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any

class GraphGRU(nn.Module):
    """
    Message passing GRU for tree structures. 
    Optimized to remove CPU-GPU synchronization and redundant computations.
    """
    def __init__(self, input_size: int, hidden_size: int, depth: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.depth = depth

        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_r = nn.Linear(input_size, hidden_size, bias=False)
        self.U_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, h: torch.Tensor, x: torch.Tensor, mess_graph: torch.Tensor) -> torch.Tensor:
        # Optimization: Create the mask directly on the target device.
        # This avoids a host-to-device transfer on every forward pass.
        mask = torch.ones((h.size(0), 1), device=x.device, dtype=x.dtype)
        mask[0, 0] = 0.0  # The 0th index is used for padding and must remain zero

        # Optimization: x does not change across iterations. 
        # Pre-compute W_r(x) outside the loop to save matrix multiplications.
        r_1 = self.W_r(x).unsqueeze(1)  # Shape: [batch, 1, hidden_size]

        for _ in range(self.depth):
            # Native PyTorch indexing replaces legacy index_select_ND
            h_nei = h[mess_graph]
            sum_h = h_nei.sum(dim=1)
            
            z_input = torch.cat([x, sum_h], dim=1)
            z = torch.sigmoid(self.W_z(z_input))

            r_2 = self.U_r(h_nei)
            r = torch.sigmoid(r_1 + r_2)
            
            gated_h = r * h_nei
            sum_gated_h = gated_h.sum(dim=1)
            
            h_input = torch.cat([x, sum_gated_h], dim=1)
            pre_h = torch.tanh(self.W_h(h_input))
            
            h = (1.0 - z) * sum_h + z * pre_h
            h = h * mask  # Zero out the padding vector safely for autograd

        return h


class JTNNEncoder(nn.Module):
    """
    Junction Tree Neural Network Encoder.
    """
    def __init__(self, hidden_size: int, depth: int, embedding: nn.Embedding):
        super().__init__()
        self.hidden_size = hidden_size
        self.depth = depth
        self.embedding = embedding

        self.output_nn = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.ReLU()
        )
        self.gru = GraphGRU(hidden_size, hidden_size, depth=depth)

    def forward(self, fnode: torch.Tensor, fmess: torch.Tensor, 
                node_graph: torch.Tensor, mess_graph: torch.Tensor, 
                scope: List[Tuple[int, int]]) -> Tuple[torch.Tensor, torch.Tensor]:
        
        # Dynamically infer device from the embedding layer weights
        device = self.embedding.weight.device

        # Move tensors to correct device natively (removes legacy create_var)
        fnode = fnode.to(device)
        fmess = fmess.to(device)
        node_graph = node_graph.to(device)
        mess_graph = mess_graph.to(device)

        # Initialize messages directly on the device
        messages = torch.zeros((mess_graph.size(0), self.hidden_size), device=device)

        fnode_emb = self.embedding(fnode)
        
        # Native advanced indexing replaces index_select_ND
        fmess_emb = fnode_emb[fmess] 
        messages = self.gru(messages, fmess_emb, mess_graph)

        # Native indexing replaces index_select_sum
        mess_nei = messages[node_graph].sum(dim=1)
        
        node_vecs = torch.cat([fnode_emb, mess_nei], dim=-1)
        node_vecs = self.output_nn(node_vecs)

        # Optimization: Extract root vectors using a single tensor index operation
        root_indices = [st for st, _ in scope]
        tree_vecs = node_vecs[root_indices]

        return tree_vecs, messages

    @staticmethod
    def tensorize(tree_batch: List[Any]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]], Dict[Tuple[int, int], int]]:
        node_batch = [] 
        scope = []
        for tree in tree_batch:
            scope.append((len(node_batch), len(tree.nodes)))
            node_batch.extend(tree.nodes)

        return JTNNEncoder.tensorize_nodes(node_batch, scope)
    
    @staticmethod
    def tensorize_nodes(node_batch: List[Any], 
                        scope: List[Tuple[int, int]]) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[Tuple[int, int]]], Dict[Tuple[int, int], int]]:
        
        # Linter fix: Explicitly type the list to prevent "tuple cannot be assigned to None" error
        messages: List[Tuple[Any, Any]] = [(None, None)]  # 1-based indexing so that 0 can serve as padding
        mess_dict: Dict[Tuple[int, int], int] = {}
        fnode = []
        
        for x in node_batch:
            fnode.append(x.wid)
            for y in x.neighbors:
                mess_dict[(x.idx, y.idx)] = len(messages)
                messages.append((x, y))

        node_graph = [[] for _ in range(len(node_batch))]
        mess_graph = [[] for _ in range(len(messages))]
        fmess = [0] * len(messages)

        for x, y in messages[1:]:
            mid1 = mess_dict[(x.idx, y.idx)]
            fmess[mid1] = x.idx 
            node_graph[y.idx].append(mid1)
            for z in y.neighbors:
                if z.idx != x.idx:
                    mid2 = mess_dict[(y.idx, z.idx)]
                    mess_graph[mid2].append(mid1)

        # Calculate max lengths safely using generators
        max_node_deg = max((len(t) for t in node_graph), default=0)
        max_mess_deg = max((len(t) for t in mess_graph), default=0)

        # Pad lengths directly
        for t in node_graph:
            t.extend([0] * (max_node_deg - len(t)))

        for t in mess_graph:
            t.extend([0] * (max_mess_deg - len(t)))

        # Convert to native PyTorch tensors
        mess_graph_t = torch.tensor(mess_graph, dtype=torch.long)
        node_graph_t = torch.tensor(node_graph, dtype=torch.long)
        fmess_t = torch.tensor(fmess, dtype=torch.long)
        fnode_t = torch.tensor(fnode, dtype=torch.long)

        # Linter fix: explicitly returning the fully typed structure
        return (fnode_t, fmess_t, node_graph_t, mess_graph_t, scope), mess_dict
