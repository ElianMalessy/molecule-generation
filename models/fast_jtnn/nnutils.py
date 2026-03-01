import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from typing import List, Tuple

def index_select_ND(source: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    index_size = index.size()
    suffix_dim = source.size()[1:]
    final_size = index_size + suffix_dim
    target = source.index_select(dim, index.view(-1))
    return target.view(final_size)

def index_select_sum(source: torch.Tensor, dim: int, index: torch.Tensor) -> torch.Tensor:
    """
    Modern PyTorch handles memory fragmentation much better.
    This replaces the slow Python `for` loop with highly optimized native tensor indexing.
    """
    # source[index] gives [N, max_neighbors, hidden_size]
    # sum(dim=1) reduces it to [N, hidden_size]
    return source[index].sum(dim=1)

def avg_pool(all_vecs: torch.Tensor, scope: list, dim: int) -> torch.Tensor:
    size = torch.tensor([le for _, le in scope], dtype=torch.float32, device=all_vecs.device)
    return all_vecs.sum(dim=dim) / size.unsqueeze(-1)

def stack_pad_tensor(tensor_list: List[Tensor]) -> Tensor:
    """Pads a list of 2D tensors to the maximum length and stacks them."""
    # Replaces the manual F.pad loop with PyTorch's native C++ padding function
    return pad_sequence(tensor_list, batch_first=True)

def flatten_tensor(tensor: Tensor, scope: List[Tuple[int, int]]) -> Tensor:
    """Converts a 3D padded tensor to a 2D matrix, removing padded zeros."""
    # List comprehension replaces the manual for-loop with append
    return torch.cat([tensor[i, :le] for i, (_, le) in enumerate(scope)], dim=0)

def inflate_tensor(tensor: Tensor, scope: List[Tuple[int, int]]) -> Tensor:
    """Converts a 2D matrix back to a 3D padded tensor."""
    # Extract variable-length slices based on scope
    slices = [tensor[st : st + le] for st, le in scope]
    # pad_sequence handles max length calculation and zero-padding natively
    return pad_sequence(slices, batch_first=True)

def GRU(x: Tensor, h_nei: Tensor, W_z: nn.Module, W_r: nn.Module, U_r: nn.Module, W_h: nn.Module) -> Tensor:
    """Custom Graph GRU cell implementation."""
    hidden_size = x.size(-1)
    sum_h = h_nei.sum(dim=1)
    
    # Compute update gate z
    z_input = torch.cat([x, sum_h], dim=1)
    z = torch.sigmoid(W_z(z_input))
    
    # Compute reset gate r
    r_1 = W_r(x).view(-1, 1, hidden_size)
    r_2 = U_r(h_nei)
    r = torch.sigmoid(r_1 + r_2)
    
    # Compute candidate hidden state pre_h
    gated_h = r * h_nei
    sum_gated_h = gated_h.sum(dim=1)
    h_input = torch.cat([x, sum_gated_h], dim=1)
    pre_h = torch.tanh(W_h(h_input))
    
    # Update state
    return (1.0 - z) * sum_h + z * pre_h
