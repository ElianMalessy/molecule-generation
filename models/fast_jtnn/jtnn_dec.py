import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Any

MAX_NB = 30
MAX_DECODE_LEN = 100

class JTNNDecoder(nn.Module):
    """
    Junction Tree Neural Network Decoder.
    Refactored for modern PyTorch idioms, type safety, and performance.
    """

    def __init__(self, vocab: Any, hidden_size: int, latent_size: int, embedding: nn.Embedding):
        super().__init__()  # Modern super() call without arguments
        self.hidden_size = hidden_size
        self.vocab_size = vocab.size()
        self.vocab = vocab
        self.embedding = embedding

        # GRU Weights
        self.W_z = nn.Linear(2 * hidden_size, hidden_size)
        self.U_r = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_r = nn.Linear(hidden_size, hidden_size)
        self.W_h = nn.Linear(2 * hidden_size, hidden_size)

        # Word Prediction Weights 
        self.W = nn.Linear(hidden_size + latent_size, hidden_size)

        # Stop Prediction Weights
        self.U = nn.Linear(hidden_size + latent_size, hidden_size)
        self.U_i = nn.Linear(2 * hidden_size, hidden_size)

        # Output Weights
        self.W_o = nn.Linear(hidden_size, self.vocab_size)
        self.U_o = nn.Linear(hidden_size, 1)

        # Loss Functions
        self.pred_loss = nn.CrossEntropyLoss(reduction='sum')
        self.stop_loss = nn.BCEWithLogitsLoss(reduction='sum')

    def aggregate(
        self, 
        hiddens: torch.Tensor, 
        contexts: torch.Tensor, 
        x_tree_vecs: torch.Tensor, 
        mode: Literal['word', 'stop']
    ) -> torch.Tensor:
        """
        Aggregates hidden states and tree vectors to produce output logits.
        
        Args:
            hiddens: Tensor of hidden states.
            contexts: 1D Tensor of indices to select from x_tree_vecs.
            x_tree_vecs: Tensor of tree node vectors.
            mode: Either 'word' for vocabulary prediction or 'stop' for termination prediction.
            
        Returns:
            Output logits (vocab_size for 'word', 1 for 'stop').
        """
        if mode == 'word':
            v_proj, v_out = self.W, self.W_o
        elif mode == 'stop':
            v_proj, v_out = self.U, self.U_o
        else:
            # Using modern f-strings for cleaner error formatting
            raise ValueError(f"Invalid aggregate mode: '{mode}'. Expected 'word' or 'stop'.")

        # Modern PyTorch idiomatic indexing: 
        # Advanced indexing (x[idx]) is cleaner and often faster than index_select in newer PyTorch versions.
        tree_contexts = x_tree_vecs[contexts]
        
        # Concatenate along the last dimension
        input_vec = torch.cat([hiddens, tree_contexts], dim=-1)
        
        # Apply projection and non-linearity
        output_vec = F.relu(v_proj(input_vec))
        
        # Return final mapped logits
        return v_out(output_vec)
