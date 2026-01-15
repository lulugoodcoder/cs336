import torch
import torch.nn as nn
from cs336_basics.Softmax import Softmax

class ScaledDotProduct(nn.Module):
    def __init__(self):
        super().__init__()

    # key (batch_size, ..., seq_len, d_k)
    # query (batch_size, ..., seq_len, d_v)
    # output (batch_size,...,d_v)
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:       
        dk = k.shape[-1]  # Changed self.k to k
        # Compute Q @ K^T / sqrt(dk), then softmax
        scores = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(dk, dtype=q.dtype))
    
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == False, float('-inf'))

        softmax = Softmax()

        # Apply softmax
        attention_weights = softmax.forward(scores, dim=-1)  # Fixed: softmax after scaling
    
        # Handle NaN from all -inf rows
        attention_weights = torch.nan_to_num(attention_weights, nan=0.0)
    
        # Multiply by values
        return attention_weights @ v