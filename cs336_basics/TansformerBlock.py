import torch
import torch.nn as nn
from cs336_basics.RootSquareMeanNorm import RootSquareMeanNorm
from cs336_basics.CausalMultiHeadSelfAttention import CausalMultiHeadSelfAttention
from cs336_basics.SwiGLU import SwiGLU

class TansformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, 
                 theta: float = 10000.0, max_seq_len: int = 2048,
                 device=None, dtype=None):
        
        super().__init__()
        
        # Create all modules in __init__ (so weights are learned!)
        self.norm1 = RootSquareMeanNorm(d_model, device=device, dtype=dtype)
        self.attention = CausalMultiHeadSelfAttention(
            d_model, num_heads, theta=theta, max_seq_len=max_seq_len,
            device=device, dtype=dtype
        )
        self.norm2 = RootSquareMeanNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None) -> torch.Tensor:
        # Sublayer 1: y = x + Attention(Norm(x))
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, token_positions)
        x = x + attn_output  # Residual connection
        
        # Sublayer 2: y = x + FFN(Norm(x))
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        x = x + ffn_output  # Residual connection
        
        return x


