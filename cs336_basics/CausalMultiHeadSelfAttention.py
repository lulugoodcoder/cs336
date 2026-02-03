import torch.nn as nn
import torch
import math
from cs336_basics.RotaryPositionalEmbedding import RotaryPositionalEmbedding
from cs336_basics.ScaledDotProduct import ScaledDotProduct


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, theta: float = 10000.0, 
                 max_seq_len: int = 2048, use_rope: bool = True, device=None, dtype=None):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # dk = dv = d_model / h
        self.d_v = d_model // num_heads
        self.use_rope = use_rope
        
        # Full projection matrices: W_Q, W_K, W_V ∈ R^(h*dk × d_model)
        # W_O ∈ R^(d_model × h*dv)
        self.W_Q = nn.Parameter(torch.empty(num_heads * self.d_k, d_model, dtype=dtype, device=device))
        self.W_K = nn.Parameter(torch.empty(num_heads * self.d_k, d_model, dtype=dtype, device=device))
        self.W_V = nn.Parameter(torch.empty(num_heads * self.d_v, d_model, dtype=dtype, device=device))
        self.W_O = nn.Parameter(torch.empty(d_model, num_heads * self.d_v, dtype=dtype, device=device))
        
        # Initialize weights
        nn.init.trunc_normal_(self.W_Q, mean=0.0, std=1.0, a=-3.0, b=3.0)
        nn.init.trunc_normal_(self.W_K, mean=0.0, std=1.0, a=-3.0, b=3.0)
        nn.init.trunc_normal_(self.W_V, mean=0.0, std=1.0, a=-3.0, b=3.0)
        nn.init.trunc_normal_(self.W_O, mean=0.0, std=1.0, a=-3.0, b=3.0)
        

        # RoPE embedding
        if self.use_rope:
            self.rope = RotaryPositionalEmbedding(theta=theta, d_k=self.d_k, max_seq_len=max_seq_len)
        self.scaled_dot_product = ScaledDotProduct()

    # def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
    #     # x: (batch_size, seq_len, d_model)
    #     batch_size, seq_len, _ = x.shape
        
    #     # Project to Q, K, V: (batch_size, seq_len, num_heads * d_k)
    #     Q = x @ self.W_Q.T  # (batch, seq_len, num_heads * d_k)
    #     K = x @ self.W_K.T
    #     V = x @ self.W_V.T
        
    #     # Reshape to separate heads: (batch_size, seq_len, num_heads, d_k)
    #     Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k)
    #     K = K.view(batch_size, seq_len, self.num_heads, self.d_k)
    #     V = V.view(batch_size, seq_len, self.num_heads, self.d_v)
        
    #     # Transpose to (batch_size, num_heads, seq_len, d_k) for attention
    #     Q = Q.transpose(1, 2)
    #     K = K.transpose(1, 2)
    #     V = V.transpose(1, 2)
        
    #     # Apply RoPE to Q and K (not V)
    #     if self.use_rope:
    #         Q = self.rope(Q, token_positions)
    #         K = self.rope(K, token_positions)
        
    #     # Create causal mask: (seq_len, seq_len)
    #     # Mask should be True where we want to MASK (block attention)
    #     causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=0)
        
    #     # Scaled dot-product attention per head
    #     # attn_output: (batch_size, num_heads, seq_len, d_v)
    #     attn_output = self.scaled_dot_product(Q, K, V, causal_mask)
        
        
    #     # ============================================================
    #     # FINAL PROJECTION: W_O × Concat(head₁, ..., head_h)
    #     # ============================================================

    #     # After scaled dot-product attention:
    #     # attn_output.shape = (batch_size, num_heads, seq_len, d_v)
    #     #                   = (32, 12, 5, 64)
    #     #
    #     # We need to:
    #     # 1. Rearrange so seq_len is in position 1
    #     # 2. Concatenate all heads together
    #     # 3. Project with W_O

    #     # ------------------------------------------------------------
    #     # Step 1: Transpose - swap heads and seq_len dimensions
    #     # ------------------------------------------------------------
    #     # From: (batch, heads, seq, d_v) = (32, 12, 5, 64)
    #     # To:   (batch, seq, heads, d_v) = (32, 5, 12, 64)
    #     attn_output = attn_output.transpose(1, 2)

    #     # Why? We want final output as (batch, seq, d_model)
    #     #      So seq_len must be in dimension 1

    #     # ------------------------------------------------------------
    #     # Step 2: Contiguous - fix memory layout
    #     # ------------------------------------------------------------
    #     # After transpose, memory is not continuous
    #     # view() requires continuous memory, so we fix it
    #     attn_output = attn_output.contiguous()

    #     # ------------------------------------------------------------
    #     # Step 3: View/Reshape - concatenate all heads
    #     # ------------------------------------------------------------
    #     # From: (batch, seq, heads, d_v) = (32, 5, 12, 64)
    #     # To:   (batch, seq, heads * d_v) = (32, 5, 768)
    #     # Transpose back and concatenate heads: (batch_size, seq_len, num_heads * d_v)
    #     attn_output = attn_output.view(batch_size, seq_len, -1)
        
    #     output = attn_output @ self.W_O.T

    #     return output

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor = None):
        # Handle arbitrary batch dimensions
        *batch_dims, seq_len, _ = x.shape  # <-- CHANGED
        
        # Project to Q, K, V
        Q = x @ self.W_Q.T
        K = x @ self.W_K.T
        V = x @ self.W_V.T
        
        # Reshape to separate heads
        Q = Q.view(*batch_dims, seq_len, self.num_heads, self.d_k)  # <-- CHANGED
        K = K.view(*batch_dims, seq_len, self.num_heads, self.d_k)
        V = V.view(*batch_dims, seq_len, self.num_heads, self.d_v)
        
        # Transpose to (..., num_heads, seq_len, d_k)
        Q = Q.transpose(-3, -2)  # <-- CHANGED
        K = K.transpose(-3, -2)
        V = V.transpose(-3, -2)
        
        # Apply RoPE
        if self.use_rope:
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)
        
        # Causal mask
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(-2, -1) / (self.d_k ** 0.5)
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        attn_output = attn_weights @ V
        
        # Transpose: (batch, num_heads, seq_len, d_v) -> (batch, seq_len, num_heads, d_v)
        attn_output = attn_output.transpose(1, 2)
        
        # Concatenate heads: (batch, seq_len, num_heads * d_v)
        batch_size = attn_output.shape[0]
        attn_output = attn_output.contiguous().view(batch_size, seq_len, self.num_heads * self.d_v)

        # Final projection
        output = attn_output @ self.W_O.T

        return output

        
