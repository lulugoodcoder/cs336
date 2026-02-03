import torch
import torch.nn as nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None, dtype=None):
        if dtype is None:
            dtype = torch.float32
        super().__init__()
        
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        
        # k ∈ {1, 2, ..., d_k/2} for 1-indexed formula
        k = torch.arange(1, self.d_k // 2 + 1, dtype=dtype, device=device)  # Shape: (d_k/2,)
        
        # Compute frequencies: Θ^{-(2k-2)/d} = 1 / Θ^{(2k-2)/d}
        # Note: division is INSIDE the exponent
        freq = self.theta ** (-(2 * k - 2) / self.d_k)  # Shape: (d_k/2,)
        
        # Create position indices: [0, 1, 2, ..., max_seq_len-1]
        positions = torch.arange(self.max_seq_len, dtype=dtype, device=device)  # Shape: (max_seq_len,)
        
        # Compute angles: θ_{i,k} = i * Θ^{-(2k-2)/d}
        # Broadcasting: (max_seq_len, 1) * (1, d_k/2) → (max_seq_len, d_k/2)
        angles = positions.unsqueeze(1) * freq.unsqueeze(0)  # Shape: (max_seq_len, d_k/2)
        
        # Precompute cos and sin
        cos = torch.cos(angles)  # Shape: (max_seq_len, d_k/2)
        sin = torch.sin(angles)  # Shape: (max_seq_len, d_k/2)
        
        # Register as buffers (not parameters, won't be trained)
        self.register_buffer('cos_cached', cos, persistent=False)
        self.register_buffer('sin_cached', sin, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Shape (..., seq_len, d_k)
            token_positions: Shape (..., seq_len)
        Returns:
            rotated: Shape (..., seq_len, d_k)
        """
        # Get cos and sin for the given token positions
        cos = self.cos_cached[token_positions]  # Shape: (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]  # Shape: (..., seq_len, d_k/2)
        
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        # Split x into even and odd indices (pairs)
        x_even = x[..., 0::2]  # Shape: (..., seq_len, d_k/2) - indices [0, 2, 4, ...]
        x_odd = x[..., 1::2]   # Shape: (..., seq_len, d_k/2) - indices [1, 3, 5, ...]
        
        # Apply 2D rotation to each pair
        # [cos -sin] [x_even]   [x_even*cos - x_odd*sin]
        # [sin  cos] [x_odd ] = [x_even*sin + x_odd*cos]
        x1 = x_even * cos - x_odd * sin  # Shape: (..., seq_len, d_k/2)
        x2 = x_even * sin + x_odd * cos  # Shape: (..., seq_len, d_k/2)
        
        # Stack and interleave back
        rotated = torch.stack([x1, x2], dim=-1)  # Shape: (..., seq_len, d_k/2, 2)
        rotated = rotated.flatten(start_dim=-2)  # Shape: (..., seq_len, d_k)
        
        return rotated

    

    