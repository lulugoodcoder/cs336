import torch
import torch.nn as nn

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        # Calculate d_ff as 8/3 * d_model, rounded to nearest multiple of 64
        if d_ff is None:
            d_ff = int(d_model * 8 / 3)
            d_ff = ((d_ff + 63) // 64) * 64  # Round up to nearest multiple of 64
    
        self.d_ff = d_ff
        
        # Initialize weights: W1, W3 ∈ R^(d_ff × d_model), W2 ∈ R^(d_model × d_ff)
        weight1 = torch.empty(self.d_ff, d_model, dtype=dtype, device=device)
        weight2 = torch.empty(d_model, self.d_ff, dtype=dtype, device=device)
        weight3 = torch.empty(self.d_ff, d_model, dtype=dtype, device=device)

        nn.init.trunc_normal_(weight1, mean=0.0, std=1.0, a=-3.0, b=3.0)
        nn.init.trunc_normal_(weight2, mean=0.0, std=1.0, a=-3.0, b=3.0)
        nn.init.trunc_normal_(weight3, mean=0.0, std=1.0, a=-3.0, b=3.0)

        self.weight1 = nn.Parameter(weight1)
        self.weight2 = nn.Parameter(weight2)
        self.weight3 = nn.Parameter(weight3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        linear1 = x @ self.weight1.T
        
        # SiLU(W1x) = (W1x) * sigmoid(W1x)
        silu = linear1 * torch.sigmoid(linear1)
        
        # W3x: (... d_model) @ (d_model, d_ff) -> (... d_ff)
        linear3 = x @ self.weight3.T
        
        # Element-wise product: (... d_ff) ⊙ (... d_ff) -> (... d_ff)
        gated = silu * linear3
        
        # W2 * gated: (... d_ff) @ (d_ff, d_model) -> (... d_model)
        return gated @ self.weight2.T
