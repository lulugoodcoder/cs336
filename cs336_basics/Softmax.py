import torch
import torch.nn as nn
class Softmax(nn.Module):
        def forward(self, x: torch.Tensor, dim: int) -> torch.Tensor:
            # Step 1: Find max along the dimension (for numerical stability)
            # keepdim=True preserves the dimension for broadcasting
            max_vals = torch.max(x, dim=dim, keepdim=True).values  # Shape: (..., 1, ...)
            
            # Step 2: Subtract max from all elements (numerical stability trick)
            x_shifted = x - max_vals  # Broadcasting: (..., n, ...) - (..., 1, ...)
            
            # Step 3: Compute exp
            exp_vals = torch.exp(x_shifted)  # Shape: same as x
            
            # Step 4: Sum along the dimension
            sum_exp = torch.sum(exp_vals, dim=dim, keepdim=True)  # Shape: (..., 1, ...)
            
            # Step 5: Normalize
            softmax_vals = exp_vals / sum_exp  # Broadcasting: (..., n, ...) / (..., 1, ...)
            
            return softmax_vals

