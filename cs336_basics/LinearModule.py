import math
import torch.nn as nn
import torch
class LinearModule(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        W = torch.empty(in_features, out_features, dtype=dtype, device=device)
        std =  math.sqrt(2 / (in_features + out_features))
        nn.init.trunc_normal_(W, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W
