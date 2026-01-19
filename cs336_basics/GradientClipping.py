from typing import Iterable

import torch


class GradientClipping:
    def __init__(self):
        self.eps = 1e-6
    
    def gradient_clipping(self, parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
        total_norm_sq = 0.0
        for p in parameters:
            if p.grad is not None:
                total_norm_sq += torch.sum(p.grad ** 2)
        
        l2_norm = torch.sqrt(total_norm_sq + self.eps)
        
        # Only clip if norm exceeds maximum
        if l2_norm > max_l2_norm:
            clip_coef = max_l2_norm / (l2_norm + self.eps)
            
            # Scale all gradients in-place
            for p in parameters:
                if p.grad is not None:
                    p.grad.mul_(clip_coef)