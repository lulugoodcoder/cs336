import torch
from torch.optim.optimizer import Optimizer

class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        """AdamW optimizer."""
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self):
        """Performs a single optimization step."""
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # Initialize state on first step
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)
                
                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']
                
                # Update moments: m ← β1*m + (1-β1)*g, v ← β2*v + (1-β2)*g²
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias-corrected step size
                step_size = group['lr'] * (1 - beta2**t)**0.5 / (1 - beta1**t)
                
                # Update: θ ← θ - step_size * m / (sqrt(v) + ε)
                p.addcdiv_(m, v.sqrt().add_(group['eps']), value=-step_size)
                
                # Weight decay: θ ← θ - lr*λ*θ
                p.mul_(1 - group['lr'] * group['weight_decay'])
        