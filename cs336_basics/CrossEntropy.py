import torch
import torch.nn as nn
# ℓᵢ = -log(softmax(oᵢ)[xᵢ₊₁])

#    = -log(exp(oᵢ[xᵢ₊₁]) / Σₐ exp(oᵢ[a]))
   
#    = -[oᵢ[xᵢ₊₁] - log(Σₐ exp(oᵢ[a]))]
   
#    = log(Σₐ exp(oᵢ[a])) - oᵢ[xᵢ₊₁]
# For numerical stability with the max trick:
# Let M = max(oᵢ)

# ℓᵢ = log(Σₐ exp(oᵢ[a] - M)) + M - oᵢ[xᵢ₊₁]

    
class CrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logits, targets):
        """
        Compute cross-entropy loss with numerical stability.

        Args:
            logits: (..., vocab_size) - predicted logits
            targets: (...) - target token indices

        Returns:
            scalar - average cross-entropy loss across batch
        """
        # Subtract max for numerical stability
        max_logits = logits.max(dim=-1, keepdim=True).values
        shifted_logits = logits - max_logits

        # Compute log(sum(exp(logits)))
        log_sum_exp = torch.log(torch.sum(torch.exp(shifted_logits), dim=-1))

        # Get logits for target tokens
        # Use gather to select logits at target indices
        target_logits = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)

        # Cross entropy: log_sum_exp + max - target_logit
        loss = log_sum_exp + max_logits.squeeze(-1) - target_logits

        # Return mean across all batch dimensions
        return loss.mean()
        