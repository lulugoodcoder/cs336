import math
import torch.nn as nn
import torch

class EmbeddingModule(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        
        # Create embedding matrix
        embedding = torch.empty(num_embeddings, embedding_dim, dtype=dtype, device=device)
        
        # Initialize with truncated normal: N(μ=0, σ²=1) truncated at [-3, 3]
        nn.init.trunc_normal_(embedding, mean=0.0, std=1.0, a=-3.0, b=3.0)
        
        self.embedding = nn.Parameter(embedding)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding[token_ids]
