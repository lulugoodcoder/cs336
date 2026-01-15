import torch
import torch.nn as nn
from cs336_basics.RootSquareMeanNorm import RootSquareMeanNorm
from cs336_basics.CausalMultiHeadSelfAttention import CausalMultiHeadSelfAttention
from cs336_basics.SwiGLU import SwiGLU
from cs336_basics.EmbeddingModule import EmbeddingModule
from cs336_basics.TansformerBlock import TansformerBlock
from cs336_basics.RootSquareMeanNorm import RootSquareMeanNorm
from cs336_basics.LinearModule import LinearModule
from cs336_basics.Softmax import Softmax

class TansformerLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, d_ff: int,
                 num_layers: int,  context_length: int, rope_theta: float,
                 device=None, dtype=None):
        super().__init__()
        self.embedding = EmbeddingModule(vocab_size, d_model)
        self.transformer_blocks = nn.ModuleList([
            TansformerBlock(d_model, num_heads, d_ff, rope_theta, context_length, device, dtype)
            for _ in range(num_layers)
        ])
        self.norm = RootSquareMeanNorm(d_model)
        self.linear = LinearModule(d_model, vocab_size)
        self.softmax = Softmax()

    def forward(self, x: torch.Tensor):
        batch_size, seq_len = x.shape
        token_positions = torch.arange(seq_len, device=x.device)
        # 扩展到 batch 维度: (batch_size, seq_len)
        token_positions = token_positions.unsqueeze(0).expand(batch_size, -1)

        hidden  = self.embedding(x)
        for tansform in self.transformer_blocks:
            hidden  = tansform(hidden, token_positions)
        hidden = self.norm(hidden)
        logits = self.linear(hidden)
        return logits


