import math

import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.embedding_dim = embedding_dim

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key = nn.Linear(embedding_dim,embedding_dim)
        self.value = nn.Linear(embedding_dim,embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch_size, seq_len, embedding_dim]

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        scores = Q @ K.transpose(-2, -1)
        scores = scores / math.sqrt(self.embedding_dim)

        seq_len = x.size(1)

        mask = torch.triu(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )

        scores = scores.masked_fill(mask, float("-inf"))

        weights = torch.softmax(scores, dim=-1)

        output = weights @ V

        return output


class MiniTransformer(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 32):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.attention = SelfAttention(embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]

        x = self.embedding(x) 
        x = self.attention(x)
        x = x[:, -1, :]
        logits = self.output(x)

        return logits