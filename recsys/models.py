from __future__ import annotations

import math

import torch
from torch import nn


class MatrixFactorization(nn.Module):
    def __init__(self, num_users: int, num_items: int, dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, dim)
        self.item_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.user_bias = nn.Embedding(num_users + 1, 1)
        self.item_bias = nn.Embedding(num_items + 1, 1, padding_idx=0)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.user_embedding.weight, std=0.02)
        nn.init.normal_(self.item_embedding.weight, std=0.02)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.user_embedding(users) * self.item_embedding(items)).sum(-1) + self.user_bias(users).squeeze(-1) + self.item_bias(items).squeeze(-1)

    def score_all(self, users: torch.Tensor) -> torch.Tensor:
        query = self.user_embedding(users)
        return query @ self.item_embedding.weight.T + self.user_bias(users) + self.item_bias.weight.T

    def query_vector(self, users: torch.Tensor) -> torch.Tensor:
        return self.user_embedding(users)


class TwoTower(nn.Module):
    """Non-sequential neural retrieval baseline with independently cacheable towers."""

    def __init__(self, num_users: int, num_items: int, dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users + 1, dim)
        self.item_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.user_tower = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))
        self.item_tower = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def encode_users(self, users: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.user_tower(self.user_embedding(users)), dim=-1)

    def encode_items(self, items: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.item_tower(self.item_embedding(items)), dim=-1)

    def forward(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.encode_users(users) * self.encode_items(items)).sum(-1) * 10.0

    def score_all(self, users: torch.Tensor) -> torch.Tensor:
        items = torch.arange(self.item_embedding.num_embeddings, device=users.device)
        return self.encode_users(users) @ self.encode_items(items).T

    def query_vector(self, users: torch.Tensor) -> torch.Tensor:
        return self.encode_users(users)


class SASRec(nn.Module):
    def __init__(self, num_items: int, max_len: int = 100, dim: int = 64, heads: int = 2, layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_items, self.max_len, self.dim = num_items, max_len, dim
        self.item_embedding = nn.Embedding(num_items + 1, dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, dim)
        block = nn.TransformerEncoderLayer(dim, heads, dim * 4, dropout, batch_first=True, norm_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(block, layers)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def encode(self, sequences: torch.Tensor) -> torch.Tensor:
        positions = torch.arange(sequences.size(1), device=sequences.device)
        hidden = self.item_embedding(sequences) * math.sqrt(self.dim) + self.position_embedding(positions)
        hidden = self.dropout(hidden)
        causal = torch.triu(torch.ones(sequences.size(1), sequences.size(1), device=sequences.device, dtype=torch.bool), diagonal=1)
        hidden = self.encoder(hidden, mask=causal, src_key_padding_mask=sequences.eq(0))
        return self.norm(hidden)

    def training_logits(self, sequences: torch.Tensor, positives: torch.Tensor, negatives: torch.Tensor):
        hidden = self.encode(sequences)
        return (hidden * self.item_embedding(positives)).sum(-1), (hidden * self.item_embedding(negatives)).sum(-1)

    def query_vector(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.encode(sequences)[:, -1]

    def score_all(self, sequences: torch.Tensor) -> torch.Tensor:
        return self.query_vector(sequences) @ self.item_embedding.weight.T
