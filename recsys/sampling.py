from __future__ import annotations

import random

import torch
from torch.utils.data import Dataset


def sample_negative(num_items: int, excluded: set[int], rng: random.Random) -> int:
    if len(excluded) >= num_items:
        raise ValueError("Cannot sample a negative when every item is excluded")
    candidate = rng.randint(1, num_items)
    while candidate in excluded:
        candidate = rng.randint(1, num_items)
    return candidate


class PointwiseDataset(Dataset):
    """One positive and fresh uniformly sampled negatives per interaction."""

    def __init__(self, sequences: dict[int, list[int]], num_items: int, negatives: int = 4, seed: int = 7):
        self.positives = [(u, i) for u, seq in sequences.items() for i in seq]
        self.seen = {u: set(seq) for u, seq in sequences.items()}
        self.num_items, self.negatives, self.seed = num_items, negatives, seed

    def __len__(self) -> int:
        return len(self.positives) * (self.negatives + 1)

    def __getitem__(self, index: int):
        positive_index, offset = divmod(index, self.negatives + 1)
        user, item = self.positives[positive_index]
        if offset == 0:
            return torch.tensor(user), torch.tensor(item), torch.tensor(1.0)
        rng = random.Random(self.seed + index)
        negative = sample_negative(self.num_items, self.seen[user], rng)
        return torch.tensor(user), torch.tensor(negative), torch.tensor(0.0)


class SASRecDataset(Dataset):
    """Causal next-item training examples with one sampled negative per position."""

    def __init__(self, sequences: dict[int, list[int]], num_items: int, max_len: int, seed: int = 7):
        self.users = sorted(sequences)
        self.sequences = sequences
        self.num_items, self.max_len, self.seed = num_items, max_len, seed

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, index: int):
        user = self.users[index]
        sequence = self.sequences[user]
        inputs = sequence[-(self.max_len + 1) : -1]
        positives = sequence[-self.max_len :]
        excluded = set(sequence)
        rng = random.Random(self.seed + index)
        negatives = [sample_negative(self.num_items, excluded, rng) for _ in positives]
        pad = self.max_len - len(inputs)
        return (
            torch.tensor([0] * pad + inputs, dtype=torch.long),
            torch.tensor([0] * pad + positives, dtype=torch.long),
            torch.tensor([0] * pad + negatives, dtype=torch.long),
        )


def padded_history(sequence: list[int], max_len: int) -> torch.Tensor:
    values = sequence[-max_len:]
    return torch.tensor([0] * (max_len - len(values)) + values, dtype=torch.long)
