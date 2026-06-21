from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .data import SequenceData
from .metrics import evaluate_full_catalog
from .models import MatrixFactorization, SASRec, TwoTower
from .sampling import SASRecDataset, padded_history


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_batch_negatives(
    users: torch.Tensor,
    num_items: int,
    negatives: int,
    seen_mask: torch.Tensor,
) -> torch.Tensor:
    """Sample unobserved items for a user batch directly on its device."""
    sampled = torch.randint(
        1, num_items + 1, (users.size(0), negatives), device=users.device
    )
    invalid = seen_mask[users[:, None], sampled]
    while invalid.any():
        sampled[invalid] = torch.randint(
            1, num_items + 1, (int(invalid.sum().item()),), device=users.device
        )
        invalid = seen_mask[users[:, None], sampled]
    return sampled


def train_pointwise(model: nn.Module, data: SequenceData, epochs: int, batch_size: int, lr: float, negatives: int, device: torch.device) -> list[float]:
    positive_users = torch.tensor(
        [user for user, sequence in data.train.items() for _ in sequence],
        dtype=torch.long,
    )
    positive_items = torch.tensor(
        [item for sequence in data.train.values() for item in sequence],
        dtype=torch.long,
    )
    dataset = TensorDataset(positive_users, positive_items)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.type == "cuda",
        persistent_workers=True,
    )
    seen_mask = torch.zeros(
        (data.num_users + 1, data.num_items + 1), dtype=torch.bool, device=device
    )
    seen_mask[
        positive_users.to(device), positive_items.to(device)
    ] = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    model.to(device)
    for _ in range(epochs):
        model.train()
        total = 0.0
        examples = 0
        for users, positives in loader:
            users = users.to(device, non_blocking=True)
            positives = positives.to(device, non_blocking=True)
            sampled = sample_batch_negatives(
                users, data.num_items, negatives, seen_mask
            )
            items = torch.cat((positives[:, None], sampled), dim=1)
            batch_users = users[:, None].expand_as(items)
            labels = torch.zeros_like(items, dtype=torch.float)
            labels[:, 0] = 1.0
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(
                model(batch_users.reshape(-1), items.reshape(-1)),
                labels.reshape(-1),
            )
            loss.backward()
            optimizer.step()
            batch_examples = items.numel()
            total += loss.item() * batch_examples
            examples += batch_examples
        losses.append(total / examples)
    return losses


def train_sasrec(model: SASRec, data: SequenceData, epochs: int, batch_size: int, lr: float, device: torch.device) -> list[float]:
    dataset = SASRecDataset(data.train, data.num_items, model.max_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type == "cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    losses = []
    model.to(device)
    for _ in range(epochs):
        model.train()
        total, tokens = 0.0, 0
        for sequences, positives, negatives in loader:
            sequences = sequences.to(device, non_blocking=True)
            positives = positives.to(device, non_blocking=True)
            negatives = negatives.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pos_logits, neg_logits = model.training_logits(sequences, positives, negatives)
            mask = positives.ne(0)
            loss = (criterion(pos_logits, torch.ones_like(pos_logits)) + criterion(neg_logits, torch.zeros_like(neg_logits)))[mask].mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * mask.sum().item()
            tokens += mask.sum().item()
        losses.append(total / max(tokens, 1))
    return losses


def evaluate_model(model: nn.Module, name: str, data: SequenceData, split: str, max_len: int, device: torch.device, batch_size: int = 256) -> dict[str, float]:
    model.eval()
    users = sorted(data.train)
    if split == "validation":
        targets = data.validation
        histories = data.train
    elif split == "test":
        targets = data.test
        histories = {u: data.train[u] + [data.validation[u]] for u in users}
    else:
        raise ValueError("split must be validation or test")
    seen = {u: set(histories[u]) for u in users}

    def score_batch(batch_users: list[int]) -> torch.Tensor:
        if name == "sasrec":
            sequences = torch.stack([padded_history(histories[u], max_len) for u in batch_users]).to(device)
            return model.score_all(sequences)
        return model.score_all(torch.tensor(batch_users, device=device))

    return evaluate_full_catalog(users, targets, seen, score_batch, data.num_items, batch_size=batch_size)


def save_checkpoint(model: nn.Module, name: str, path: Path, config: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": name, "config": config, "state_dict": model.state_dict()}, path)


def load_checkpoint(path: Path, device: torch.device = torch.device("cpu")) -> nn.Module:
    payload = torch.load(path, map_location=device, weights_only=False)
    name, config = payload["model"], payload["config"]
    if name == "mf":
        model = MatrixFactorization(**config)
    elif name == "two_tower":
        model = TwoTower(**config)
    elif name == "sasrec":
        model = SASRec(**config)
    else:
        raise ValueError(f"Unknown model type: {name}")
    model.load_state_dict(payload["state_dict"])
    return model.to(device).eval()
