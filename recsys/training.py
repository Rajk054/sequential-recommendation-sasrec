from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import SequenceData
from .metrics import evaluate_full_catalog
from .models import MatrixFactorization, SASRec, TwoTower
from .sampling import PointwiseDataset, SASRecDataset, padded_history


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_pointwise(model: nn.Module, data: SequenceData, epochs: int, batch_size: int, lr: float, negatives: int, device: torch.device) -> list[float]:
    dataset = PointwiseDataset(data.train, data.num_items, negatives=negatives)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    losses = []
    model.to(device)
    for _ in range(epochs):
        model.train()
        total = 0.0
        for users, items, labels in loader:
            users, items, labels = users.to(device), items.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(users, items), labels)
            loss.backward()
            optimizer.step()
            total += loss.item() * users.size(0)
        losses.append(total / len(dataset))
    return losses


def train_sasrec(model: SASRec, data: SequenceData, epochs: int, batch_size: int, lr: float, device: torch.device) -> list[float]:
    dataset = SASRecDataset(data.train, data.num_items, model.max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    losses = []
    model.to(device)
    for _ in range(epochs):
        model.train()
        total, tokens = 0.0, 0
        for sequences, positives, negatives in loader:
            sequences, positives, negatives = sequences.to(device), positives.to(device), negatives.to(device)
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

