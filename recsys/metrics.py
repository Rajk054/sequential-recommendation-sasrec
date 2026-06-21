from __future__ import annotations

import math
from collections.abc import Callable

import torch


def ranking_metrics(top_items: torch.Tensor, targets: torch.Tensor, ks: tuple[int, ...] = (5, 10, 20)) -> dict[str, float]:
    """Recall and NDCG for one relevant held-out item per user."""
    result: dict[str, float] = {}
    for k in ks:
        matches = top_items[:, :k].eq(targets[:, None])
        hit = matches.any(dim=1)
        result[f"Recall@{k}"] = hit.float().mean().item()
        positions = matches.float().argmax(dim=1) + 1
        discounts = torch.where(hit, 1.0 / torch.log2(positions.float() + 1), torch.zeros_like(positions, dtype=torch.float))
        result[f"NDCG@{k}"] = discounts.mean().item()
    return result


@torch.no_grad()
def evaluate_full_catalog(
    users: list[int],
    targets: dict[int, int],
    seen: dict[int, set[int]],
    score_batch: Callable[[list[int]], torch.Tensor],
    num_items: int,
    batch_size: int = 256,
    ks: tuple[int, ...] = (5, 10, 20),
) -> dict[str, float]:
    """Rank each target against every item not already observed by that user."""
    sums = {f"Recall@{k}": 0.0 for k in ks} | {f"NDCG@{k}": 0.0 for k in ks}
    count = 0
    max_k = min(max(ks), num_items)
    for start in range(0, len(users), batch_size):
        batch_users = users[start : start + batch_size]
        scores = score_batch(batch_users).detach().cpu()
        scores[:, 0] = -torch.inf
        for row, user in enumerate(batch_users):
            blocked = seen[user] - {targets[user]}
            if blocked:
                scores[row, torch.tensor(list(blocked))] = -torch.inf
        top = scores.topk(max_k, dim=1).indices
        batch_targets = torch.tensor([targets[u] for u in batch_users])
        metrics = ranking_metrics(top, batch_targets, ks)
        for name, value in metrics.items():
            sums[name] += value * len(batch_users)
        count += len(batch_users)
    return {name: value / count for name, value in sums.items()}


def relative_gain(candidate: float, baseline: float) -> float | None:
    return None if math.isclose(baseline, 0.0) else (candidate - baseline) / baseline

