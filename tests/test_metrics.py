import torch

from recsys.metrics import ranking_metrics


def test_ranking_metrics_use_rank_discount():
    top = torch.tensor([[4, 2, 1], [3, 5, 6]])
    targets = torch.tensor([2, 9])
    metrics = ranking_metrics(top, targets, ks=(1, 3))
    assert metrics["Recall@1"] == 0.0
    assert metrics["Recall@3"] == 0.5
    assert abs(metrics["NDCG@3"] - 0.3154649) < 1e-6

