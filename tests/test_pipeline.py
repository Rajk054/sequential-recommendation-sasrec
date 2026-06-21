from pathlib import Path

import pandas as pd
import torch

from recsys.data import load_movielens, synthetic_data
from recsys.models import MatrixFactorization, SASRec, TwoTower
from recsys.sampling import PointwiseDataset, SASRecDataset, padded_history


def test_leave_two_out_is_chronological(tmp_path: Path):
    rows = [(1, item, 5, item * 10) for item in range(1, 6)]
    path = tmp_path / "ratings.dat"
    pd.DataFrame(rows).to_csv(path, sep="::", header=False, index=False)
    data = load_movielens(path)
    assert data.train[1] == [1, 2, 3]
    assert data.validation[1] == 4
    assert data.test[1] == 5


def test_negative_samples_are_unobserved():
    data = synthetic_data(num_users=4)
    dataset = PointwiseDataset(data.train, data.num_items, negatives=2)
    for index in range(len(dataset)):
        user, item, label = dataset[index]
        if label.item() == 0:
            assert item.item() not in set(data.train[user.item()])


def test_model_score_shapes():
    data = synthetic_data(num_users=4)
    mf = MatrixFactorization(data.num_users, data.num_items, dim=8)
    tower = TwoTower(data.num_users, data.num_items, dim=8)
    users = torch.tensor([1, 2])
    assert mf.score_all(users).shape == (2, data.num_items + 1)
    assert tower.score_all(users).shape == (2, data.num_items + 1)

    model = SASRec(data.num_items, max_len=8, dim=8, heads=2, layers=1, dropout=0.0)
    sequences = torch.stack([padded_history(data.train[user], 8) for user in (1, 2)])
    assert model.score_all(sequences).shape == (2, data.num_items + 1)
    inputs, positives, negatives = SASRecDataset(data.train, data.num_items, 8)[0]
    pos_logits, neg_logits = model.training_logits(inputs[None], positives[None], negatives[None])
    assert pos_logits.shape == neg_logits.shape == (1, 8)

