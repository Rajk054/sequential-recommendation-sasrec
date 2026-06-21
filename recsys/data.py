from __future__ import annotations

import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"


@dataclass
class SequenceData:
    train: dict[int, list[int]]
    validation: dict[int, int]
    test: dict[int, int]
    user_to_index: dict[int, int]
    item_to_index: dict[int, int]

    @property
    def num_users(self) -> int:
        return len(self.user_to_index)

    @property
    def num_items(self) -> int:
        return len(self.item_to_index)


def download_movielens(data_dir: Path) -> Path:
    """Download and extract MovieLens 1M if ratings.dat is not present."""
    ratings = data_dir / "ml-1m" / "ratings.dat"
    if ratings.exists():
        return ratings
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "ml-1m.zip"
    if not archive.exists():
        urllib.request.urlretrieve(ML1M_URL, archive)
    with zipfile.ZipFile(archive) as zipped:
        zipped.extractall(data_dir)
    return ratings


def load_movielens(path: Path, min_interactions: int = 5) -> SequenceData:
    """Create chronological leave-two-out splits with contiguous one-based IDs."""
    frame = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["user_id", "item_id", "rating", "timestamp"],
        usecols=["user_id", "item_id", "timestamp"],
    )
    counts = frame.groupby("user_id").size()
    frame = frame[frame.user_id.isin(counts[counts >= min_interactions].index)]
    frame = frame.sort_values(["user_id", "timestamp"], kind="stable")

    users = sorted(frame.user_id.unique().tolist())
    items = sorted(frame.item_id.unique().tolist())
    user_map = {raw: idx + 1 for idx, raw in enumerate(users)}
    item_map = {raw: idx + 1 for idx, raw in enumerate(items)}
    frame["user"] = frame.user_id.map(user_map)
    frame["item"] = frame.item_id.map(item_map)

    train: dict[int, list[int]] = {}
    validation: dict[int, int] = {}
    test: dict[int, int] = {}
    for user, group in frame.groupby("user", sort=False):
        sequence = group.item.astype(int).tolist()
        train[int(user)] = sequence[:-2]
        validation[int(user)] = sequence[-2]
        test[int(user)] = sequence[-1]
    return SequenceData(train, validation, test, user_map, item_map)


def synthetic_data(num_users: int = 32, num_items: int = 80, seed: int = 7) -> SequenceData:
    """Small deterministic dataset for tests and pipeline smoke runs."""
    rng = np.random.default_rng(seed)
    train, validation, test = {}, {}, {}
    for user in range(1, num_users + 1):
        cluster = (user - 1) % 4
        pool = np.arange(cluster * 20 + 1, cluster * 20 + 21)
        seq = rng.choice(pool, size=12, replace=False).astype(int).tolist()
        train[user], validation[user], test[user] = seq[:-2], seq[-2], seq[-1]
    return SequenceData(
        train,
        validation,
        test,
        {i: i for i in range(1, num_users + 1)},
        {i: i for i in range(1, num_items + 1)},
    )


def save_mappings(data: SequenceData, path: Path) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "user_to_index": {str(k): v for k, v in data.user_to_index.items()},
        "item_to_index": {str(k): v for k, v in data.item_to_index.items()},
        "train": {str(k): v for k, v in data.train.items()},
        "validation": {str(k): v for k, v in data.validation.items()},
        "test": {str(k): v for k, v in data.test.items()},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

