from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from recsys.retrieval import build_hnsw, item_vectors, query_vector
from recsys.sampling import padded_history
from recsys.training import load_checkpoint


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values), q))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare exact and HNSW retrieval latency")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--queries", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.artifacts_dir / "sasrec.pt", device)
    data = json.loads((args.artifacts_dir / "dataset.json").read_text(encoding="utf-8"))
    histories = list(data["train"].values())
    vectors = item_vectors(model, device)
    ids = np.arange(1, vectors.shape[0] + 1, dtype="int64")
    index = build_hnsw(vectors, ids)
    rng = np.random.default_rng(7)
    chosen = rng.choice(len(histories), size=args.queries, replace=True)
    exact_ms, ann_ms, overlaps = [], [], []

    for selected in chosen:
        sequence = padded_history(histories[int(selected)], model.max_len).unsqueeze(0).to(device)
        query = query_vector(model, sequence)
        start = time.perf_counter_ns()
        exact = np.argpartition(query @ vectors.T, -args.k, axis=1)[:, -args.k:]
        exact_ms.append((time.perf_counter_ns() - start) / 1e6)
        start = time.perf_counter_ns()
        _, approximate = index.search(query, args.k)
        ann_ms.append((time.perf_counter_ns() - start) / 1e6)
        exact_ids = {int(i + 1) for i in exact[0]}
        overlaps.append(len(exact_ids & set(approximate[0].tolist())) / args.k)

    report = {
        "queries": args.queries,
        "k": args.k,
        "exact": {"median_ms": statistics.median(exact_ms), "p95_ms": percentile(exact_ms, 95)},
        "hnsw": {"median_ms": statistics.median(ann_ms), "p95_ms": percentile(ann_ms, 95)},
        "ann_recall_at_k": statistics.mean(overlaps),
    }
    output = args.artifacts_dir / "retrieval_benchmark.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

