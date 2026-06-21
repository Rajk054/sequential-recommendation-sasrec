from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:+.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Render measured model comparisons as Markdown")
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output", type=Path, default=Path("RESULTS.md"))
    args = parser.parse_args()
    results = json.loads((args.artifacts_dir / "evaluation.json").read_text(encoding="utf-8"))
    retrieval_path = args.artifacts_dir / "retrieval_benchmark.json"
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8")) if retrieval_path.exists() else None
    lines = ["# Measured results", "", "All ranking metrics use full-catalog leave-one-out test evaluation.", "", "| Model | NDCG@10 | Recall@10 | Train seconds |", "|---|---:|---:|---:|"]
    for name in ("mf", "two_tower", "sasrec"):
        row = results[name]
        lines.append(f"| {name} | {row['test']['NDCG@10']:.4f} | {row['test']['Recall@10']:.4f} | {row['train_seconds']:.1f} |")
    lines += ["", "## Complexity trade-off", ""]
    for baseline in ("mf", "two_tower"):
        gains = results["comparison"][f"sasrec_vs_{baseline}"]
        lines.append(f"- SASRec vs {baseline}: {pct(gains['NDCG@10'])} NDCG@10 and {pct(gains['Recall@10'])} Recall@10.")
    lines.append("- Treat the Transformer as justified only if its ranking gain is material for the product and outweighs its additional training and query-encoding cost.")
    if retrieval:
        lines += ["", "## Retrieval benchmark", "", f"HNSW median/p95 retrieval latency: {retrieval['hnsw']['median_ms']:.3f}/{retrieval['hnsw']['p95_ms']:.3f} ms; exact median/p95: {retrieval['exact']['median_ms']:.3f}/{retrieval['exact']['p95_ms']:.3f} ms; ANN recall@{retrieval['k']}: {retrieval['ann_recall_at_k']:.3f}."]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
