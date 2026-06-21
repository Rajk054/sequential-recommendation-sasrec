from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from recsys.data import download_movielens, load_movielens, save_mappings, synthetic_data
from recsys.metrics import relative_gain
from recsys.models import MatrixFactorization, SASRec, TwoTower
from recsys.training import evaluate_model, save_checkpoint, seed_everything, train_pointwise, train_sasrec


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and compare sequential recommendation models")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--synthetic", action="store_true", help="Run a fast integration check without downloading MovieLens")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--max-len", type=int, default=100)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    data = synthetic_data() if args.synthetic else load_movielens(download_movielens(args.data_dir))
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    save_mappings(data, args.artifacts_dir / "dataset.json")

    specs = {
        "mf": (MatrixFactorization(data.num_users, data.num_items, args.dim), {"num_users": data.num_users, "num_items": data.num_items, "dim": args.dim}),
        "two_tower": (TwoTower(data.num_users, data.num_items, args.dim), {"num_users": data.num_users, "num_items": data.num_items, "dim": args.dim}),
        "sasrec": (SASRec(data.num_items, args.max_len, args.dim), {"num_items": data.num_items, "max_len": args.max_len, "dim": args.dim}),
    }
    results: dict[str, dict] = {}
    for name, (model, config) in specs.items():
        started = time.perf_counter()
        if name == "sasrec":
            losses = train_sasrec(model, data, args.epochs, args.batch_size, args.lr, device)
        else:
            losses = train_pointwise(model, data, args.epochs, args.batch_size, args.lr, args.negatives, device)
        validation = evaluate_model(model, name, data, "validation", args.max_len, device)
        test = evaluate_model(model, name, data, "test", args.max_len, device)
        save_checkpoint(model, name, args.artifacts_dir / f"{name}.pt", config)
        results[name] = {"train_seconds": time.perf_counter() - started, "final_loss": losses[-1], "validation": validation, "test": test}
        print(f"{name}: NDCG@10={test['NDCG@10']:.4f}, Recall@10={test['Recall@10']:.4f}")

    results["comparison"] = {}
    for baseline in ("mf", "two_tower"):
        results["comparison"][f"sasrec_vs_{baseline}"] = {
            metric: relative_gain(results["sasrec"]["test"][metric], results[baseline]["test"][metric])
            for metric in ("NDCG@10", "Recall@10")
        }
    results["run"] = {"seed": args.seed, "epochs": args.epochs, "device": str(device), "python": platform.python_version(), "torch": torch.__version__, "synthetic": args.synthetic}
    (args.artifacts_dir / "evaluation.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote {args.artifacts_dir / 'evaluation.json'}")


if __name__ == "__main__":
    main()
