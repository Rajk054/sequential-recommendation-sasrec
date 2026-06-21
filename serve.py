from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from recsys.models import SASRec
from recsys.retrieval import build_hnsw, item_vectors, query_vector
from recsys.sampling import padded_history
from recsys.training import load_checkpoint


class RecommendRequest(BaseModel):
    item_ids: list[int] = Field(min_length=1, description="Raw MovieLens item IDs in chronological order")
    k: int = Field(default=10, ge=1, le=100)


def create_app(artifacts_dir: Path = Path("artifacts")) -> FastAPI:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_checkpoint(artifacts_dir / "sasrec.pt", device)
    if not isinstance(model, SASRec):
        raise ValueError("The serving checkpoint must be SASRec")
    dataset = json.loads((artifacts_dir / "dataset.json").read_text(encoding="utf-8"))
    raw_to_item = {int(raw): int(index) for raw, index in dataset["item_to_index"].items()}
    item_to_raw = {index: raw for raw, index in raw_to_item.items()}
    vectors = item_vectors(model, device)
    index = build_hnsw(vectors, np.arange(1, model.num_items + 1, dtype="int64"))

    app = FastAPI(title="Sequential Recommendation API", version="1.0.0")

    @app.get("/health")
    def health():
        return {"status": "ok", "model": "sasrec", "catalog_size": model.num_items, "index": "HNSW"}

    @app.post("/recommend")
    def recommend(request: RecommendRequest):
        unknown = [item for item in request.item_ids if item not in raw_to_item]
        if unknown:
            raise HTTPException(422, f"Unknown MovieLens item IDs: {unknown[:5]}")
        history = [raw_to_item[item] for item in request.item_ids]
        sequence = padded_history(history, model.max_len).unsqueeze(0).to(device)
        started = time.perf_counter_ns()
        query = query_vector(model, sequence)
        # Retrieve extra candidates because consumed items must be removed post-ANN.
        search_k = min(model.num_items, request.k + len(set(history)) + 50)
        scores, ids = index.search(query, search_k)
        consumed, recommendations = set(history), []
        for item, score in zip(ids[0], scores[0]):
            if item > 0 and int(item) not in consumed:
                recommendations.append({"item_id": item_to_raw[int(item)], "score": float(score)})
                if len(recommendations) == request.k:
                    break
        latency_ms = (time.perf_counter_ns() - started) / 1e6
        return {"recommendations": recommendations, "retrieval_latency_ms": latency_ms}

    return app


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.artifacts_dir), host=args.host, port=args.port)


if __name__ == "__main__":
    main()

