from __future__ import annotations

import numpy as np
import torch

from .models import MatrixFactorization, SASRec, TwoTower


def item_vectors(model: torch.nn.Module, device: torch.device) -> np.ndarray:
    ids = torch.arange(1, model.item_embedding.num_embeddings, device=device)
    with torch.no_grad():
        if isinstance(model, TwoTower):
            vectors = model.encode_items(ids)
        else:
            vectors = model.item_embedding(ids)
    return vectors.detach().cpu().numpy().astype("float32")


def build_hnsw(vectors: np.ndarray, item_ids: np.ndarray, connections: int = 32, ef_search: int = 64):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Install faiss-cpu to build the ANN index") from exc
    base = faiss.IndexHNSWFlat(vectors.shape[1], connections, faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efSearch = ef_search
    index = faiss.IndexIDMap(base)
    index.add_with_ids(vectors, item_ids.astype("int64"))
    return index


def query_vector(model: torch.nn.Module, users_or_sequences: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        vector = model.query_vector(users_or_sequences)
    return vector.detach().cpu().numpy().astype("float32")

