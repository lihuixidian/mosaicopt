"""Synthetic data generator for the MosaicOpt prototype.

The generator builds a single base relation ``memory`` with the
following schema:

    id            : int                 (primary key)
    subsystem     : string              (5 categories)
    ts            : int                 (epoch seconds, last 90 days)
    text          : string              (BM25 corpus, ~10 words per row)
    emb           : list<float32>[768]  (primary embedding)
    emb_alt       : list<float32>[768]  (secondary embedding)
    graph_node    : int                 (node id in the derivation graph)

The distributions are chosen to mirror a small agentic-memory store:
* subsys uniform over 5 categories ("fault-tolerance", "perf",
  "frontend", "tooling", "docs");
* timestamps uniformly over the last 90 days;
* embeddings drawn from a Gaussian mixture so that vector queries
  have a meaningful selectivity;
* text is a deterministic LCG-generated bag of words (no semantic
  content) - this keeps the experiment reproducible.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from mosaicopt.physical.store import TableStore


# Vocabulary for the synthetic text corpus. We mix a few technical
# terms so that BM25 has structure (different docs share some tokens).
_VOCAB = [
    "index", "optimizer", "query", "transaction", "memory",
    "embedding", "vector", "graph", "rewrite", "cost",
    "recall", "join", "predicate", "plan", "rule",
    "scan", "hnsw", "ivf", "selectivity", "cursor",
    "subsystem", "trace", "checkpoint", "latency", "throughput",
    "snapshot", "materialization", "fusion", "operator", "rewrite",
]

_SUBSYSTEMS = ["fault-tolerance", "perf", "frontend", "tooling", "docs"]


def gen(
    n: int = 10_000,
    dim: int = 768,
    seed: int = 42,
    sigma: float = 0.1,
) -> "MemoryDataset":
    """Generate a synthetic memory dataset of `n` rows.

    Parameters
    ----------
    n : int
        Number of rows.
    dim : int
        Embedding dimensionality.
    seed : int
        RNG seed.
    sigma : float
        Standard deviation of the per-row Gaussian noise around the
        cluster center. Smaller sigma = tighter clusters =
        more bimodal selectivity distribution. Larger sigma =
        smoother selectivity gradient. Default 0.1 matches the
        prototype used in E1-E4.

    Returns
    -------
    MemoryDataset
        Contains the numpy arrays + the pyarrow table + the
        TableStore view.
    """
    rng = np.random.default_rng(seed)

    # IDs
    ids = np.arange(n, dtype=np.int64)

    # Subsystem: stratified
    subsys_idx = rng.integers(0, len(_SUBSYSTEMS), size=n)
    subsystems = np.array([_SUBSYSTEMS[i] for i in subsys_idx])

    # Timestamps: last 90 days
    now = int(time.time())
    day = 86_400
    ts = now - rng.integers(0, 90 * day, size=n).astype(np.int64)

    # Embeddings: 5 Gaussian clusters (one per subsystem) + noise.
    # Center each cluster on a random unit vector, sample L2-normalized.
    centers = rng.standard_normal((len(_SUBSYSTEMS), dim)).astype("float32")
    centers /= np.linalg.norm(centers, axis=1, keepdims=True) + 1e-9

    embs = np.empty((n, dim), dtype="float32")
    for i in range(n):
        # 70% from the cluster center, 30% noise
        if rng.random() < 0.7:
            v = centers[subsys_idx[i]] + sigma * rng.standard_normal(dim).astype("float32")
        else:
            v = rng.standard_normal(dim).astype("float32")
        embs[i] = v / (np.linalg.norm(v) + 1e-9)

    # Second embedding: 80% same cluster, 20% different
    embs_alt = np.empty((n, dim), dtype="float32")
    for i in range(n):
        if rng.random() < 0.8:
            v = centers[subsys_idx[i]] + sigma * rng.standard_normal(dim).astype("float32")
        else:
            j = (subsys_idx[i] + rng.integers(1, len(_SUBSYSTEMS))) % len(_SUBSYSTEMS)
            v = centers[j] + sigma * rng.standard_normal(dim).astype("float32")
        embs_alt[i] = v / (np.linalg.norm(v) + 1e-9)

    # Text: 10 words per row, sampled without replacement
    texts: List[str] = []
    for i in range(n):
        words = rng.choice(_VOCAB, size=10, replace=True)
        texts.append(" ".join(words))

    # Graph node id (same as id for simplicity)
    graph_nodes = ids.copy()

    table = pa.table({
        "id": ids,
        "subsystem": subsystems,
        "ts": ts,
        "text": texts,
        "emb": _to_arrow_list(embs),
        "emb_alt": _to_arrow_list(embs_alt),
        "graph_node": graph_nodes,
    })

    store = TableStore.from_arrow("memory", table)
    return MemoryDataset(
        ids=ids,
        subsystems=subsystems,
        ts=ts,
        texts=texts,
        embs=embs,
        embs_alt=embs_alt,
        graph_nodes=graph_nodes,
        centers=centers,
        table=table,
        store=store,
    )


def _to_arrow_list(arr: np.ndarray) -> pa.Array:
    """Convert a (n, d) float32 array to a pa.list_(float32()) column."""
    flat = arr.reshape(-1)
    values = pa.array(flat, type=pa.float32())
    n = arr.shape[0]
    d = arr.shape[1]
    offsets = np.arange(0, (n + 1) * d, d, dtype=np.int32)
    return pa.ListArray.from_arrays(pa.array(offsets), values)


@dataclass
class MemoryDataset:
    """Holds the synthetic dataset and convenience accessors."""
    ids: np.ndarray
    subsystems: np.ndarray
    ts: np.ndarray
    texts: List[str]
    embs: np.ndarray
    embs_alt: np.ndarray
    graph_nodes: np.ndarray
    centers: np.ndarray
    table: pa.Table
    store: TableStore

    def vector(self, col: str) -> np.ndarray:
        if col == "emb":
            return self.embs
        if col == "emb_alt":
            return self.embs_alt
        raise KeyError(col)
