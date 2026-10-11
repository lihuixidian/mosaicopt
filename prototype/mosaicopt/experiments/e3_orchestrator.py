"""E3: Multi-modal vs multi-path orchestrator.

Setup
-----
We compare two orchestration strategies on the Q1->Q6 example:
    (a) Multi-modal: single fused query through the algebra
        - one pass over the relation
        - materializes both d1 and d2 per surviving row
        - selects with a combined predicate
        - top-K with a combined sort key
    (b) Multi-path: per-modality separate queries then merge
        - path 1: ANN over `emb` -> top-K1
        - path 2: ANN over `emb_alt` -> top-K2
        - merge: union then sort by combined key

The multi-modal plan is the canonical algebraic plan; the
multi-path plan is the structure used by Mem0/Letta.

Metrics
-------
* latency (ms)
* recall@K (vs the exact ground truth)
* code complexity proxy (number of distinct plan operators used)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from mosaicopt.experiments.data_gen import MemoryDataset
from mosaicopt.physical.ann import AnnIndex


def _topk_via_heap(scores: np.ndarray, k: int, ascending: bool = True) -> np.ndarray:
    """Return the indices of the k smallest (or largest) elements."""
    if ascending:
        # smallest k
        if k >= len(scores):
            return np.argsort(scores)
        part = np.argpartition(scores, k)[:k]
        return part[np.argsort(scores[part])]
    else:
        # largest k
        if k >= len(scores):
            return np.argsort(-scores)
        part = np.argpartition(-scores, k)[:k]
        return part[np.argsort(-scores[part])]


def run(data: MemoryDataset, k: int = 10) -> "E3Result":
    """Run the multi-modal vs multi-path orchestrator experiment.

    Returns
    -------
    E3Result
        Per-strategy latency, recall, and complexity counts.
    """
    embs = data.embs
    embs_alt = data.embs_alt
    n, dim = embs.shape
    rng = np.random.default_rng(99)
    q1 = rng.standard_normal(dim).astype("float32"); q1 /= np.linalg.norm(q1) + 1e-9
    q2 = rng.standard_normal(dim).astype("float32"); q2 /= np.linalg.norm(q2) + 1e-9
    # 70-30 fusion
    w1, w2 = 0.7, 0.3
    tau1, tau2 = 1.0, 1.0   # ranges - set wide enough to admit most rows

    # --- (a) Multi-modal: fused plan --------------------------------
    # Compute distances for all rows
    t0 = time.perf_counter()
    d1 = np.linalg.norm(embs - q1.reshape(1, -1), axis=1)
    d2 = np.linalg.norm(embs_alt - q2.reshape(1, -1), axis=1)
    keep = (d1 < tau1) & (d2 < tau2)
    scores = w1 * d1 + w2 * d2
    scores[~keep] = np.inf
    top = _topk_via_heap(scores, k=k, ascending=True)
    t_multi_modal = (time.perf_counter() - t0) * 1000.0
    multi_modal_ids = set(top.tolist())

    # --- (b) Multi-path: per-modality HNSW + merge ------------------
    t0 = time.perf_counter()
    idx1 = AnnIndex(embs, dim, strategy="hnsw")
    idx2 = AnnIndex(embs_alt, dim, strategy="hnsw")
    # Path 1: top-K via HNSW on emb
    ids1, d1_ann = idx1.knn_query(q1, k=k)
    # Path 2: top-K via HNSW on emb_alt
    ids2, d2_ann = idx2.knn_query(q2, k=k)
    # Merge: union, re-evaluate exact distances
    union = list(set(ids1.tolist()) | set(ids2.tolist()))
    if union:
        u = np.array(union)
        d1u = np.linalg.norm(embs[u] - q1.reshape(1, -1), axis=1)
        d2u = np.linalg.norm(embs_alt[u] - q2.reshape(1, -1), axis=1)
        score = w1 * d1u + w2 * d2u
        order = _topk_via_heap(score, k=k, ascending=True)
        top_mp = u[order]
    else:
        top_mp = np.array([], dtype=np.int64)
    t_multi_path = (time.perf_counter() - t0) * 1000.0
    multi_path_ids = set(top_mp.tolist())

    # --- Ground truth: exact fused plan, no top-k shortcut ----------
    d1t = np.linalg.norm(embs - q1.reshape(1, -1), axis=1)
    d2t = np.linalg.norm(embs_alt - q2.reshape(1, -1), axis=1)
    keep_t = (d1t < tau1) & (d2t < tau2)
    score_t = w1 * d1t + w2 * d2t
    score_t[~keep_t] = np.inf
    top_truth = _topk_via_heap(score_t, k=k, ascending=True)
    truth_set = set(top_truth.tolist())

    def recall(pred_set: set) -> float:
        if not truth_set:
            return 1.0
        return len(pred_set & truth_set) / len(truth_set)

    return E3Result(
        n=n,
        k=k,
        latency_multi_modal_ms=t_multi_modal,
        latency_multi_path_ms=t_multi_path,
        recall_multi_modal=recall(multi_modal_ids),
        recall_multi_path=recall(multi_path_ids),
        # Complexity proxy: count of distinct physical operators used
        ops_multi_modal=5,    # Scan + 2*Materialize + Select + TopK
        ops_multi_path=8,     # 2x (Scan + HNSW + topK) + union + sort
    )


@dataclass
class E3Result:
    n: int
    k: int
    latency_multi_modal_ms: float
    latency_multi_path_ms: float
    recall_multi_modal: float
    recall_multi_path: float
    ops_multi_modal: int
    ops_multi_path: int

    def to_csv(self, path: str) -> None:
        import csv
        keys = ["n", "k", "latency_multi_modal_ms", "latency_multi_path_ms",
                "recall_multi_modal", "recall_multi_path",
                "ops_multi_modal", "ops_multi_path"]
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerow({k: getattr(self, k) for k in keys})

    def plot(self, results_dir: str) -> None:
        """Generate the E3 latency comparison plot."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5.0, 3.5))
        labels = ["Multi-modal\n(algebra fused)", "Multi-path\n(per-modality)"]
        lats = [self.latency_multi_modal_ms, self.latency_multi_path_ms]
        recs = [self.recall_multi_modal, self.recall_multi_path]
        x = np.arange(len(labels))
        bars = ax.bar(x, lats, color=["#003049", "#C1121F"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("End-to-end latency (ms)")
        ax.set_title(f"E3: Multi-modal vs multi-path (k={self.k})")
        for i, (bar, lat, rec) in enumerate(zip(bars, lats, recs)):
            ax.text(bar.get_x() + bar.get_width() / 2, lat * 1.02,
                    f"{lat:.1f}ms\nrec={rec:.2f}", ha="center", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{results_dir}/e3_latency.png", dpi=150)
        plt.close(fig)

    def summary(self) -> Dict[str, float]:
        speedup = self.latency_multi_path_ms / self.latency_multi_modal_ms if self.latency_multi_modal_ms > 0 else 0.0
        return {
            "k": self.k,
            "latency_multi_modal_ms": self.latency_multi_modal_ms,
            "latency_multi_path_ms": self.latency_multi_path_ms,
            "recall_multi_modal": self.recall_multi_modal,
            "recall_multi_path": self.recall_multi_path,
            "speedup_multi_modal": speedup,
            "ops_multi_modal": self.ops_multi_modal,
            "ops_multi_path": self.ops_multi_path,
        }
