"""E1: M1 rewrite effect.

Setup
-----
We pick 100 random queries with selectivity in {0.1, 0.3, 0.5, 0.7, 0.9}.
Each query is of the form
    Q = Sigma_{d < tau} ( Mu_{||V-q|| -> d} (memory) )
where V is the `emb` column, q is a query vector, and tau is chosen
to give the target selectivity.

For each query we measure two execution strategies:

(a) Naive:    scan all rows, compute ||V - q||, filter
    Equivalent to: Sigma_{d < tau}(Mu_{||V-q|| -> d}(memory))

(b) Rewritten: filter using an index first, then compute distance on
    survivors only.
    Equivalent to: Mu_{||V-q|| -> d}(Sigma_{d < tau}(memory))
    which is M1. Because we cannot pre-filter by the distance (we
    have not computed it), the implementation uses a flat scan with
    a partial pre-filter (top-k via HNSW, then exact re-eval).

For a true M1, we'd need to apply the distance threshold as a *range*
pre-filter using the HNSW index. The HNSW range query gives us
exactly that: only the rows within radius tau of q. The naive plan
materializes d for *every* row first, then filters.

Metrics
-------
* end-to-end latency (ms)
* output match (the two plans should return the same row ids)
* selectivity of the filter on the materialized column
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from mosaicopt.experiments.data_gen import MemoryDataset
from mosaicopt.physical.ann import AnnIndex


def _tau_for_selectivity(embs: np.ndarray, q: np.ndarray, s: float) -> float:
    """Pick a distance threshold tau that gives approximately selectivity s."""
    d = np.linalg.norm(embs - q.reshape(1, -1), axis=1)
    # Threshold = s-th percentile of distances
    return float(np.quantile(d, s))


def run(data: MemoryDataset, n_queries: int = 100) -> "E1Result":
    """Run the M1 rewrite experiment.

    Returns
    -------
    E1Result
        Per-query latency for both strategies + the trace of the
        RBO (which rules fired).
    """
    embs = data.embs
    n, dim = embs.shape
    rng = np.random.default_rng(123)

    selectivities = [0.1, 0.3, 0.5, 0.7, 0.9]
    # Build HNSW index once (amortized)
    hnsw_idx = AnnIndex(embs, dim, strategy="hnsw")
    flat_idx = AnnIndex(embs, dim, strategy="flat")  # also flat, for the (c) test

    rows: List[Dict] = []
    for q_idx in range(n_queries):
        sel = selectivities[q_idx % len(selectivities)]
        q = rng.standard_normal(dim).astype("float32")
        q /= np.linalg.norm(q) + 1e-9
        tau = _tau_for_selectivity(embs, q, sel)

        # (a) Naive: full scan + materialize + filter (exact baseline)
        t0 = time.perf_counter()
        d_all = np.linalg.norm(embs - q.reshape(1, -1), axis=1)
        naive_keep = d_all < tau
        t_naive = (time.perf_counter() - t0) * 1000.0

        # (b) M1 with the SAME physical operator: pre-filter via flat range
        #     scan, then compute d on survivors. This should be the SAME
        #     output as (a), since flat range is exact. Demonstrates the
        #     *logical* equivalence of M1.
        t0 = time.perf_counter()
        ids_flat, _ = flat_idx.range_query(q, threshold=tau)
        rewritten_flat_ids = ids_flat  # flat range is exact
        t_rewritten_flat = (time.perf_counter() - t0) * 1000.0

        # (c) M1 with HNSW: pre-filter via HNSW range (approximate),
        #     then re-evaluate exact distances. Demonstrates the
        #     *physical* divergence from N1.
        t0 = time.perf_counter()
        ids_hnsw, _ = hnsw_idx.range_query(q, threshold=tau)
        if len(ids_hnsw) > 0:
            d_hnsw = np.linalg.norm(embs[ids_hnsw] - q.reshape(1, -1), axis=1)
            rewritten_hnsw_ids = ids_hnsw[d_hnsw < tau]
        else:
            rewritten_hnsw_ids = np.array([], dtype=np.int64)
        t_rewritten_hnsw = (time.perf_counter() - t0) * 1000.0

        # Output
        naive_ids = set(np.where(naive_keep)[0].tolist())
        flat_ids = set(rewritten_flat_ids.tolist())
        hnsw_ids = set(rewritten_hnsw_ids.tolist())
        match_flat = naive_ids == flat_ids
        if len(naive_ids) > 0:
            recall_hnsw = len(hnsw_ids & naive_ids) / len(naive_ids)
        else:
            recall_hnsw = 1.0

        speedup_hnsw = t_naive / t_rewritten_hnsw if t_rewritten_hnsw > 0 else float("inf")

        rows.append({
            "q_idx": q_idx,
            "selectivity": sel,
            "tau": tau,
            "n_naive_kept": len(naive_ids),
            "n_flat_kept": len(flat_ids),
            "n_hnsw_kept": len(hnsw_ids),
            "latency_naive_ms": t_naive,
            "latency_flat_m1_ms": t_rewritten_flat,
            "latency_hnsw_m1_ms": t_rewritten_hnsw,
            "speedup_hnsw": speedup_hnsw,
            "output_match_flat": match_flat,
            "recall_hnsw": recall_hnsw,
        })

    return E1Result(rows=rows)


@dataclass
class E1Result:
    rows: List[Dict]

    def to_csv(self, path: str) -> None:
        import csv
        if not self.rows:
            return
        keys = list(self.rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

    def summary(self) -> Dict[str, float]:
        """Aggregate metrics for the paper."""
        n = len(self.rows)
        avg_naive = float(np.mean([r["latency_naive_ms"] for r in self.rows]))
        avg_flat = float(np.mean([r["latency_flat_m1_ms"] for r in self.rows]))
        avg_hnsw = float(np.mean([r["latency_hnsw_m1_ms"] for r in self.rows]))
        match_rate_flat = float(np.mean([1.0 if r["output_match_flat"] else 0.0 for r in self.rows]))
        avg_recall_hnsw = float(np.mean([r["recall_hnsw"] for r in self.rows]))
        avg_speedup_hnsw = avg_naive / avg_hnsw if avg_hnsw > 0 else 0.0
        return {
            "n_queries": n,
            "avg_latency_naive_ms": avg_naive,
            "avg_latency_flat_m1_ms": avg_flat,
            "avg_latency_hnsw_m1_ms": avg_hnsw,
            "avg_speedup_hnsw_vs_naive": avg_speedup_hnsw,
            "output_match_flat_rate": match_rate_flat,
            "avg_recall_hnsw": avg_recall_hnsw,
        }
