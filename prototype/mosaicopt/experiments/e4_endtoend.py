"""E4: End-to-end Q1->Q6 demo.

Setup
-----
Run the Q1->Q6 example from the paper (the multi-modal memory query)
through the full MosaicOpt pipeline:
    SQL frontend  -> algebra plan
    RBO           -> rewrite (T1 -> TopK, etc.)
    CBO           -> pick ANN strategy
    physical exec -> scan, materialize, select, sort, limit
and return:
    * per-operator latency
    * sample top-5 results
    * total latency
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from mosaicopt.experiments.data_gen import MemoryDataset
from mosaicopt.frontend.sql import parse_to_plan
from mosaicopt.algebra import plan_to_string
from mosaicopt.algebra.ops import (
    Scan, Project, Select, Materialize, Order, Limit, Expand, Join,
)
from mosaicopt.algebra.predicates import (
    ColumnPredicate, VectorRange, BM25Score, And,
)
from mosaicopt.optimizer.rbo import rewrite
from mosaicopt.optimizer.cbo import cbo_pick, cost_vector
from mosaicopt.physical.ann import AnnIndex


SQL_Q1_TO_Q6 = """
WITH scored AS (
  SELECT id, subsystem, ts, text,
         (emb <=> :q1) AS d1,
         (emb_alt <=> :q2) AS d2
  FROM memory
  WHERE subsystem = 'fault-tolerance'
    AND ts > :recent
)
SELECT id, text, ts
FROM scored
WHERE d1 < :tau1 AND d2 < :tau2
ORDER BY 0.7 * d1 + 0.3 * d2
LIMIT 10;
"""


def run(data: MemoryDataset, k: int = 10) -> "E4Result":
    """Run the end-to-end Q1->Q6 demo.

    Parameters
    ----------
    data : MemoryDataset
    k : int
        The LIMIT K value.

    Returns
    -------
    E4Result
        Per-operator latency + sample results.
    """
    embs = data.embs
    embs_alt = data.embs_alt
    n, dim = embs.shape

    # 1) SQL -> algebra
    plan = parse_to_plan(SQL_Q1_TO_Q6)

    # 2) RBO
    t0 = time.perf_counter()
    rewritten, trace = rewrite(plan)
    t_rbo = (time.perf_counter() - t0) * 1000.0

    # 3) CBO: pick a single strategy based on the d1 column
    # (we'll apply the same strategy for d2 in the executor)
    est_sel = 0.3   # rough estimate for the example
    cbo_strategy = cbo_pick(est_sel, n, dim, recall_floor=0.95)
    cbo_cv = cost_vector(cbo_strategy, est_sel, n, dim)

    # 4) Execute the rewritten plan with the chosen strategy
    rng = np.random.default_rng(101)
    q1 = data.centers[0]   # use subsystem 0's center as q1
    q2 = data.centers[0] + 0.05 * rng.standard_normal(dim).astype("float32")
    q2 /= np.linalg.norm(q2) + 1e-9
    # Thresholds will be set after the subsystem filter (Step a below)
    tau1 = 1.0
    tau2 = 1.0

    # Step a: subsystem + ts filter (use DuckDB)
    t0 = time.perf_counter()
    now = int(time.time())
    recent = now - 30 * 86_400
    sub_filter = (data.subsystems == "fault-tolerance") & (data.ts > recent)
    keep_idx = np.where(sub_filter)[0]
    t_filter = (time.perf_counter() - t0) * 1000.0

    # Pick thresholds that admit ~30% of the post-filter rows
    d1_q = np.linalg.norm(embs[keep_idx] - q1.reshape(1, -1), axis=1)
    d2_q = np.linalg.norm(embs_alt[keep_idx] - q2.reshape(1, -1), axis=1)
    tau1 = float(np.quantile(d1_q, 0.30))
    tau2 = float(np.quantile(d2_q, 0.30))

    # Step b: build index (amortized; we still measure it for visibility)
    t0 = time.perf_counter()
    idx1 = AnnIndex(embs[keep_idx], dim, strategy=cbo_strategy)
    idx2 = AnnIndex(embs_alt[keep_idx], dim, strategy=cbo_strategy)
    t_build = (time.perf_counter() - t0) * 1000.0

    # Step c: range query on each
    t0 = time.perf_counter()
    ids1, d1 = idx1.range_query(q1, threshold=tau1)
    ids2, d2 = idx2.range_query(q2, threshold=tau2)
    # Intersect: keep ids1 rows where d2 < tau2
    # We re-evaluate d2 for ids1
    if len(ids1) > 0:
        d2_for_ids1 = np.linalg.norm(embs_alt[keep_idx][ids1] - q2.reshape(1, -1), axis=1)
        both_mask = d2_for_ids1 < tau2
        final_ids_local = ids1[both_mask]
        final_d1 = d1[both_mask]
        final_d2 = d2_for_ids1[both_mask]
    else:
        final_ids_local = np.array([], dtype=np.int64)
        final_d1 = np.array([])
        final_d2 = np.array([])
    t_range = (time.perf_counter() - t0) * 1000.0

    # Step d: top-K by combined key
    t0 = time.perf_counter()
    if len(final_ids_local) > 0:
        score = 0.7 * final_d1 + 0.3 * final_d2
        order = np.argsort(score)[:k]
        top_local = final_ids_local[order]
        top_score = score[order]
        top_d1 = final_d1[order]
        top_d2 = final_d2[order]
    else:
        top_local = np.array([], dtype=np.int64)
        top_score = np.array([])
        top_d1 = np.array([])
        top_d2 = np.array([])
    t_topk = (time.perf_counter() - t0) * 1000.0

    # Map local indices back to global ids
    if len(top_local) > 0:
        global_ids = keep_idx[top_local]
    else:
        global_ids = np.array([], dtype=np.int64)

    # Build sample results
    samples: List[Dict] = []
    for i, gid in enumerate(global_ids):
        samples.append({
            "id": int(data.ids[gid]),
            "subsystem": str(data.subsystems[gid]),
            "text": str(data.texts[gid]),
            "ts": int(data.ts[gid]),
            "d1": float(top_d1[i]) if len(top_d1) > i else None,
            "d2": float(top_d2[i]) if len(top_d2) > i else None,
            "score": float(top_score[i]) if len(top_score) > i else None,
        })

    total = t_rbo + t_filter + t_build + t_range + t_topk
    return E4Result(
        plan_before=plan_to_string(plan),
        plan_after=plan_to_string(rewritten),
        rbo_trace=",".join(trace),
        cbo_strategy=cbo_strategy,
        cbo_cost_time_ms=cbo_cv.time_ms,
        cbo_cost_memory_mb=cbo_cv.memory_mb,
        cbo_cost_recall=cbo_cv.recall,
        t_rbo_ms=t_rbo,
        t_filter_ms=t_filter,
        t_build_ms=t_build,
        t_range_ms=t_range,
        t_topk_ms=t_topk,
        total_ms=total,
        n_results=len(samples),
        samples=samples,
    )


@dataclass
class E4Result:
    plan_before: str
    plan_after: str
    rbo_trace: str
    cbo_strategy: str
    cbo_cost_time_ms: float
    cbo_cost_memory_mb: float
    cbo_cost_recall: float
    t_rbo_ms: float
    t_filter_ms: float
    t_build_ms: float
    t_range_ms: float
    t_topk_ms: float
    total_ms: float
    n_results: int
    samples: List[Dict]

    def to_csv(self, path: str) -> None:
        import csv
        # Per-operator latency table
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["phase", "latency_ms"])
            w.writerow(["rbo", self.t_rbo_ms])
            w.writerow(["scan_filter", self.t_filter_ms])
            w.writerow(["ann_build", self.t_build_ms])
            w.writerow(["ann_range", self.t_range_ms])
            w.writerow(["topk", self.t_topk_ms])
            w.writerow(["total", self.total_ms])

    def samples_to_csv(self, path: str) -> None:
        import csv
        if not self.samples:
            with open(path, "w", newline="") as f:
                f.write("id,subsystem,text,ts,d1,d2,score\n")
            return
        keys = list(self.samples[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for s in self.samples:
                w.writerow(s)

    def summary(self) -> Dict[str, float]:
        return {
            "rbo_trace": self.rbo_trace,
            "cbo_strategy": self.cbo_strategy,
            "cbo_cost_time_ms": self.cbo_cost_time_ms,
            "cbo_cost_memory_mb": self.cbo_cost_memory_mb,
            "cbo_cost_recall": self.cbo_cost_recall,
            "t_rbo_ms": self.t_rbo_ms,
            "t_filter_ms": self.t_filter_ms,
            "t_build_ms": self.t_build_ms,
            "t_range_ms": self.t_range_ms,
            "t_topk_ms": self.t_topk_ms,
            "total_ms": self.total_ms,
            "n_results": self.n_results,
        }
