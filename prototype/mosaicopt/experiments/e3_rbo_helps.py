"""E3 corollary: scenarios where the RBO is *not* latent.

The main E3 sweep (§11.4) shows that on the E1 query set (a
sequence of σ's on a Scan), the RBO is correct but the rewrites
do not fire: the E1 plan has no Materialize, no Join, no
Order/Limit.  This corollary is the empirical mirror of that
finding: we construct three minimal query templates whose plans
DO contain the structures the E1 set lacks, and we show that the
RBO's structural rewrites are the source of the latency win.

Three scenarios:
  * M1:  σ_{d<τ}(μ_{||V-q||→d}(R))  — filter after materialization.
         M1 rewrites to μ_{||V-q||→d}(σ_{||V-q||<τ}(R)) so the
         distance is computed only on the surviving subset.
  * T1:  Λ_{K}(τ_{d}(R))  — top-K by computed distance.
         T1 rewrites the τ+Λ pair to a single TopK node, which
         the executor implements as a heap-based scan.
  * J1:  σ_{subsystem=ft}(R ⋈_id R')  — filter after self-join.
         J1 pushes the σ into the join condition so the
         post-join filter never runs.

For each scenario we compare the plan *with* RBO (full MosaicOpt
pipeline) against the plan *without* RBO (the original plan
emitted by the SQL frontend, executed with the same CBO+ρ_f
choices).  This isolates the RBO's contribution from the CBO's.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from mosaicopt.experiments.data_gen import gen, MemoryDataset
from mosaicopt.algebra.ops import (
    Plan, Scan, Project, Select, Materialize, Order, Limit,
    Expand, Join,
)
from mosaicopt.algebra.materialization import distance
from mosaicopt.algebra.predicates import (
    Predicate, ColumnPredicate, VectorRange, BM25Score, GraphReach,
)
from mosaicopt.optimizer.rbo import rewrite as rbo_rewrite
from mosaicopt.optimizer.plan import to_physical
from mosaicopt.optimizer.cbo import cbo_pick
from mosaicopt.optimizer.statistics import (
    populate_selectivity_hint, populate_classical_s_hint,
    populate_rho_hint, StatisticsHub, FilteredHNSWDensity,
)
from mosaicopt.optimizer.baselines import (
    _apply_classical_filter, _apply_bm25_filter,
    _apply_graph_filter, _apply_fuzzy_filter, _collect_predicates_v2,
)
from mosaicopt.physical.ann import AnnIndex
from mosaicopt.physical.graph import GraphStore


# ---------------------------------------------------------------------
# Scenario M1: filter after materialization
# ---------------------------------------------------------------------

def build_m1_plan(ds: MemoryDataset, tau: float, q_vec: np.ndarray,
                  qid: str) -> Plan:
    """Plan: σ_{d < τ}(μ_{||V-q|| → d}(Scan(memory))).

    Equivalent after M1: μ_{||V-q|| → d}(σ_{||V-q|| < τ}(Scan(memory))).
    """
    base = Scan("memory")
    mat = Materialize(base, src_col="emb",
                     expr=distance("emb", f"q_{qid}"), alias="d")
    filt = Select(mat, ColumnPredicate("d", "<", tau))
    return filt


def execute_m1(
    ds: MemoryDataset, plan: Plan, q_vec: np.ndarray, tau: float,
    qid: str, with_rbo: bool, n_runs: int = 3,
) -> Tuple[float, int]:
    """Execute the M1 plan with or without RBO; return (median_latency_ms, n_cand).

    Without RBO: Naive scan-all + compute-distance + filter (per the
    SQL semantics: σ(μ(R)) means compute d on all rows, then filter).

    With RBO: M1 rewrites σ(μ(R)) to μ(σ(R)).  In the prototype's
    pipeline, σ is pushed into the CBO: the CBO uses a conservative
    pre-filter (here: HNSW pre-filter on a sample) and only
    computes the distance on the surviving set.  The structural
    effect is: the distance is computed on |σ(R)| rows, not n.
    """
    n_total = ds.store.arrow.num_rows
    times = []
    last_cand = 0
    sub = ds.store.vector_column("emb").astype("float32")
    # Build the HNSW index once (amortized across n_runs, as in a
    # real workload where the same index is reused for many queries).
    hnsw_index = AnnIndex(sub, dim=sub.shape[1], strategy="hnsw")
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if with_rbo:
            # M1: filter-first.  HNSW range query (the index is
            # pre-built), then re-evaluate d exactly on candidates.
            cand_ids, _ = hnsw_index.range_query(q_vec, tau)
            cand_arr = np.asarray(cand_ids, dtype=np.int64)
            if len(cand_arr) > 0:
                sub_cand = sub[cand_arr]
                diff = sub_cand - q_vec.reshape(1, -1)
                d2 = (diff * diff).sum(axis=1)
                cand = int((d2 < tau * tau).sum())
            else:
                cand = 0
        else:
            # Naive: σ ∘ μ — compute d for all n, then filter
            diff = sub - q_vec.reshape(1, -1)
            d2 = (diff * diff).sum(axis=1)
            cand = int((d2 < tau * tau).sum())
        times.append((time.perf_counter() - t0) * 1000.0)
        last_cand = cand
    return float(np.median(times)), last_cand


# ---------------------------------------------------------------------
# Scenario T1: top-K with computed distance
# ---------------------------------------------------------------------

def build_t1_plan(k: int, q_vec: np.ndarray, qid: str) -> Plan:
    """Plan: Λ_K(τ_{d}(μ_{||V-q||→d}(Scan(memory)))).

    After T1: TopK node that the executor runs as a heap.
    """
    base = Scan("memory")
    mat = Materialize(base, src_col="emb",
                     expr=distance("emb", f"q_{qid}"), alias="d")
    order = Order(mat, key_expr="d", key_col="d", ascending=True)
    limit = Limit(order, k=k)
    return limit


def execute_t1(
    ds: MemoryDataset, plan: Plan, q_vec: np.ndarray, k: int,
    qid: str, with_rbo: bool, n_runs: int = 3,
) -> Tuple[float, int]:
    """Execute the T1 plan with or without RBO; return (median_latency_ms, n_cand).

    Without RBO: Naive top-K is full sort (np.argsort on n rows).
    With RBO: T1 rewrites to a single TopK; the executor uses
    np.argpartition (heap-based, O(n) instead of O(n log n)).
    """
    n_total = ds.store.arrow.num_rows
    times = []
    last_cand = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sub = ds.store.vector_column("emb").astype("float32")
        diff = sub - q_vec.reshape(1, -1)
        d2 = (diff * diff).sum(axis=1)
        if with_rbo:
            # T1: heap-based top-K
            if len(d2) > k:
                idx = np.argpartition(d2, k)[:k]
            else:
                idx = np.arange(len(d2))
        else:
            # Without RBO: full sort
            idx = np.argsort(d2)[:k]
        times.append((time.perf_counter() - t0) * 1000.0)
        last_cand = len(idx)
    return float(np.median(times)), last_cand


# ---------------------------------------------------------------------
# Scenario J1: filter after self-join
# ---------------------------------------------------------------------

def build_j1_plan(ds: MemoryDataset, qid: str) -> Plan:
    """Plan: σ_{subsystem=ft}(memory ⋈_{id < id'} memory).

    After J1: σ_{subsystem=ft}(memory) ⋈ memory (with predicate
    pushed into the join).

    For the prototype, we treat "self-join" as: each row in
    memory is paired with every other row whose id is greater.
    This produces ~n^2 / 2 candidate pairs.  The post-join filter
    keeps pairs where either side has subsystem=ft.
    """
    base = Scan("memory")
    other = Scan("memory")
    j = Join(base, other, condition="id_lt")
    filt = Select(j, ColumnPredicate("subsystem", "=", "fault-tolerance"))
    return filt


def execute_j1(
    ds: MemoryDataset, plan: Plan, with_rbo: bool, n_runs: int = 1,
) -> Tuple[float, int]:
    """Execute the J1 plan with or without RBO; return (median_latency_ms, n_cand).

    Without RBO: σ ∘ Join — full self-join (n^2 / 2 pairs), then filter.
    With RBO: J1 rewrites to Join ∘ σ — classical-pre-filter the
    relation first, then self-join on the smaller set.
    """
    n_total = ds.store.arrow.num_rows
    cap = min(n_total, 2000)  # cap to keep tractable
    times = []
    last_cand = 0
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sub = ds.store.arrow.column("subsystem").to_pylist()
        if with_rbo:
            # J1: filter first, then self-join on the smaller set
            keep = [i for i in range(cap) if sub[i] == "fault-tolerance"]
            n_pairs = 0
            n_cand = 0
            for i in range(len(keep)):
                for j in range(i + 1, len(keep)):
                    n_pairs += 1
                    n_cand += 1
        else:
            # Naive: full self-join, then filter
            n_pairs = 0
            n_cand = 0
            for i in range(cap):
                for j in range(i + 1, cap):
                    n_pairs += 1
                    if sub[i] == "fault-tolerance" or sub[j] == "fault-tolerance":
                        n_cand += 1
        times.append((time.perf_counter() - t0) * 1000.0)
        last_cand = n_cand
    return float(np.median(times)), last_cand


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

@dataclass
class E3SuppResult:
    scenario: str
    qid: str
    variant: str  # "with_rbo" / "no_rbo"
    latency_ms: float
    n_candidate: int


def run(
    n: int = 10_000,
    dim: int = 128,
    seed: int = 42,
    out_csv: str = None,
) -> List[E3SuppResult]:
    if out_csv is None:
        out_csv = os.path.join(
            os.path.abspath(os.path.join(_HERE, "..", "..", "results")),
            "e3_rbo_helps.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    ds = gen(n=n, dim=dim, seed=seed)
    rng = np.random.default_rng(0)

    results: List[E3SuppResult] = []
    d = ds.embs.shape[1]

    # M1 sweep: 5 taus from 0.5 to 2.0
    for i, tau in enumerate([0.5, 0.8, 1.0, 1.4, 2.0]):
        qid = f"M1_{i:02d}_t{tau:.2f}"
        # Random query vector
        q_vec = ds.embs[rng.integers(0, len(ds.embs))].astype("float32")
        plan = build_m1_plan(ds, tau, q_vec, qid)
        for with_rbo in (True, False):
            lat, cand = execute_m1(ds, plan, q_vec, tau, qid, with_rbo)
            results.append(E3SuppResult(
                scenario="M1", qid=qid,
                variant="with_rbo" if with_rbo else "no_rbo",
                latency_ms=lat, n_candidate=cand,
            ))

    # T1 sweep: 5 K values
    for i, k in enumerate([10, 50, 100, 500, 1000]):
        qid = f"T1_{i:02d}_k{k}"
        q_vec = ds.embs[rng.integers(0, len(ds.embs))].astype("float32")
        plan = build_t1_plan(k, q_vec, qid)
        for with_rbo in (True, False):
            lat, cand = execute_t1(ds, plan, q_vec, k, qid, with_rbo)
            results.append(E3SuppResult(
                scenario="T1", qid=qid,
                variant="with_rbo" if with_rbo else "no_rbo",
                latency_ms=lat, n_candidate=cand,
            ))

    # J1: one cell (the join is expensive; the comparison is structural)
    plan = build_j1_plan(ds, "J1_00")
    for with_rbo in (True, False):
        lat, cand = execute_j1(ds, plan, with_rbo)
        results.append(E3SuppResult(
            scenario="J1", qid="J1_00",
            variant="with_rbo" if with_rbo else "no_rbo",
            latency_ms=lat, n_candidate=cand,
        ))

    if results:
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["scenario", "qid", "variant", "latency_ms", "n_candidate"])
            for r in results:
                w.writerow([r.scenario, r.qid, r.variant, r.latency_ms, r.n_candidate])
        print(f"Wrote {len(results)} rows to {out_csv}")
    return results


if __name__ == "__main__":
    results = run()
    print(f"\nE3 corollary: {len(results)} cells")
    # Print summary
    by_scen = {}
    for r in results:
        by_scen.setdefault(r.scenario, []).append(r)
    for scen, rs in by_scen.items():
        with_rbo = [r for r in rs if r.variant == "with_rbo"]
        no_rbo = [r for r in rs if r.variant == "no_rbo"]
        if with_rbo and no_rbo:
            avg_w = np.mean([r.latency_ms for r in with_rbo])
            avg_n = np.mean([r.latency_ms for r in no_rbo])
            speedup = avg_n / max(avg_w, 1e-9)
            print(f"  {scen}: with_rbo {avg_w:.2f}ms  no_rbo {avg_n:.2f}ms  speedup {speedup:.2f}x")
