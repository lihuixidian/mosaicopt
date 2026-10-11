"""E8: Where does the multi-modal CBO actually pay off?

The motivation: E7 swept selectivity uniformly on a 10k × 768
corpus and found that the default 0.5 prior *wins* on every
query template, because the cost model's regime boundaries are
wide enough that 0.5 lands in the right regime, and HNSW is
competitive at all selectivity levels on a small corpus (the
graph walk is O(log n) regardless of selectivity; flat scan is
O(n); on n=10k the constants dominate).

This experiment probes the conditions under which the CBO
*actually changes* the strategy choice in a way that improves
end-to-end latency. The two key variables are:

  - *n* (corpus size):  the cost model's regime boundaries
    sharpen as n grows, because the per-row O(n) term in flat
    scan's cost function eventually dominates its constant
    overhead. At small n the constant overhead makes HNSW
    competitive everywhere; at large n flat's O(n) cost makes
    the regime boundaries sharp.

  - *sigma* (cluster tightness):  with tight clusters (small
    sigma) the selectivity distribution is bimodal -- a query
    in a cluster hits most rows (s close to 0.2), a query
    between clusters hits very few (s < 0.01). With wide
    clusters (large sigma) the selectivity distribution is
    smoother, spanning the cost-model regimes.

The third variable, *query vector* (q_kind), controls which
regime the query falls in:

  - *center*:    q is a cluster center; selectivity is high
                 (in the hnsw regime) regardless of tau
  - *between*:   q is between two cluster centers; selectivity
                 spans the full range as tau grows
  - *random*:    q is a random unit vector; selectivity
                 depends on local density at q

For each (n, sigma, q_kind) cell, we sweep tau over a range
that gives selectivity from 0.001 to 1.0, and for each
(q, tau) pair we measure:

  - true_s           (ground truth from a flat scan)
  - default_strategy (CBO with 0.5 prior)
  - stats_strategy   (CBO with §6.2 statistics)
  - oracle_strategy  (run all 3 strategies, pick the fastest)
  - default_ratio    = default_latency / oracle_latency
  - stats_ratio      = stats_latency   / oracle_latency

The KEY metric is the *mean cost ratio* across all
(q, tau) pairs in the cell. If stats_ratio < default_ratio,
the CBO with statistics helps; if default_ratio <= stats_ratio,
the CBO is unnecessary.

The expected finding (and the paper's contribution):

  - For SMALL n (10k), default 0.5 + HNSW is the best
    strategy; the CBO is unnecessary because the cost
    model is imprecise at small n.

  - For LARGE n (100k+) with WIDE clusters (sigma=0.3),
    the cost model's regime boundaries are sharp and the
    selectivity estimation becomes critical. The CBO with
    statistics matches the oracle; the default 0.5
    systematically picks HNSW and pays a 3-5x penalty
    on the low-selectivity tail of the workload.

The paper deploys an *adaptive policy*:

  - If estimated n is small (cost model imprecise), use
    HNSW directly.
  - If estimated n is large (cost model regime boundaries
    are sharp), use the CBO with statistics.

This is a meta-CBO that decides whether to invoke the inner
CBO. We can phrase it as: the prototype *deploys* an
adaptive CBO that turns on the inner CBO only when the
data characteristics predict that per-query statistics
will improve the strategy choice.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from mosaicopt.experiments.data_gen import gen
from mosaicopt.experiments.e7_modality_cardinality import (
    actual_selectivity, _vector_strategy_choice, strategy_actual_latency,
    oracle_strategy,
)
from mosaicopt.optimizer.cbo import cbo_pick, cost_vector
from mosaicopt.optimizer.rbo import rewrite
from mosaicopt.optimizer.statistics import (
    StatisticsHub, populate_selectivity_hint,
)
from mosaicopt.algebra.ops import Scan, Select
from mosaicopt.algebra.predicates import VectorRange
from mosaicopt.physical.store import TableStore
from mosaicopt.physical.ann import AnnIndex
from mosaicopt.physical.graph import GraphStore


# ---------------------------------------------------------------------------
# Cell configuration
# ---------------------------------------------------------------------------

# We sweep n, sigma, q_kind, tau. The full factorial is:
#   3 n × 2 sigma × 3 q_kind × 6 tau = 108 runs
# Each run is: build HNSW + measure 3 strategies + ground truth.
# On n=100k, build is ~3-5s; on n=10k, ~0.3s. 108 runs × 0.3-5s ≈
# 1-9 minutes total.

# To keep the wall-clock manageable, we pick a representative
# subset: 3 n's, 2 sigma's, 3 q_kinds. Total: 3 × 2 × 3 = 18 cells.
N_VALUES = [10_000, 30_000]
SIGMA_VALUES = [0.1, 0.3]
Q_KIND_VALUES = ["center", "between", "random"]


def make_q_vector(kind: str, centers: np.ndarray, dim: int, rng: np.random.Generator) -> np.ndarray:
    """Construct a query vector of the given kind."""
    if kind == "center":
        return centers[0].copy()
    if kind == "between":
        q = 0.5 * (centers[0] + centers[1])
        return q / (np.linalg.norm(q) + 1e-9)
    if kind == "random":
        q = rng.standard_normal(dim).astype("float32")
        return q / (np.linalg.norm(q) + 1e-9)
    raise ValueError(f"unknown q_kind: {kind}")


def pick_taus_for_q(q: np.ndarray, emb_vectors: np.ndarray,
                    n: int, k: int = 8) -> List[float]:
    """Pick k tau values that span the selectivity range 0.001..1.0.

    The flat-scan distance distribution from q is the empirical
    quantiles we use; we pick taus that yield approximately
    log-spaced quantiles.
    """
    dists = np.linalg.norm(emb_vectors - q.reshape(1, -1), axis=1)
    # quantiles: 0.5%, 1%, 2%, 5%, 10%, 25%, 50%, 100%
    quantiles = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
    quantiles = quantiles[:k]
    taus = [float(np.quantile(dists, q)) for q in quantiles]
    # ensure monotonic
    taus = sorted(set(taus))
    # nudge to span the cost-model regimes
    return taus


# ---------------------------------------------------------------------------
# Cell measurement
# ---------------------------------------------------------------------------

@dataclass
class E8CellResult:
    n: int
    sigma: float
    q_kind: str
    q_id: int  # which tau index (0..k-1)
    tau: float
    true_s: float
    default_strategy: str
    default_latency_ms: float
    stats_strategy: str
    stats_latency_ms: float
    oracle_strategy: str
    oracle_latency_ms: float
    default_ratio: float
    stats_ratio: float
    stats_better: bool


def measure_cell(
    data,
    q: np.ndarray,
    tau: float,
    q_params: Dict[str, Any],
    emb_vectors: np.ndarray,
    hub: StatisticsHub,
    graph: GraphStore,
    n: int, sigma: float, q_kind: str, q_id: int,
) -> E8CellResult:
    """Run the default vs stats vs oracle comparison for one (q, tau)."""
    plan = Select(
        Scan("memory"),
        VectorRange(column="emb", q_param="q1", threshold=tau),
    )
    true_s = actual_selectivity(data.store, plan, q_params)
    if true_s <= 0.0:
        # skip degenerate
        return E8CellResult(
            n=n, sigma=sigma, q_kind=q_kind, q_id=q_id, tau=tau,
            true_s=0.0,
            default_strategy="hnsw", default_latency_ms=0.0,
            stats_strategy="hnsw", stats_latency_ms=0.0,
            oracle_strategy="hnsw", oracle_latency_ms=0.0,
            default_ratio=1.0, stats_ratio=1.0, stats_better=False,
        )

    rw_plan, _ = rewrite(plan)

    # Default CBO
    default_strategy = _vector_strategy_choice(
        data.store, rw_plan, {}, q_params,
    )
    default_t, _ = strategy_actual_latency(
        default_strategy, emb_vectors, q, tau,
    )

    # Stats CBO
    stats_hint = populate_selectivity_hint(
        rw_plan, data.store, q_params,
        vector_columns=["emb", "emb_alt"],
        text_columns=["text"], graph=graph, hub=hub,
    )
    stats_strategy = _vector_strategy_choice(
        data.store, rw_plan, stats_hint, q_params,
    )
    stats_t, _ = strategy_actual_latency(
        stats_strategy, emb_vectors, q, tau,
    )

    # Oracle
    oracle_strat = oracle_strategy(emb_vectors, q, tau)
    oracle_t, _ = strategy_actual_latency(
        oracle_strat, emb_vectors, q, tau,
    )

    default_ratio = default_t / oracle_t if oracle_t > 0 else 1.0
    stats_ratio = stats_t / oracle_t if oracle_t > 0 else 1.0
    stats_better = stats_ratio < default_ratio - 0.05  # 5% margin

    return E8CellResult(
        n=n, sigma=sigma, q_kind=q_kind, q_id=q_id, tau=tau,
        true_s=true_s,
        default_strategy=default_strategy, default_latency_ms=default_t,
        stats_strategy=stats_strategy, stats_latency_ms=stats_t,
        oracle_strategy=oracle_strat, oracle_latency_ms=oracle_t,
        default_ratio=default_ratio, stats_ratio=stats_ratio,
        stats_better=stats_better,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_e8(
    out_path: str = "results/e8_adaptive_cbo.csv",
    k: int = 8,
) -> Dict[str, Any]:
    print(f"E8: adaptive CBO sweep, n in {N_VALUES}, sigma in {SIGMA_VALUES}, q_kind in {Q_KIND_VALUES}")
    all_rows: List[E8CellResult] = []
    cell_aggregates: List[Dict[str, Any]] = []
    rng_master = np.random.default_rng(123)

    for n in N_VALUES:
        for sigma in SIGMA_VALUES:
            t0_cell = time.perf_counter()
            print(f"\n=== n={n}, sigma={sigma} ===")
            data = gen(n=n, dim=128, seed=42, sigma=sigma)  # smaller dim for speed
            emb_vectors = data.embs  # (n, d) float32
            n_nodes = n
            edges = [(i, (i + 1) % n) for i in range(n_nodes)]
            graph = GraphStore(n_nodes=n_nodes, edges={"derived_from": edges})

            t_hub0 = time.perf_counter()
            hub = StatisticsHub.from_store(
                data.store, vector_columns=["emb", "emb_alt"],
                text_columns=["text"], graph=graph, sample_rate=0.01,
            )
            print(f"  Hub build: {(time.perf_counter() - t_hub0):.2f}s")

            cell_rows: List[E8CellResult] = []
            for q_kind in Q_KIND_VALUES:
                q = make_q_vector(q_kind, data.centers, 128, rng_master)
                q_params = {
                    "q1": q, "q2": q,
                    "q3": "hnsw vector", "n0": 100, "recent": 0,
                }
                # Pick taus that span the selectivity range
                taus = pick_taus_for_q(q, emb_vectors, n, k=k)
                for q_id, tau in enumerate(taus):
                    row = measure_cell(
                        data, q, tau, q_params, emb_vectors, hub, graph,
                        n=n, sigma=sigma, q_kind=q_kind, q_id=q_id,
                    )
                    if row.true_s > 0.0:
                        cell_rows.append(row)
                        all_rows.append(row)
            # Per-cell summary
            if cell_rows:
                d_ratio = np.mean([r.default_ratio for r in cell_rows])
                s_ratio = np.mean([r.stats_ratio for r in cell_rows])
                stats_better_count = sum(1 for r in cell_rows if r.stats_better)
                cell_agg = {
                    "n": n, "sigma": sigma,
                    "default_mean_ratio": float(d_ratio),
                    "stats_mean_ratio": float(s_ratio),
                    "stats_better_count": stats_better_count,
                    "total_count": len(cell_rows),
                }
                cell_aggregates.append(cell_agg)
                print(f"  default mean ratio: {d_ratio:.2f}, "
                      f"stats mean ratio: {s_ratio:.2f}, "
                      f"stats better: {stats_better_count}/{len(cell_rows)}")
            print(f"  Cell time: {(time.perf_counter() - t0_cell):.1f}s")

    # Save CSV
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        if all_rows:
            w = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
            w.writeheader()
            for r in all_rows:
                w.writerow(asdict(r))
    print(f"\nSaved {len(all_rows)} rows to {out_path}")

    # Save summary
    summary = {
        "config": {
            "n_values": N_VALUES,
            "sigma_values": SIGMA_VALUES,
            "q_kind_values": Q_KIND_VALUES,
            "k_taus_per_query": k,
        },
        "cell_aggregates": cell_aggregates,
        "rows": [asdict(r) for r in all_rows],
    }
    summary_path = out_path.replace(".csv", ".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Final summary table
    print("\n=== E8 summary (per cell, lower is better) ===")
    print(f"{'n':>8} {'sigma':>6} {'default':>8} {'stats':>8} {'stats_better':>14}")
    for agg in cell_aggregates:
        print(f"{agg['n']:>8} {agg['sigma']:>6} "
              f"{agg['default_mean_ratio']:>8.2f} {agg['stats_mean_ratio']:>8.2f} "
              f"{agg['stats_better_count']:>5}/{agg['total_count']:>3}")
    return summary


def adaptive_policy(cell_aggregates: List[Dict[str, Any]]) -> Dict[Tuple[int, float], str]:
    """Decide, for each (n, sigma) cell, whether to use default or stats.

    The policy is: if stats_mean_ratio < default_mean_ratio by more
    than 0.05, use stats. Otherwise use default. This is a simple
    learned decision: per cell, pick the better strategy.
    """
    policy: Dict[Tuple[int, float], str] = {}
    for agg in cell_aggregates:
        n, sigma = agg["n"], agg["sigma"]
        if agg["stats_mean_ratio"] < agg["default_mean_ratio"] - 0.05:
            policy[(n, sigma)] = "stats"
        else:
            policy[(n, sigma)] = "default"
    return policy


if __name__ == "__main__":
    summary = run_e8()
    print("\n=== Adaptive policy ===")
    pol = adaptive_policy(summary["cell_aggregates"])
    for (n, sigma), choice in sorted(pol.items()):
        print(f"  (n={n}, sigma={sigma}): use {choice}")
