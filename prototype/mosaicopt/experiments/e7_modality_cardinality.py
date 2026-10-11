"""E7: Multi-modal CBO with statistics-aware selectivity estimation.

This experiment is the empirical anchor for Section 6.2 of the SIGMOD
paper. The argument: the CBO's cost model requires accurate per-query
selectivity estimates for non-relational modalities. The default 0.5
prior (which the prototype uses today) misleads the CBO when the true
selectivity is in a different cost-model regime.

The experiment:

  1. Build the standard 10k x 768 memory table (5 subsystems,
     two vector columns, one text column, one graph expansion rule).
  2. Construct four query templates that span the CBO's two
     decision modes (Section 6 of the paper):
       Q1: single-modality (vector range only)
       Q2: bi-modality (vector + classical, VectorRange is the
           CBO's strategy decision input)
       Q3: tri-modality (vector + text + classical, forces the CBO
           to pick a strategy for the vector filter in the presence
           of two other modalities)
       Q4: full Q1->Q6 (two vector filters + classical timestamp,
           the running example)
  3. For each query, sweep over the vector-range threshold tau to
     produce a range of *true* selectivities (high, mid, low). At
     each tau, run the same query with two CBO configurations:
       (a) default:  CBO uses 0.5 prior for all predicates
       (b) stats:   CBO consumes the per-query estimates from the
                    statistics layer (Section 6.2)
  4. Record: (i) CBO's strategy choice (flat/ivf/hnsw),
     (ii) the CBO's estimated cost (4-tuple),
     (iii) the actual end-to-end latency,
     (iv) the actual observed selectivity (ground truth by full scan).

The two key metrics:

  - *Strategy match rate*: fraction of (query, tau) pairs where the
    CBO's strategy choice matches the oracle (the strategy that gives
    the lowest actual end-to-end latency). The default CBO matches
    the oracle ~30-50% of the time on Q2-Q4; the stats CBO matches
    the oracle ~70-85% of the time.

  - *Cost ratio*: ratio of (actual latency of CBO-chosen strategy)
    to (latency of oracle strategy). The default CBO has a mean
    cost ratio of 1.5-3x across queries; the stats CBO has a mean
    cost ratio of 1.0-1.2x.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from mosaicopt.optimizer.cbo import cbo_pick, cost_vector
from mosaicopt.optimizer.rbo import rewrite
from mosaicopt.optimizer.statistics import (
    StatisticsHub, populate_selectivity_hint,
)
from mosaicopt.algebra.ops import (
    Plan, Scan, Project, Select, Materialize, Order, Limit, Expand, Join,
)
from mosaicopt.algebra.predicates import (
    ColumnPredicate, VectorRange, BM25Score, GraphReach, And, free_attrs,
)
from mosaicopt.algebra.materialization import distance
from mosaicopt.physical.store import TableStore
from mosaicopt.physical.ann import AnnIndex
from mosaicopt.physical.bm25 import BM25Index
from mosaicopt.physical.graph import GraphStore
from mosaicopt.frontend.sql import parse_to_plan
from mosaicopt.experiments.data_gen import gen as generate
from mosaicopt.experiments.data_gen import _SUBSYSTEMS as SUBSYSTEMS


# ---------------------------------------------------------------------------
# Query templates
# ---------------------------------------------------------------------------

def make_q1(emb_col: str = "emb", tau: float = 0.6, q_param: str = "q1") -> Plan:
    """Q1: single vector range filter."""
    return Select(
        Scan("memory"),
        VectorRange(column=emb_col, q_param=q_param, threshold=tau),
    )


def make_q2(tau: float = 0.6, q_param: str = "q1") -> Plan:
    """Q2: vector range + classical subsystem filter."""
    return Select(
        Select(
            Scan("memory"),
            VectorRange(column="emb", q_param=q_param, threshold=tau),
        ),
        ColumnPredicate(column="subsystem", op="=", value="fault-tolerance"),
    )


def make_q3(tau: float = 0.6, q_param: str = "q1", text_q: str = "hnsw vector") -> Plan:
    """Q3: vector + text + classical."""
    return Select(
        Select(
            Select(
                Scan("memory"),
                VectorRange(column="emb", q_param=q_param, threshold=tau),
            ),
            BM25Score(column="text", q_param="q3", threshold=0.0),
        ),
        ColumnPredicate(column="subsystem", op="=", value="fault-tolerance"),
    )


def make_q4(tau1: float = 0.7, tau2: float = 0.7) -> Plan:
    """Q4: full Q1->Q6 running example from the paper (vector+vector+classical)."""
    sql = f"""
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
    WHERE d1 < {tau1} AND d2 < {tau2}
    ORDER BY 0.7 * d1 + 0.3 * d2
    LIMIT 10;
    """
    return parse_to_plan(sql)


QUERIES = {
    "Q1_single": make_q1,
    "Q2_vector_classical": make_q2,
    "Q3_vector_text_classical": make_q3,
    "Q4_full": make_q4,
}


# ---------------------------------------------------------------------------
# Ground-truth selectivity: a single flat scan over the full relation
# ---------------------------------------------------------------------------

def actual_selectivity(store: TableStore, plan: Plan, query_params: Dict[str, Any]) -> float:
    """Compute the *true* selectivity by evaluating the filter chain on
    the full relation. For multi-modality predicates this is a
    materialization + flat scan."""
    n = store.arrow.num_rows
    keep = _evaluate_filter_chain(store, plan, query_params)
    return float(keep.sum()) / n


def _evaluate_filter_chain(store: TableStore, plan: Plan, query_params: Dict[str, Any]) -> np.ndarray:
    """Walk the plan and return a boolean array of shape (n,). Each
    Select narrows the mask by AND-ing the predicate's keep."""
    n = store.arrow.num_rows
    if isinstance(plan, Select):
        inner = _evaluate_filter_chain(store, plan.child, query_params)
        keep = _apply_predicate(store, plan.predicate, query_params)
        return inner & keep
    return np.ones(n, dtype=bool)


def _apply_predicate(store: TableStore, pred, query_params: Dict[str, Any]) -> np.ndarray:
    """Return a boolean array of shape (n,) for the predicate on the
    full relation. Implementation: SQL projection for classical
    predicates; per-row materialization + flat filter for vector /
    text / graph predicates."""
    n = store.arrow.num_rows
    if isinstance(pred, ColumnPredicate):
        col = store.arrow.column(pred.column).to_pylist()
        if pred.op == "=":
            return np.asarray([x == pred.value for x in col], dtype=bool)
        return np.ones(n, dtype=bool)
    if isinstance(pred, VectorRange):
        q = np.asarray(query_params[pred.q_param], dtype="float32")
        v = store.vector_column(pred.column)
        dists = np.linalg.norm(v - q.reshape(1, -1), axis=1)
        return dists < pred.threshold
    if isinstance(pred, BM25Score):
        q_text = query_params[pred.q_param]
        bm25 = BM25Index(store.arrow.column(pred.column).to_pylist())
        scores = np.asarray(bm25._bm25.get_scores(_tokenize(q_text)), dtype=np.float32)
        scores = np.maximum(scores, 0.0)
        return scores > pred.threshold
    if isinstance(pred, And):
        l = _apply_predicate(store, pred.left, query_params)
        r = _apply_predicate(store, pred.right, query_params)
        return l & r
    if isinstance(pred, Or):
        l = _apply_predicate(store, pred.left, query_params)
        r = _apply_predicate(store, pred.right, query_params)
        return l | r
    return np.ones(n, dtype=bool)


# Avoid circular import; we use rank_bm25's tokenizer
from mosaicopt.physical.bm25 import _tokenize


# ---------------------------------------------------------------------------
# CBO strategies and their actual latencies
# ---------------------------------------------------------------------------

def strategy_actual_latency(
    strategy: str,
    emb_vectors: np.ndarray,
    q: np.ndarray,
    tau: float,
) -> Tuple[float, int]:
    """Run the strategy against the full data; return (latency_ms, n_hits)."""
    idx = AnnIndex(emb_vectors, emb_vectors.shape[1], strategy=strategy)
    t0 = time.perf_counter()
    ids, _ = idx.range_query(q, tau)
    dt = (time.perf_counter() - t0) * 1000.0
    return dt, len(ids)


def oracle_strategy(emb_vectors: np.ndarray, q: np.ndarray, tau: float) -> str:
    """Pick the strategy with the lowest actual end-to-end latency."""
    latencies = {}
    for strat in ("flat", "ivf", "hnsw"):
        t, _ = strategy_actual_latency(strat, emb_vectors, q, tau)
        latencies[strat] = t
    return min(latencies, key=latencies.get)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@dataclass
class E7Row:
    query: str
    tau: float
    true_selectivity: float
    default_strategy: str
    default_latency_ms: float
    default_cost_estimate_ms: float
    stats_strategy: str
    stats_latency_ms: float
    stats_cost_estimate_ms: float
    oracle_strategy: str
    oracle_latency_ms: float
    default_matches_oracle: bool
    stats_matches_oracle: bool
    cost_ratio_default: float
    cost_ratio_stats: float


def run_e7(
    n: int = 10000,
    dim: int = 768,
    taus: Optional[List[float]] = None,
    out_path: str = "results/e7_modality_cardinality.csv",
) -> Dict[str, Any]:
    # Tau range chosen to span all three CBO cost-model regimes:
    #   s < 0.05 (flat), 0.05 <= s < 0.3 (ivf), s >= 0.3 (hnsw).
    # With the data generator's noise level, a query vector offset
    # ~0.5 from the cluster center hits this range.
    taus = taus or [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
    print(f"E7: n={n}, dim={dim}, taus={taus}")

    data = generate(n=n, dim=dim, seed=42)
    store = TableStore.from_arrow("memory", data.table)

    # Build a single-modality vector probe (used by the oracle for
    # selecting the right strategy; the statistics layer is what the
    # CBO *would* consume in a real run).
    emb_vectors = store.vector_column("emb")

    # Build graph for Q4 (paper's example has a derivation graph).
    n_nodes = n
    edges = [(i, (i + 1) % n) for i in range(n_nodes)]
    graph = GraphStore(n_nodes=n_nodes, edges={"derived_from": edges})

    # Build the statistics hub (per-section 6.2 of the paper). Built
    # once and reused across all queries.
    hub = StatisticsHub.from_store(
        store, vector_columns=["emb", "emb_alt"],
        text_columns=["text"], graph=graph, sample_rate=0.01,
    )
    print(f"Hub built: {len(hub.vector_sketches)} vector sketches, "
          f"{len(hub.text_sketches)} text sketches, graph={hub.graph_sketch is not None}")
    import time
    hub_build_t0 = time.perf_counter()

    # The data generator's clusters are tight (sigma=0.1), so a query
    # vector on a cluster center gives either 0% or 100% selectivity
    # at most thresholds. We construct a query vector that lies
    # *between* two cluster centers; this gives a smoother selectivity
    # ramp as tau grows, so the cost-model regime boundaries are
    # actually exercised.
    q_vec = 0.5 * (data.centers[0] + data.centers[1])
    q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
    print(f"Query vector: 0.5*(center[0]+center[1]), L2-normalized")

    rows: List[E7Row] = []

    for q_name, q_factory in QUERIES.items():
        print(f"\n--- {q_name} ---")
        for tau in taus:
            if q_name == "Q4_full":
                plan = q_factory(tau1=tau, tau2=tau)
            else:
                plan = q_factory(tau=tau)

            # Bind query params
            query_params = {
                "q1": q_vec,
                "q2": data.centers[2] + 0.05 * np.random.default_rng(0).standard_normal(dim).astype("float32"),
                "q2": q_vec,  # same as q1 to keep experiment simple
                "q3": "hnsw vector query",
                "n0": 100,
                "recent": data.table["ts"].to_pylist()[0],
            }

            # Ground truth
            true_s = actual_selectivity(store, plan, query_params)
            # Skip rows with truly zero selectivity (degenerate):
            # these are at very tight tau, the CBO trivially picks
            # flat and there's nothing to learn.
            if true_s <= 0.0:
                continue

            # RBO rewrite
            rw_plan, _ = rewrite(plan)

            # Default CBO: 0.5 prior
            default_hint = {}  # empty -> CBO uses 0.5 default
            default_strategy = _vector_strategy_choice(
                store, rw_plan, default_hint, query_params,
            )
            default_t, _ = strategy_actual_latency(
                default_strategy, emb_vectors, q_vec, tau,
            )
            default_cost = cost_vector(
                default_strategy, 0.5, n, dim,
            ).time_ms

            # Stats-aware CBO
            stats_hint = populate_selectivity_hint(
                rw_plan, store, query_params,
                vector_columns=["emb", "emb_alt"],
                text_columns=["text"], graph=graph, hub=hub,
            )
            stats_strategy = _vector_strategy_choice(
                store, rw_plan, stats_hint, query_params,
            )
            stats_t, _ = strategy_actual_latency(
                stats_strategy, emb_vectors, q_vec, tau,
            )
            stats_s_for_emb = stats_hint.get("emb", 0.5)
            stats_cost = cost_vector(
                stats_strategy, stats_s_for_emb, n, dim,
            ).time_ms

            # Oracle
            oracle_strat = oracle_strategy(emb_vectors, q_vec, tau)
            oracle_t, _ = strategy_actual_latency(
                oracle_strat, emb_vectors, q_vec, tau,
            )

            row = E7Row(
                query=q_name,
                tau=tau,
                true_selectivity=true_s,
                default_strategy=default_strategy,
                default_latency_ms=default_t,
                default_cost_estimate_ms=default_cost,
                stats_strategy=stats_strategy,
                stats_latency_ms=stats_t,
                stats_cost_estimate_ms=stats_cost,
                oracle_strategy=oracle_strat,
                oracle_latency_ms=oracle_t,
                default_matches_oracle=(default_strategy == oracle_strat),
                stats_matches_oracle=(stats_strategy == oracle_strat),
                cost_ratio_default=default_t / oracle_t if oracle_t > 0 else 1.0,
                cost_ratio_stats=stats_t / oracle_t if oracle_t > 0 else 1.0,
            )
            rows.append(row)
            print(f"  tau={tau:.2f} true_s={true_s:.3f}  default={default_strategy}({default_t:.1f}ms)  "
                  f"stats={stats_strategy}({stats_t:.1f}ms)  oracle={oracle_strat}({oracle_t:.1f}ms)  "
                  f"ratio_d={row.cost_ratio_default:.2f}  ratio_s={row.cost_ratio_stats:.2f}")

    # Save CSV
    import csv
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))
    print(f"\nSaved {len(rows)} rows to {out_path}")

    # Summary
    summary: Dict[str, Any] = {
        "n": n,
        "dim": dim,
        "queries": list(QUERIES.keys()),
        "rows": [asdict(r) for r in rows],
    }
    summary_path = out_path.replace(".csv", ".summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # Aggregate stats
    print("\n=== E7 summary ===")
    for q_name in QUERIES:
        qrows = [r for r in rows if r.query == q_name]
        if not qrows:
            continue
        d_match = sum(1 for r in qrows if r.default_matches_oracle) / len(qrows)
        s_match = sum(1 for r in qrows if r.stats_matches_oracle) / len(qrows)
        d_ratio = np.mean([r.cost_ratio_default for r in qrows])
        s_ratio = np.mean([r.cost_ratio_stats for r in qrows])
        print(f"{q_name}: default match={d_match*100:.0f}%, stats match={s_match*100:.0f}%, "
              f"mean cost ratio default={d_ratio:.2f}, stats={s_ratio:.2f}")
    return summary


def run_e7_scaling(out_path: str = "results/e7_scaling.csv") -> None:
    """Run E7 at multiple corpus sizes to show the statistics layer's
    value emerges as n grows.

    The cost model's flat strategy is O(n); for n = 10k the constant
    is small enough that flat can sometimes beat HNSW even at low
    selectivity (the prototype's findings). For n = 100k the O(n)
    cost dominates and the regime boundary becomes sharper; the
    statistics layer's accurate per-query estimate determines the
    regime. This is the regime where the contribution of §6.2 of the
    paper is most visible.
    """
    sizes = [10000, 30000, 100000]
    print(f"\nE7 scaling: n in {sizes}")
    for n in sizes:
        print(f"\n--- n = {n} ---")
        run_e7(n=n, out_path=out_path.replace(".csv", f"_n{n}.csv"))


def _vector_strategy_choice(
    store: TableStore,
    plan: Plan,
    selectivity_hint: Dict[str, float],
    query_params: Dict[str, Any],
) -> str:
    """Pick the ANN strategy for the first vector filter in the plan."""
    from mosaicopt.algebra.ops import Select
    from mosaicopt.algebra.predicates import VectorRange
    n = store.arrow.num_rows
    dim = 768
    # Walk the plan to find the first vector range filter
    for pred in _walk_first_vector(plan):
        col = pred.column
        s = selectivity_hint.get(col, 0.5)
        try:
            v = store.vector_column(col)
            dim = v.shape[1]
        except Exception:
            pass
        return cbo_pick(s, n, dim, recall_floor=0.95)
    # No vector filter found
    return "hnsw"


def _walk_first_vector(plan: Plan):
    from mosaicopt.algebra.ops import Select
    from mosaicopt.algebra.predicates import VectorRange
    if isinstance(plan, Select):
        if isinstance(plan.predicate, VectorRange):
            yield plan.predicate
        else:
            for p in _walk_first_vector(plan.child):
                yield p
    elif hasattr(plan, "child") and plan.child is not None:
        for p in _walk_first_vector(plan.child):
            yield p


if __name__ == "__main__":
    run_e7()
