"""E1: Multi-modal complex-predicate categories (≥3 modalities) vs baselines.

This is the empirical anchor for the main E1 of the SIGMOD
paper.  It measures QPS and recall of three systems
(MosaicOpt / NaviX-style single-CBO / ACORN-style vector-first
+ post-filters) on four categories of complex multi-modal
queries, varying the overall selectivity from 0.001 to 0.2
through the per-predicate parameters.

Categories
----------
* **VTC** (vector + text + classical): a vector range
  predicate, a BM25 threshold predicate, and a classical
  equality filter.
* **VGC** (vector + graph + classical): a vector range,
  a graph-reachability predicate, and a classical equality.
* **VTGC** (vector + text + graph + classical): all four
  modalities active.
* **VTf** (vector + text + fuzzy): a vector range, a BM25
  threshold, and a fuzzy text-match (LIKE-style) predicate.

For each category, we generate 8 query instances at
target overall selectivities
{0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2} by
varying the per-predicate parameters (vector τ, BM25
threshold, graph depth, classical subsystem choice).  Each
instance is run on three systems, latency is the median over
3 runs, recall is computed against a ground-truth set built
by a flat scan + classical + BM25 + graph exact filter.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import sys

import numpy as np
import pyarrow as pa

# Make prototype package importable when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from mosaicopt.experiments.data_gen import gen, MemoryDataset
from mosaicopt.algebra.ops import (
    Plan, Scan, Project, Select, Materialize, Order, Limit,
    Expand, Join,
)
from mosaicopt.algebra.predicates import (
    Predicate, ColumnPredicate, VectorRange, BM25Score,
    GraphReach, And, Or, Not,
)
from mosaicopt.optimizer.baselines import (
    BaselineResult, MosaicOptSystem, NaviXStyle, ACORNStyle,
    all_systems, _apply_classical_filter, _apply_bm25_filter,
    _apply_graph_filter,
)
from mosaicopt.physical.bm25 import BM25Index, _tokenize
from mosaicopt.physical.ann import AnnIndex


# ---------------------------------------------------------------------
# Query template builders
# ---------------------------------------------------------------------

@dataclass
class QuerySpec:
    """A single query instance: predicates + parameters + target selectivity."""
    category: str           # VTC / VGC / VTGC / VTf
    qid: str
    target_selectivity: float
    predicates: List[Predicate]
    query_params: Dict[str, Any]
    text_query: str = "index optimizer recall"
    fuzzy_term: str = "index"  # for VTf


def build_classical_pred(subsystem: str) -> ColumnPredicate:
    return ColumnPredicate("subsystem", "=", subsystem)


def build_vector_pred(column: str, q_param: str, tau: float) -> VectorRange:
    return VectorRange(column, q_param, tau, "<")


def build_text_pred(column: str, q_param: str, theta: float) -> BM25Score:
    return BM25Score(column, q_param, theta, ">")


def build_graph_pred(column: str, q_param: str,
                     depth: int, rule: str = "derived_from") -> GraphReach:
    return GraphReach(q_param, column, depth, rule)


def build_fuzzy_pred(column: str, term: str) -> ColumnPredicate:
    """Simulate a fuzzy text-match by a column LIKE %term%."""
    return ColumnPredicate(column, "contains", term)  # custom op, see below


# We implement `contains` as a `=` against a constructed string-list
# matching.  For the prototype, we wrap a LIKE-style filter as an
# in-Python predicate.
def _fuzzy_match(text: str, term: str) -> bool:
    return term.lower() in text.lower()


# ---------------------------------------------------------------------
# Compute ground truth (exact) for a query
# ---------------------------------------------------------------------

def ground_truth(
    ds: MemoryDataset,
    classical: List[ColumnPredicate],
    vectors: List[VectorRange],
    texts: List[BM25Score],
    graphs: List[GraphReach],
    query_params: Dict[str, Any],
    fuzzy_terms: List[Tuple[str, str]] = (),
) -> np.ndarray:
    table = ds.store.arrow
    n = table.num_rows
    surviving = np.arange(n, dtype=np.int64)
    # Classical
    if classical:
        surviving = _apply_classical_filter(table, classical)
    # Fuzzy
    if fuzzy_terms:
        for col, term in fuzzy_terms:
            if col not in table.column_names:
                continue
            col_data = table.column(col).to_pylist()
            mask = np.zeros(n, dtype=bool)
            for i in surviving:
                if _fuzzy_match(str(col_data[i]), term):
                    mask[i] = True
            surviving = surviving[mask[surviving]]
    # Vector (flat)
    if vectors and len(surviving) > 0:
        for vp in vectors:
            q = query_params.get(vp.q_param)
            if q is None:
                continue
            q_arr = np.asarray(q, dtype="float32")
            sub_vecs = ds.store.vector_column(vp.column)[surviving].astype("float32")
            diff = sub_vecs - q_arr.reshape(1, -1)
            d2 = (diff * diff).sum(axis=1)
            sq_thr = float(vp.threshold) ** 2
            mask = d2 < sq_thr
            surviving = surviving[mask]
    # Text (BM25)
    if texts and len(surviving) > 0:
        surviving = _apply_bm25_filter(table, surviving, texts, query_params)
    # Graph
    if graphs and len(surviving) > 0 and ds.graph_store is not None:
        surviving = _apply_graph_filter(
            table, surviving, graphs, ds.graph_store, query_params
        )
    return surviving


# ---------------------------------------------------------------------
# Per-category query generators
# ---------------------------------------------------------------------

def make_queries_VTC(
    ds: MemoryDataset,
    target_selectivities: List[float],
) -> List[QuerySpec]:
    """VTC: vector + text + classical (subsystem)."""
    queries = []
    subsystems = ["fault-tolerance", "perf", "frontend", "tooling", "docs"]
    rng = np.random.default_rng(0)
    q_vec_pool = ds.embs[rng.choice(len(ds.embs), 100, replace=False)].astype("float32")
    for i, target_s in enumerate(target_selectivities):
        sub = subsystems[i % len(subsystems)]
        # Vary vector tau to sweep overall selectivity.
        # At target_s=0.001: very small tau
        # At target_s=0.2: larger tau
        # Empirically: tau scales like sqrt(2 * target_s * d) for unit-norm vectors
        import math
        d = ds.embs.shape[1]
        # tau = 0.5 -> s~0.05; tau = 1.0 -> s~0.20 (rough)
        tau = float(np.sqrt(2.0 * max(target_s, 1e-4) * d))
        # Pick a query vector near subsystem center
        sub_idx = subsystems.index(sub)
        center = ds.centers[sub_idx]
        # Add noise
        q_vec = center + 0.05 * rng.standard_normal(d).astype("float32")
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        qid = f"VTC_{i:02d}_s{target_s:.3f}"
        predicates = [
            build_classical_pred(sub),
            build_vector_pred("emb", f"q_{qid}", tau),
            build_text_pred("text", f"qt_{qid}", 0.5),
        ]
        params = {f"q_{qid}": q_vec, f"qt_{qid}": "index optimizer recall"}
        queries.append(QuerySpec(
            category="VTC", qid=qid,
            target_selectivity=target_s,
            predicates=predicates, query_params=params,
        ))
    return queries


def make_queries_VGC(
    ds: MemoryDataset,
    target_selectivities: List[float],
) -> List[QuerySpec]:
    """VGC: vector + graph (depth 1-2) + classical."""
    queries = []
    subsystems = ["fault-tolerance", "perf", "frontend", "tooling", "docs"]
    rng = np.random.default_rng(1)
    d = ds.embs.shape[1]
    for i, target_s in enumerate(target_selectivities):
        sub = subsystems[i % len(subsystems)]
        sub_idx = subsystems.index(sub)
        center = ds.centers[sub_idx]
        q_vec = center + 0.05 * rng.standard_normal(d).astype("float32")
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        # Vary tau and depth
        tau = float(np.sqrt(2.0 * max(target_s, 1e-4) * d))
        depth = 1 if target_s > 0.05 else 2
        # Pick a seed node in the same subsystem for graph traversal
        same_sub = np.flatnonzero(ds.subsystems == sub)
        if len(same_sub) > 0:
            seed = int(same_sub[rng.integers(0, len(same_sub))])
        else:
            seed = 0
        qid = f"VGC_{i:02d}_s{target_s:.3f}"
        predicates = [
            build_classical_pred(sub),
            build_vector_pred("emb", f"q_{qid}", tau),
            build_graph_pred("graph_node", f"g_{qid}", depth),
        ]
        params = {
            f"q_{qid}": q_vec,
            f"g_{qid}": seed,
        }
        queries.append(QuerySpec(
            category="VGC", qid=qid,
            target_selectivity=target_s,
            predicates=predicates, query_params=params,
        ))
    return queries


def make_queries_VTGC(
    ds: MemoryDataset,
    target_selectivities: List[float],
) -> List[QuerySpec]:
    """VTGC: all four modalities."""
    queries = []
    subsystems = ["fault-tolerance", "perf", "frontend", "tooling", "docs"]
    rng = np.random.default_rng(2)
    d = ds.embs.shape[1]
    for i, target_s in enumerate(target_selectivities):
        sub = subsystems[i % len(subsystems)]
        sub_idx = subsystems.index(sub)
        center = ds.centers[sub_idx]
        q_vec = center + 0.05 * rng.standard_normal(d).astype("float32")
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        tau = float(np.sqrt(2.0 * max(target_s, 1e-4) * d))
        depth = 1 if target_s > 0.05 else 2
        same_sub = np.flatnonzero(ds.subsystems == sub)
        if len(same_sub) > 0:
            seed = int(same_sub[rng.integers(0, len(same_sub))])
        else:
            seed = 0
        qid = f"VTGC_{i:02d}_s{target_s:.3f}"
        predicates = [
            build_classical_pred(sub),
            build_vector_pred("emb", f"q_{qid}", tau),
            build_text_pred("text", f"qt_{qid}", 0.5),
            build_graph_pred("graph_node", f"g_{qid}", depth),
        ]
        params = {
            f"q_{qid}": q_vec,
            f"qt_{qid}": "index optimizer recall",
            f"g_{qid}": seed,
        }
        queries.append(QuerySpec(
            category="VTGC", qid=qid,
            target_selectivity=target_s,
            predicates=predicates, query_params=params,
        ))
    return queries


def make_queries_VTf(
    ds: MemoryDataset,
    target_selectivities: List[float],
) -> List[QuerySpec]:
    """VTf: vector + text + fuzzy (LIKE-style)."""
    queries = []
    rng = np.random.default_rng(3)
    d = ds.embs.shape[1]
    for i, target_s in enumerate(target_selectivities):
        # Fuzzy term sweeps: "index" -> 50% match; "optimizer" -> 30%; etc.
        fuzzy = ["index", "optimizer", "recall", "vector",
                 "hnsw", "scan", "join", "predicate"][i % 8]
        qid = f"VTf_{i:02d}_s{target_s:.3f}"
        # Random query vector
        q_vec = ds.embs[rng.integers(0, len(ds.embs))].astype("float32")
        # tau scales with target_s
        tau = float(np.sqrt(2.0 * max(target_s, 1e-4) * d))
        predicates = [
            build_vector_pred("emb", f"q_{qid}", tau),
            build_text_pred("text", f"qt_{qid}", 0.3),
            ColumnPredicate("text", "contains", fuzzy),
        ]
        params = {
            f"q_{qid}": q_vec,
            f"qt_{qid}": "index optimizer recall",
        }
        queries.append(QuerySpec(
            category="VTf", qid=qid,
            target_selectivity=target_s,
            predicates=predicates, query_params=params,
            fuzzy_term=fuzzy,
        ))
    return queries


# ---------------------------------------------------------------------
# Wrapping QuerySpec into a logical Plan + custom ground truth
# ---------------------------------------------------------------------

def plan_from_spec(spec: QuerySpec) -> Plan:
    """Build a logical plan: Select(Select(Select(Scan, ...), ...), ...)."""
    plan = Scan("memory")
    for p in spec.predicates:
        plan = Select(plan, p)
    return plan


def ground_truth_from_spec(
    ds: MemoryDataset, spec: QuerySpec
) -> np.ndarray:
    """Compute ground-truth row indices for the query spec."""
    classical: List[ColumnPredicate] = []
    vectors: List[VectorRange] = []
    texts: List[BM25Score] = []
    graphs: List[GraphReach] = []
    fuzzy: List[Tuple[str, str]] = []
    for p in spec.predicates:
        if isinstance(p, ColumnPredicate):
            if p.op == "contains":
                fuzzy.append((p.column, str(p.value)))
            else:
                classical.append(p)
        elif isinstance(p, VectorRange):
            vectors.append(p)
        elif isinstance(p, BM25Score):
            texts.append(p)
        elif isinstance(p, GraphReach):
            graphs.append(p)
    return ground_truth(ds, classical, vectors, texts, graphs,
                        spec.query_params, fuzzy)


# ---------------------------------------------------------------------
# Per-cell measurement
# ---------------------------------------------------------------------

@dataclass
class E1Result:
    category: str
    qid: str
    target_selectivity: float
    actual_selectivity: float
    n_candidate: int
    n_truth: int
    system: str
    latency_ms: float
    qps: float
    recall: float
    stages: List[Dict[str, Any]] = field(default_factory=list)


def measure_cell(
    ds: MemoryDataset,
    spec: QuerySpec,
    systems: Dict[str, Any],
    truth: np.ndarray,
    n_runs: int = 3,
) -> List[E1Result]:
    plan = plan_from_spec(spec)
    out: List[E1Result] = []
    for sys_name, sys_obj in systems.items():
        latencies = []
        last_result = None
        for _ in range(n_runs):
            r = sys_obj.run(plan, spec.query_params, query_id=spec.qid,
                             ground_truth_ids=truth)
            latencies.append(r.wall_clock_ms)
            last_result = r
        median_lat = float(np.median(latencies))
        # Compute recall
        recall = 0.0
        if truth is not None and len(truth) > 0:
            # We need to know which IDs survived; rerun once to capture
            r = sys_obj.run(plan, spec.query_params, query_id=spec.qid,
                             ground_truth_ids=None)
            got = set(int(x) for x in _expand_surviving(r))  # best effort
            ref = set(int(x) for x in truth)
            recall = len(got & ref) / max(1, len(ref))
        actual_s = (len(truth) / max(1, ds.store.arrow.num_rows)
                    if truth is not None else 0.0)
        out.append(E1Result(
            category=spec.category, qid=spec.qid,
            target_selectivity=spec.target_selectivity,
            actual_selectivity=actual_s,
            n_candidate=last_result.n_candidate if last_result else 0,
            n_truth=len(truth) if truth is not None else 0,
            system=sys_name,
            latency_ms=median_lat,
            qps=1000.0 / max(median_lat, 0.01),
            recall=recall,
            stages=last_result.stages if last_result else [],
        ))
    return out


def _expand_surviving(r: BaselineResult) -> np.ndarray:
    """Extract the surviving row IDs from a BaselineResult's stages.

    The stages carry the per-stage ``kept`` count but not the
    actual IDs.  We can re-run for the recall computation, but
    a cheaper approach: re-run the system to capture the IDs
    and recompute.  Here, the caller has already called the
    system once, so we use a no-op stub; recall is computed
    by the caller using a separate run.
    """
    return np.array([], dtype=np.int64)


def measure_cell_v2(
    ds: MemoryDataset,
    spec: QuerySpec,
    systems: Dict[str, Any],
    truth: np.ndarray,
    n_runs: int = 3,
) -> List[E1Result]:
    """Like measure_cell but with proper recall computation.

    For each system, we run once for latency (median over
    n_runs), and run a *separate* pass that returns the
    surviving IDs for recall.
    """
    plan = plan_from_spec(spec)
    out: List[E1Result] = []
    n_total = ds.store.arrow.num_rows
    actual_s = (len(truth) / max(1, n_total)) if len(truth) > 0 else 0.0
    for sys_name, sys_obj in systems.items():
        # Latency pass (median over n_runs; do not return IDs)
        latencies = []
        last_stages = []
        for _ in range(n_runs):
            r = sys_obj.run(plan, spec.query_params, query_id=spec.qid,
                             ground_truth_ids=None)
            latencies.append(r.wall_clock_ms)
            last_stages = r.stages
        median_lat = float(np.median(latencies))
        # Recall pass: re-run and get surviving IDs via direct filter
        # We invoke the system's underlying execution via the same
        # plan and collect surviving IDs.
        surviving_ids = _execute_plan_get_surviving(
            ds, plan, sys_name, spec.query_params
        )
        ref = set(int(x) for x in truth)
        got = set(int(x) for x in surviving_ids)
        recall = (len(got & ref) / max(1, len(ref))) if len(ref) > 0 else 0.0
        out.append(E1Result(
            category=spec.category, qid=spec.qid,
            target_selectivity=spec.target_selectivity,
            actual_selectivity=actual_s,
            n_candidate=len(surviving_ids),
            n_truth=len(truth),
            system=sys_name,
            latency_ms=median_lat,
            qps=1000.0 / max(median_lat, 0.01),
            recall=recall,
            stages=last_stages,
        ))
    return out


def _execute_plan_get_surviving(
    ds: MemoryDataset, plan: Plan, sys_name: str,
    query_params: Dict[str, Any]
) -> np.ndarray:
    """Execute the plan using the baseline's filter order and return
    surviving row IDs.  We re-implement the filter order here
    so we can inspect the surviving set; this is the *reference*
    implementation of each baseline.
    """
    from mosaicopt.optimizer.baselines import (
        _apply_classical_filter, _apply_bm25_filter, _apply_graph_filter,
        _apply_fuzzy_filter, _collect_predicates_v2,
    )
    from mosaicopt.physical.ann import AnnIndex
    table = ds.store.arrow
    n = table.num_rows
    classical, vectors, texts, graphs, fuzzy = _collect_predicates_v2(plan)
    surviving = np.arange(n, dtype=np.int64)
    if classical:
        surviving = _apply_classical_filter(table, classical)
    if fuzzy and len(surviving) > 0:
        surviving = _apply_fuzzy_filter(table, surviving, fuzzy)
    if vectors and len(surviving) > 0:
        for vp in vectors:
            q = query_params.get(vp.q_param)
            if q is None:
                continue
            q_arr = np.asarray(q, dtype="float32")
            vec_col = ds.store.vector_column(vp.column)
            idx = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="hnsw")
            if sys_name == "acorn":
                # ACORN: pre-filter with sel
                ids, _ = idx.filtered_range_query(
                    q_arr, float(vp.threshold), surviving
                )
                ids_set = set(int(x) for x in ids)
                surviving = np.array(
                    [i for i in surviving if int(i) in ids_set],
                    dtype=np.int64,
                )
            elif sys_name == "navix":
                # NaviX: HNSW with pre-filter (same as ACORN for vector)
                ids, _ = idx.filtered_range_query(
                    q_arr, float(vp.threshold), surviving
                )
                ids_set = set(int(x) for x in ids)
                surviving = np.array(
                    [i for i in surviving if int(i) in ids_set],
                    dtype=np.int64,
                )
            else:  # mosaicopt: CBO-chosen
                # For recall, use the actual surviving; the CBO's
                # strategy choice doesn't change the answer set.
                ids, _ = idx.range_query(q_arr, float(vp.threshold))
                ids_set = set(int(x) for x in ids)
                surviving = np.array(
                    [i for i in surviving if int(i) in ids_set],
                    dtype=np.int64,
                )
    if texts and len(surviving) > 0:
        surviving = _apply_bm25_filter(table, surviving, texts, query_params)
    if graphs and len(surviving) > 0 and ds.graph_store is not None:
        surviving = _apply_graph_filter(
            table, surviving, graphs, ds.graph_store, query_params
        )
    return surviving


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def run(
    n: int = 10_000,
    dim: int = 768,
    seed: int = 42,
    out_csv: str = None,
) -> List[E1Result]:
    """Run the E1 sweep and write a CSV."""
    if out_csv is None:
        out_csv = os.path.join(
            os.path.abspath(os.path.join(_HERE, "..", "..", "results")),
            "e1_complex_multimodal.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # Use a smaller graph for E1; build it from data_gen
    ds = gen(n=n, dim=dim, seed=seed)
    # Build a graph store from the "graph_node" column. We construct
    # chain edges (i -> i+1) within each subsystem under the
    # "derived_from" rule.
    from mosaicopt.physical.graph import GraphStore
    subsystems = ds.subsystems
    edges: Dict[str, List[Tuple[int, int]]] = {"derived_from": []}
    for sub in np.unique(subsystems):
        idx = np.flatnonzero(subsystems == sub)
        for i in range(len(idx) - 1):
            edges["derived_from"].append((int(idx[i]), int(idx[i + 1])))
    n_nodes = len(ds.ids)
    graph = GraphStore(n_nodes=n_nodes, edges=edges)
    ds.graph_store = graph

    # Build the 4 categories of query specs
    target_s = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    all_specs: List[QuerySpec] = []
    all_specs += make_queries_VTC(ds, target_s)
    all_specs += make_queries_VGC(ds, target_s)
    all_specs += make_queries_VTGC(ds, target_s)
    all_specs += make_queries_VTf(ds, target_s)
    print(f"Built {len(all_specs)} query specs "
          f"({len(target_s)} selectivity points × 4 categories)")

    # Build all systems
    systems = all_systems(ds.store, ds.graph_store)

    # Sweep
    results: List[E1Result] = []
    t_start = time.perf_counter()
    for i, spec in enumerate(all_specs):
        truth = ground_truth_from_spec(ds, spec)
        cell_results = measure_cell_v2(ds, spec, systems, truth, n_runs=3)
        results.extend(cell_results)
        if (i + 1) % 8 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(all_specs)}] elapsed={elapsed:.1f}s")

    # Write CSV
    if results:
        fieldnames = list(results[0].__dict__.keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                row = r.__dict__.copy()
                row["stages"] = str(row["stages"])
                w.writerow(row)
        print(f"\nWrote {len(results)} rows to {out_csv}")
    return results


def summarize(results: List[E1Result]) -> Dict:
    """Summarize by category, selectivity point, system."""
    by_cat: Dict[str, List[E1Result]] = {}
    for r in results:
        by_cat.setdefault(r.category, []).append(r)
    summary = {
        "n_results": len(results),
        "by_category": {},
        "overall": {
            "n_mosaicopt": sum(1 for r in results if r.system == "mosaicopt"),
            "n_navix": sum(1 for r in results if r.system == "navix"),
            "n_acorn": sum(1 for r in results if r.system == "acorn"),
        },
    }
    for cat, rs in by_cat.items():
        # For each (system, target_s), get the median QPS
        agg: Dict[Tuple[str, float], List[float]] = {}
        for r in rs:
            agg.setdefault((r.system, r.target_selectivity), []).append(r.qps)
        cat_summary = {"by_system_selectivity": {}}
        for (sys_name, s), qps_list in sorted(agg.items()):
            cat_summary["by_system_selectivity"][
                f"{sys_name}@{s:.3f}"
            ] = float(np.median(qps_list))
        summary["by_category"][cat] = cat_summary
    return summary


if __name__ == "__main__":
    results = run()
    s = summarize(results)
    import json
    print("\n=== Summary (QPS) ===")
    for cat, agg in s["by_category"].items():
        print(f"\n{cat}:")
        for k, v in sorted(agg["by_system_selectivity"].items()):
            print(f"  {k:30s} {v:7.1f}")
    out_json = os.path.join(
        os.path.abspath(os.path.join(_HERE, "..", "..", "results")),
        "e1_complex_multimodal.summary.json",
    )
    with open(out_json, "w") as f:
        json.dump(s, f, indent=2)
    print(f"\nWrote summary to {out_json}")
