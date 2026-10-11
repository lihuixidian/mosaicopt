"""E3: RBO ablation (with-RBO vs no-RBO).

Compares MosaicOpt with the full RBO (rewrites enabled) against
MosaicOpt with RBO disabled (no predicate pushdown, no
Materialize-then-Filter, no IndexPrune).  The point is to measure
how much of the latency win comes from the RBO layer.

We use the E1 query set (4 categories × 8 selectivity points).
For each cell, we run both variants and report QPS and recall.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from mosaicopt.experiments.data_gen import gen, MemoryDataset
from mosaicopt.algebra.ops import Plan, Scan, Select
from mosaicopt.algebra.predicates import (
    Predicate, ColumnPredicate, VectorRange, BM25Score, GraphReach,
)
from mosaicopt.optimizer.baselines import (
    BaselineResult, _apply_classical_filter, _apply_bm25_filter,
    _apply_graph_filter, _apply_fuzzy_filter, _collect_predicates_v2,
)
from mosaicopt.optimizer.rbo import rewrite as optimize_plan
from mosaicopt.optimizer.plan import to_physical
from mosaicopt.optimizer.cbo import cbo_pick
from mosaicopt.optimizer.statistics import (
    populate_selectivity_hint, populate_classical_s_hint,
    populate_rho_hint, StatisticsHub, FilteredHNSWDensity,
)
from mosaicopt.physical.ann import AnnIndex
from mosaicopt.physical.graph import GraphStore
from mosaicopt.experiments.e1_complex_multimodal import (
    QuerySpec, plan_from_spec, ground_truth_from_spec, measure_cell_v2,
    make_queries_VTC, make_queries_VGC, make_queries_VTGC, make_queries_VTf,
)


# ---------------------------------------------------------------------
# MosaicOpt variant: with RBO and without RBO
# ---------------------------------------------------------------------

class MosaicOptWithRBO:
    """MosaicOpt with the full RBO rewrites enabled (default)."""

    def __init__(self, store, graph=None):
        self.store = store
        self.graph = graph

    def run(self, plan, query_params=None, query_id="q", ground_truth_ids=None):
        query_params = query_params or {}
        # RBO step
        try:
            plan = optimize_plan(plan, self.store, query_params=query_params)
        except Exception:
            pass
        return _execute_mosaicopt(
            self.store, self.graph, plan, query_params,
            query_id, ground_truth_ids, label="with_rbo",
        )


class MosaicOptWithoutRBO:
    """MosaicOpt with RBO disabled (no rewrites)."""

    def __init__(self, store, graph=None):
        self.store = store
        self.graph = graph

    def run(self, plan, query_params=None, query_id="q", ground_truth_ids=None):
        query_params = query_params or {}
        return _execute_mosaicopt(
            self.store, self.graph, plan, query_params,
            query_id, ground_truth_ids, label="no_rbo",
        )


def _execute_mosaicopt(store, graph, plan, query_params, query_id, ground_truth_ids, label):
    """Shared execution: classical pre-filter → CBO vector → BM25 → graph.

    The only difference between with-RBO and no-RBO is the plan they
    receive: with-RBO goes through optimize_plan() (which applies
    redexes), no-RBO uses the original plan.
    """
    table = store.arrow
    n_total = table.num_rows
    sel_hint = {}
    cs_hint = populate_classical_s_hint(plan, store=store)
    rho_hint = {}
    try:
        hub = StatisticsHub.from_store(store, [], [], graph=graph, sample_rate=0.0)
        sel_hint = populate_selectivity_hint(plan, store, query_params, hub=hub)
    except Exception:
        sel_hint = {}
    try:
        fd = FilteredHNSWDensity(store, vector_columns=[], n_sample_queries=5, seed=0)
        rho_hint = populate_rho_hint(plan, fd, store)
    except Exception:
        rho_hint = {}
    t0 = time.perf_counter()
    try:
        phys = to_physical(
            plan, store,
            selectivity_hint=sel_hint or None,
            rho_hint=rho_hint or None,
            classical_s_hint=cs_hint or None,
        )
    except Exception:
        t = (time.perf_counter() - t0) * 1000.0
        return BaselineResult(
            system=label, query_id=query_id,
            n_input=n_total, n_predicate_selectivity=0.0,
            n_candidate=0, wall_clock_ms=t, recall=0.0,
            extra={"error": "to_physical failed"},
        )
    classical, vectors, texts, graphs, fuzzy = _collect_predicates_v2(plan)
    surviving_ids = np.arange(n_total, dtype=np.int64)
    stages = []
    if classical:
        t1 = time.perf_counter()
        surviving_ids = _apply_classical_filter(table, classical)
        t_classical = (time.perf_counter() - t1) * 1000.0
        stages.append({"stage": "classical", "kept": len(surviving_ids),
                       "ms": t_classical})
    if fuzzy and len(surviving_ids) > 0:
        t1 = time.perf_counter()
        surviving_ids = _apply_fuzzy_filter(table, surviving_ids, fuzzy)
        t_fz = (time.perf_counter() - t1) * 1000.0
        stages.append({"stage": f"fuzzy({len(fuzzy)} preds)",
                       "kept": len(surviving_ids), "ms": t_fz})
    if vectors and len(surviving_ids) > 0:
        t1 = time.perf_counter()
        for vp in vectors:
            q = query_params.get(vp.q_param)
            if q is None:
                continue
            q_arr = np.asarray(q, dtype="float32")
            from mosaicopt.optimizer.baselines import s_v_estimate
            s = float(s_v_estimate(store, surviving_ids, vp, q_arr))
            rho = float(rho_hint.get(vp.column, 1.0))
            cs = 1.0
            if surviving_ids.size > 0:
                cs = len(surviving_ids) / max(1, n_total)
            strategy = cbo_pick(s, n_total, 128, rho=rho, classical_s=cs)
            vec_col = store.vector_column(vp.column)
            if strategy == "hnsw":
                hnsw = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="hnsw")
                new_ids, _ = hnsw.filtered_range_query(
                    q_arr, float(vp.threshold), surviving_ids
                )
            elif strategy == "ivf":
                ivf = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="ivf")
                new_ids, _ = ivf.range_query(q_arr, float(vp.threshold))
            else:
                flat = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="flat")
                new_ids, _ = flat.range_query(q_arr, float(vp.threshold))
            new_set = set(int(x) for x in new_ids)
            surviving_ids = np.array(
                [i for i in surviving_ids if int(i) in new_set],
                dtype=np.int64,
            )
        t_vec = (time.perf_counter() - t1) * 1000.0
        stages.append({"stage": f"vector({len(vectors)} preds)",
                       "kept": len(surviving_ids), "ms": t_vec})
    if texts and len(surviving_ids) > 0:
        t1 = time.perf_counter()
        surviving_ids = _apply_bm25_filter(table, surviving_ids, texts, query_params)
        t_txt = (time.perf_counter() - t1) * 1000.0
        stages.append({"stage": f"text({len(texts)} preds)",
                       "kept": len(surviving_ids), "ms": t_txt})
    if graphs and len(surviving_ids) > 0 and graph is not None:
        t1 = time.perf_counter()
        surviving_ids = _apply_graph_filter(table, surviving_ids, graphs, graph, query_params)
        t_gr = (time.perf_counter() - t1) * 1000.0
        stages.append({"stage": f"graph({len(graphs)} preds)",
                       "kept": len(surviving_ids), "ms": t_gr})
    t_total = (time.perf_counter() - t0) * 1000.0
    recall = 0.0
    if ground_truth_ids is not None and len(ground_truth_ids) > 0:
        got = set(int(x) for x in surviving_ids)
        ref = set(int(x) for x in ground_truth_ids)
        recall = len(got & ref) / max(1, len(ref))
    sel = (len(ground_truth_ids) / max(1, n_total)) if ground_truth_ids is not None else 0.0
    return BaselineResult(
        system=label, query_id=query_id,
        n_input=n_total, n_predicate_selectivity=sel,
        n_candidate=len(surviving_ids), wall_clock_ms=t_total,
        recall=recall, stages=stages,
    )


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def run(
    n: int = 10_000,
    dim: int = 128,
    seed: int = 42,
    out_csv: str = None,
) -> List[Dict[str, Any]]:
    if out_csv is None:
        out_csv = os.path.join(
            os.path.abspath(os.path.join(_HERE, "..", "..", "results")),
            "e3_rbo_ablation.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    ds = gen(n=n, dim=dim, seed=seed)
    edges = {"derived_from": []}
    for sub in np.unique(ds.subsystems):
        idx = np.flatnonzero(ds.subsystems == sub)
        for i in range(len(idx) - 1):
            edges["derived_from"].append((int(idx[i]), int(idx[i + 1])))
    graph = GraphStore(n_nodes=len(ds.ids), edges=edges)
    ds.graph_store = graph

    target_s = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    all_specs: List[QuerySpec] = []
    all_specs += make_queries_VTC(ds, target_s)
    all_specs += make_queries_VGC(ds, target_s)
    all_specs += make_queries_VTGC(ds, target_s)
    all_specs += make_queries_VTf(ds, target_s)
    print(f"Built {len(all_specs)} query specs (4 categories × 8 selectivity points)")

    systems = {
        "with_rbo": MosaicOptWithRBO(ds.store, ds.graph_store),
        "no_rbo": MosaicOptWithoutRBO(ds.store, ds.graph_store),
    }

    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for i, spec in enumerate(all_specs):
        truth = ground_truth_from_spec(ds, spec)
        cell_results = measure_cell_v2(ds, spec, systems, truth, n_runs=1)
        for r in cell_results:
            rows.append({
                "category": spec.category,
                "qid": spec.qid,
                "target_selectivity": spec.target_selectivity,
                "actual_selectivity": r.actual_selectivity,
                "n_truth": r.n_truth,
                "n_candidate": r.n_candidate,
                "variant": r.system,
                "latency_ms": r.latency_ms,
                "qps": r.qps,
                "recall": r.recall,
            })
        if (i + 1) % 8 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(all_specs)}] elapsed={elapsed:.1f}s")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} rows to {out_csv}")
    return rows


if __name__ == "__main__":
    rows = run()
    print(f"\nE3 done: {len(rows)} cells")
