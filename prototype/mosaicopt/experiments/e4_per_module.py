"""E4 per-module: sweep n to find where each CBO module is effective.

The main E4 (results/e4_cbo_ablation.csv) shows that at n=10^4,
default_0.5 is the fastest and `full` is the slowest — the CBO
modules are overhead at prototype scale.  This experiment sweeps
n to find the regime where each module starts to matter.

Configurations (same as e4_cbo_ablation.py):
  * default_0.5    — no stats, no rho, no adaptive
  * with_stats     — StatisticsHub enabled
  * with_rho       — rho_f estimation enabled
  * with_adaptive  — adaptive policy
  * full           — all modules

For each (n, selectivity) cell, we report the *latency* of each
configuration.  The "effective" scenario for each module is the
(n, selectivity) cell where that configuration is faster than
default_0.5.

Sweep:
  n in {5_000, 10_000, 30_000}  (covers prototype + ~3x scale)
  selectivity in {0.005, 0.05, 0.2, 0.5}
  1 query template (VTC: vector + text + classical)
  5 configurations
  Total: 3 * 4 * 1 * 5 = 60 cells
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
from mosaicopt.algebra.predicates import (
    ColumnPredicate, VectorRange, BM25Score, GraphReach,
)
from mosaicopt.optimizer.baselines import (
    BaselineResult, _apply_classical_filter, _apply_bm25_filter,
    _apply_graph_filter, _apply_fuzzy_filter, _collect_predicates_v2,
)
from mosaicopt.optimizer.plan import to_physical
from mosaicopt.optimizer.cbo import cbo_pick
from mosaicopt.optimizer.statistics import (
    populate_selectivity_hint, populate_classical_s_hint,
    populate_rho_hint, StatisticsHub, FilteredHNSWDensity,
)
from mosaicopt.physical.ann import AnnIndex
from mosaicopt.optimizer.rbo import rewrite as rbo_rewrite
from mosaicopt.physical.graph import GraphStore
from mosaicopt.experiments.e1_complex_multimodal import (
    make_queries_VTC, ground_truth_from_spec, measure_cell_v2,
)
from mosaicopt.experiments.e4_cbo_ablation import (
    MosaicOptVariant, CONFIGS, N0_ADAPTIVE,
)


def run(
    n_values: List[int] = (5_000, 10_000, 30_000),
    selectivities: List[float] = (0.005, 0.05, 0.2, 0.5),
    dim: int = 128,
    seed: int = 42,
    out_csv: str = None,
) -> List[Dict[str, Any]]:
    if out_csv is None:
        out_csv = os.path.join(
            os.path.abspath(os.path.join(_HERE, "..", "..", "results")),
            "e4_per_module.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for n in n_values:
        print(f"\n=== n = {n} ===")
        ds = gen(n=n, dim=dim, seed=seed)
        edges = {"derived_from": []}
        for sub in np.unique(ds.subsystems):
            idx = np.flatnonzero(ds.subsystems == sub)
            for i in range(len(idx) - 1):
                edges["derived_from"].append((int(idx[i]), int(idx[i + 1])))
        graph = GraphStore(n_nodes=len(ds.ids), edges=edges)
        ds.graph_store = graph

        # Build queries at this n
        all_specs = make_queries_VTC(ds, list(selectivities))
        systems = {
            label: MosaicOptVariant(ds.store, ds.graph_store, flags, label)
            for label, flags in CONFIGS
        }
        t_start = time.perf_counter()
        for spec in all_specs:
            truth = ground_truth_from_spec(ds, spec)
            cell_results = measure_cell_v2(ds, spec, systems, truth, n_runs=1)
            for r in cell_results:
                rows.append({
                    "n": n,
                    "qid": spec.qid,
                    "target_selectivity": spec.target_selectivity,
                    "n_truth": r.n_truth,
                    "n_candidate": r.n_candidate,
                    "variant": r.system,
                    "latency_ms": r.latency_ms,
                    "qps": r.qps,
                    "recall": r.recall,
                })
        elapsed = time.perf_counter() - t_start
        print(f"  {len(all_specs)} cells in {elapsed:.1f}s")

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
    print(f"\nE4 per-module: {len(rows)} cells")

    # Per-module summary: for each (n, target_s), which config wins?
    import collections
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r["n"], r["target_selectivity"])].append(r)
    print("\nFastest config per (n, target_s):")
    print("  n        sel     fastest         QPS     |  default_0.5 QPS  | speedup")
    for (n, sel), rs in sorted(by_cell.items()):
        rs_by_var = {r["variant"]: r for r in rs}
        fastest = min(rs, key=lambda r: r["latency_ms"])
        default = rs_by_var.get("default_0.5", {})
        speedup = (default.get("qps", 0) / max(0.1, fastest["qps"]))
        print(f"  {n:6d}  {sel:.3f}  {fastest['variant']:14s}  {fastest['qps']:6.1f}  |  {default.get('qps', 0):6.1f}        |  {speedup:.2f}x")
