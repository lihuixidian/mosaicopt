"""E4 per-module v2: pure vector query (no BM25, no graph) at multiple n.

This is a focused scenario where the CBO's strategy choice is the
dominant factor: a single vector range query.  At n=10^4 the CBO
modules are overhead; at n=10^5+ the cost model's regime boundary
starts to be correct; at n=10^6+ the CBO matters.

Sweep:
  n in {10_000, 30_000, 100_000}
  selectivity in {0.005, 0.05, 0.2}
  5 configurations (default_0.5, with_stats, with_rho, with_adaptive, full)
  Total: 3 * 3 * 5 = 45 cells
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from mosaicopt.experiments.data_gen import gen, MemoryDataset
from mosaicopt.algebra.ops import Plan, Scan, Select
from mosaicopt.algebra.predicates import VectorRange
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
from mosaicopt.optimizer.baselines import s_v_estimate
from mosaicopt.experiments.e4_cbo_ablation import (
    MosaicOptVariant, CONFIGS, N0_ADAPTIVE,
)


def run(
    n_values=(10_000, 30_000),
    selectivities=(0.005, 0.05, 0.2),
    dim: int = 128,
    seed: int = 42,
    out_csv: str = None,
) -> List[Dict[str, Any]]:
    if out_csv is None:
        out_csv = os.path.join(
            os.path.abspath(os.path.join(_HERE, "..", "..", "results")),
            "e4_per_module_pure_vector.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for n in n_values:
        print(f"\n=== n = {n} (pure vector) ===")
        ds = gen(n=n, dim=dim, seed=seed)
        systems = {
            label: MosaicOptVariant(ds.store, None, flags, label)
            for label, flags in CONFIGS
        }
        rng = np.random.default_rng(0)
        d = ds.embs.shape[1]
        for i, sel in enumerate(selectivities):
            tau = float(np.sqrt(2.0 * max(sel, 1e-4) * d))
            q_vec = ds.embs[rng.integers(0, len(ds.embs))].astype("float32")
            qid = f"V_{i:02d}_s{sel:.3f}"
            plan = Select(Scan("memory"),
                          VectorRange("emb", f"q_{qid}", tau, "<"))
            params = {f"q_{qid}": q_vec}
            for sys_name, sys_obj in systems.items():
                latencies = []
                for _ in range(3):
                    r = sys_obj.run(plan, params, query_id=qid,
                                     ground_truth_ids=None)
                    latencies.append(r.wall_clock_ms)
                med_lat = float(np.median(latencies))
                rows.append({
                    "n": n,
                    "qid": qid,
                    "target_selectivity": sel,
                    "variant": sys_name,
                    "latency_ms": med_lat,
                    "qps": 1000.0 / max(med_lat, 0.01),
                })
                print(f"  {qid} {sys_name:14s} lat={med_lat:7.2f}ms")
        elapsed_so_far = time.time()
    if rows:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} rows to {out_csv}")
    return rows


if __name__ == "__main__":
    import collections
    t0 = time.perf_counter()
    rows = run(n_values=(10_000, 30_000),
               selectivities=(0.005, 0.05, 0.2),
               out_csv=r'E:\CloudStation\MosaicOpt\prototype\results\e4_per_module_pure_vector.csv')
    print(f"\nelapsed: {time.perf_counter()-t0:.1f}s, n_rows={len(rows)}")
    print("\nFastest config per (n, sel):")
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r['n'], r['target_selectivity'])].append(r)
    for (n, sel), rs in sorted(by_cell.items()):
        rs_by_var = {r['variant']: r for r in rs}
        fastest = min(rs, key=lambda r: r['latency_ms'])
        default = rs_by_var.get('default_0.5', {})
        speedup = (default.get('qps', 0) / max(0.1, fastest['qps']))
        print(f"  n={n:6d} sel={sel:.3f}: fastest={fastest['variant']:14s} qps={fastest['qps']:6.1f}, default_0.5 qps={default.get('qps', 0):6.1f}, speedup={speedup:.2f}x")
