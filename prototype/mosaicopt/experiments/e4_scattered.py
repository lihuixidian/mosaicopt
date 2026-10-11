"""E4 scattered: build a VTC query with a scattered classical filter.

The E1 query set uses subsystem_X (a clustered filter with rho=1.0).
This experiment uses a scattered filter (id mod 7 = 0, or
id_random_p1) which has rho_f < 0.3 per the E9 finding.

For these scattered filters, with_rho should demote hnsw to flat
(rho < 0.3 threshold).  The question is: at n=10k, which is
actually faster — the demoted flat (with_rho) or the un-demoted
hnsw (default_0.5)?

Per E8, hnsw is uniformly faster at n=10k.  So with_rho picking
flat would be SLOWER.  This is a calibration issue, not a
structural flaw.

We run 4 (n, scattered_pattern) cells × 5 configs = 20 cells
to verify the demote fires correctly but document the calibration
gap.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from mosaicopt.experiments.data_gen import gen, MemoryDataset
from mosaicopt.algebra.ops import Plan, Scan, Select
from mosaicopt.algebra.predicates import (
    ColumnPredicate, VectorRange, BM25Score,
)
from mosaicopt.optimizer.baselines import (
    BaselineResult, _apply_classical_filter, _apply_bm25_filter,
    _collect_predicates_v2,
)
from mosaicopt.optimizer.cbo import cbo_pick
from mosaicopt.optimizer.statistics import (
    populate_rho_hint, FilteredHNSWDensity,
)
from mosaicopt.physical.ann import AnnIndex
from mosaicopt.experiments.e4_cbo_ablation import (
    MosaicOptVariant, CONFIGS,
)


def run(
    n_values=(10_000, 30_000),
    patterns=("id_mod_7_0", "id_mod_5_0", "id_random_p1"),
    dim: int = 128,
    seed: int = 42,
    out_csv: str = None,
) -> List[Dict[str, Any]]:
    if out_csv is None:
        out_csv = os.path.join(
            os.path.abspath(os.path.join(_HERE, "..", "..", "results")),
            "e4_scattered_filter.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for n in n_values:
        print(f"\n=== n = {n} (scattered filters) ===")
        ds = gen(n=n, dim=dim, seed=seed)
        systems = {
            label: MosaicOptVariant(ds.store, None, flags, label)
            for label, flags in CONFIGS
        }
        rng = np.random.default_rng(0)
        d = ds.embs.shape[1]
        ids = ds.ids
        subsystems = ds.subsystems
        for pi, pattern in enumerate(patterns):
            # Build the filter predicate based on the pattern
            if pattern == "id_mod_7_0":
                # 1/7 selectivity, scattered
                mask = (ids % 7 == 0)
            elif pattern == "id_mod_5_0":
                # 1/5 selectivity, scattered
                mask = (ids % 5 == 0)
            elif pattern == "id_random_p1":
                # 1% random
                rng_local = np.random.default_rng(pi + n)
                keep = rng_local.choice(n, size=max(1, n // 100), replace=False)
                mask = np.zeros(n, dtype=bool)
                mask[keep] = True
            else:
                raise ValueError(pattern)
            # Pick a query vector
            q_vec = ds.embs[rng.integers(0, len(ds.embs))].astype("float32")
            tau = 1.0  # moderate selectivity
            sel_actual = float(mask.sum()) / max(1, n)
            qid = f"{pattern}_n{n}"
            for sys_name, sys_obj in systems.items():
                # We construct a synthetic plan: σ_{id mod p = 0} ∘ σ_{||V-q|| < τ} ∘ Scan
                # We use ColumnPredicate with a custom op "mod" — but the
                # prototype doesn't support it natively.  Instead, we
                # use a pre-filter: surviving_ids is the mask, then
                # vector range on that subset.
                from mosaicopt.optimizer.plan import to_physical
                # Construct a "manual" predicate: ColumnPredicate with op
                # "in" and a frozenset of allowed ids
                allowed = frozenset(int(i) for i in np.flatnonzero(mask))
                pred = ColumnPredicate("id", "in", allowed)
                plan = Select(Scan("memory"), pred)
                # Add a vector range
                plan = Select(plan, VectorRange("emb", f"q_{qid}", tau, "<"))
                params = {f"q_{qid}": q_vec}
                latencies = []
                last = None
                for _ in range(3):
                    r = sys_obj.run(plan, params, query_id=qid,
                                     ground_truth_ids=None)
                    latencies.append(r.wall_clock_ms)
                    last = r
                med_lat = float(np.median(latencies))
                rows.append({
                    "n": n,
                    "pattern": pattern,
                    "qid": qid,
                    "actual_selectivity": sel_actual,
                    "variant": sys_name,
                    "latency_ms": med_lat,
                    "qps": 1000.0 / max(med_lat, 0.01),
                })
                print(f"  {pattern:15s} {sys_name:14s} lat={med_lat:7.2f}ms sel={sel_actual:.3f}")
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
    rows = run()
    print(f"\nelapsed: {time.perf_counter()-t0:.1f}s, n_rows={len(rows)}")
    print("\nFor each (n, pattern), which config is fastest?")
    by_cell = collections.defaultdict(list)
    for r in rows:
        by_cell[(r['n'], r['pattern'])].append(r)
    for (n, p), rs in sorted(by_cell.items()):
        rs_by_var = {r['variant']: r for r in rs}
        fastest = min(rs, key=lambda r: r['latency_ms'])
        default = rs_by_var.get('default_0.5', {})
        speedup = (default.get('qps', 0) / max(0.1, fastest['qps']))
        print(f"  n={n:6d} pattern={p:15s}: fastest={fastest['variant']:14s} qps={fastest['qps']:6.1f}, default_0.5 qps={default.get('qps', 0):6.1f}, speedup={speedup:.2f}x")
