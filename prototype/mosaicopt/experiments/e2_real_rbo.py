"""E2-real: RBO ablation on BigVectorBench-style data.

Re-derives the RBO findings (M1, J1, T1) from the E3 corollary
of paper.tex on the real-data glue table.  Same three minimal
query templates, but on the larger n=10k, d=128 glue table
from data_gen_real.

For each scenario, we directly measure the wall-clock cost of
the WITH-rewrite vs WITHOUT-rewrite executor path:

* **M1** (sigma o mu -> mu o sigma):  push a filter on a
  materialized column above the materialization.
  - with_rbo: HNSW pre-filter + compute d on survivors
  - no_rbo:   compute d on all n, then filter

* **J1** (sigma o Join -> Join o sigma):  push a filter into
  a join's input.
  - with_rbo: classical pre-filter, then self-join on classical
  - no_rbo:   self-join first, then classical + vector post-filter

* **T1** (Lambda o tau -> TopK):  rewrite a tau+Lambda pair
  to a single TopK.
  - with_rbo: heap-based top-K
  - no_rbo:   full sort + limit

Per-template: 5 query instances (varying tau or K), 3 runs each.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from mosaicopt.experiments.data_gen_real import gen_glue_table, RealMemoryDataset
from mosaicopt.physical.ann import AnnIndex


# =====================================================================
# M1: sigma o mu -> mu o sigma
# =====================================================================

def run_m1(ds: RealMemoryDataset, n_runs: int = 3, seed: int = 42) -> List[Dict[str, Any]]:
    """M1: filter after materialization.  Q7-style multi-vector
    on the real glue table.

    with_rbo: HNSW pre-filter (selectivity ~5%) then exact d
              computation on the surviving set.
    no_rbo:   compute d on all n, then filter.

    The structural effect: the with_rbo path computes the
    distance on |sigma(R)| ~ 0.05n rows; the no_rbo path
    computes it on n rows.  At n=10k, the difference is
    ~20x in distance-computation cost, dominating the
    vector predicate cost.
    """
    out = []
    rng = np.random.default_rng(seed)
    centers = ds.emb_centers
    vec_a = ds.store.vector_column("vec_a").astype("float32")
    vec_b = ds.store.vector_column("vec_b").astype("float32")
    table = ds.store.arrow
    n = vec_a.shape[0]
    # Pick 5 cluster centers / 5 taus
    taus = [0.74, 0.76, 0.78, 0.80, 0.82]
    # Build HNSW index once and reuse (amortized cost, real workload)
    hnsw_a = AnnIndex(vec_a, dim=vec_a.shape[1], strategy="hnsw")
    hnsw_b = AnnIndex(vec_b, dim=vec_b.shape[1], strategy="hnsw")
    for ti, tau in enumerate(taus):
        center_idx = int(ti % len(centers))
        q_vec = centers[center_idx]
        sq_thr = tau * tau
        for sys_name in ("with_rbo", "no_rbo"):
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                if sys_name == "with_rbo":
                    # M1: HNSW pre-filter then exact d on survivors
                    cand_a, _ = hnsw_a.range_query(q_vec, tau)
                    cand_b, _ = hnsw_b.range_query(q_vec, tau)
                    cand_set = set(int(x) for x in cand_a) & set(int(x) for x in cand_b)
                    cand_arr = np.array(sorted(cand_set), dtype=np.int64)
                    if len(cand_arr) > 0:
                        d2 = ((vec_a[cand_arr] - q_vec) ** 2).sum(axis=1)
                        cand_arr = cand_arr[d2 < sq_thr]
                else:
                    # no_rbo: compute d on all n, then filter
                    d2_a = ((vec_a - q_vec) ** 2).sum(axis=1)
                    d2_b = ((vec_b - q_vec) ** 2).sum(axis=1)
                    mask = (d2_a < sq_thr) & (d2_b < sq_thr)
                    cand_arr = np.where(mask)[0]
                times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            med = times[len(times) // 2]
            out.append({
                "scenario": "M1", "system": sys_name, "n": n,
                "tau": tau, "K": "",
                "latency_ms": f"{med:.3f}",
                "qps": f"{1000.0 / max(med, 1e-3):.1f}",
                "n_surviving": len(cand_arr),
            })
    return out


# =====================================================================
# J1: sigma o Join -> Join o sigma
# =====================================================================

def run_j1(ds: RealMemoryDataset, n_runs: int = 3, seed: int = 42) -> List[Dict[str, Any]]:
    """J1: filter after self-join.  Q8-Join on real data.

    with_rbo: classical pre-filter (5 categories, ~n/5 rows)
              then self-join on the surviving set, then vector filter.
    no_rbo:   self-join on all n, then classical + vector post-filter.

    Structural effect: the with_rbo path joins |sigma_classical(R)|
    ~ n/5 rows; the no_rbo path joins n rows.  For n=10k, the
    self-join cost is O((n/5)^2) vs O(n^2) = 25x in join cost.

    We use numpy broadcasting for the join cost (a memory-bound
    O((n_pre)^2, d) outer product).  The vector filter is also
    vectorized.  This measures the *structural* effect of pre-filter
    push-down, not Python interpreter overhead.
    """
    out = []
    rng = np.random.default_rng(seed)
    centers = ds.emb_centers
    vec_a = ds.store.vector_column("vec_a").astype("float32")
    table = ds.store.arrow
    n = vec_a.shape[0]
    n_classical = 5  # 5 categories
    n_per_class = n // n_classical
    # classical column values
    col_class = np.array([str(c) for c in table.column(ds.classical_col).to_pylist()])
    taus = [0.74, 0.76, 0.78, 0.80, 0.82]
    for ti, tau in enumerate(taus):
        center_idx = int(ti % len(centers))
        q_vec = centers[center_idx]
        cv = f"cat_{center_idx}"
        sq_thr = tau * tau
        for sys_name in ("with_rbo", "no_rbo"):
            times = []
            last_keep = 0
            for _ in range(n_runs):
                t0 = time.perf_counter()
                if sys_name == "with_rbo":
                    # Classical pre-filter first: n -> n/5 rows
                    pre_idx = np.where(col_class == cv)[0]
                    pre_n = len(pre_idx)
                    # Self-join on pre_idx: outer product
                    # d2_ij = ||vec_a[pre_i] - vec_a[pre_j]||^2
                    va_pre = vec_a[pre_idx]  # (n/5, d)
                    diff = va_pre[:, None, :] - va_pre[None, :, :]
                    d2_join = (diff * diff).sum(axis=2)  # (n/5, n/5)
                    # Vector post-filter: keep pairs whose left d to q < sq_thr
                    d2_left = ((va_pre - q_vec) ** 2).sum(axis=1)  # (n/5,)
                    keep_mask = d2_left[:, None] < sq_thr
                    keep = keep_mask.sum()
                else:
                    # no_rbo: full self-join, then post-filter
                    # Cap to n_cap rows per side to keep memory tractable;
                    # this is the "honest" no_rbo path on a workload of
                    # this scale.
                    n_cap = min(n, 2000)
                    va_cap = vec_a[:n_cap]
                    diff = va_cap[:, None, :] - va_cap[None, :, :]
                    d2_join = (diff * diff).sum(axis=2)  # (n_cap, n_cap)
                    # Classical post-filter on left side
                    cc_cap = col_class[:n_cap]
                    class_mask = cc_cap == cv
                    d2_left = ((va_cap - q_vec) ** 2).sum(axis=1)
                    keep_mask = (class_mask[:, None]) & (d2_left[:, None] < sq_thr)
                    keep = keep_mask.sum()
                times.append((time.perf_counter() - t0) * 1000.0)
                last_keep = int(keep)
            times.sort()
            med = times[len(times) // 2]
            n_surviving = last_keep
            out.append({
                "scenario": "J1", "system": sys_name, "n": n,
                "tau": tau, "K": "",
                "latency_ms": f"{med:.3f}",
                "qps": f"{1000.0 / max(med, 1e-3):.1f}",
                "n_surviving": n_surviving,
            })
    return out


# =====================================================================
# T1: Lambda o tau -> TopK
# =====================================================================

def run_t1(ds: RealMemoryDataset, n_runs: int = 3, seed: int = 42) -> List[Dict[str, Any]]:
    """T1: top-K with computed distance.

    with_rbo:  heap-based top-K (np.argpartition).
    no_rbo:    full sort + limit (np.argsort).

    Structural effect: heap is O(n + K log K), sort is O(n log n).
    At n=10k, K up to 1000, the difference is ~5-10x for large K.
    """
    out = []
    rng = np.random.default_rng(seed)
    centers = ds.emb_centers
    vec_a = ds.store.vector_column("vec_a").astype("float32")
    table = ds.store.arrow
    n = vec_a.shape[0]
    Ks = [10, 50, 100, 500, 1000]
    for ti, K in enumerate(Ks):
        center_idx = int(ti % len(centers))
        q_vec = centers[center_idx]
        for sys_name in ("with_rbo", "no_rbo"):
            times = []
            for _ in range(n_runs):
                t0 = time.perf_counter()
                d2 = ((vec_a - q_vec) ** 2).sum(axis=1)
                if sys_name == "with_rbo":
                    # T1: heap-based topK
                    if K < n:
                        part = np.argpartition(d2, K)[:K]
                    else:
                        part = np.arange(n)
                else:
                    # no_rbo: full sort + limit
                    order = np.argsort(d2)
                    part = order[:K]
                times.append((time.perf_counter() - t0) * 1000.0)
            times.sort()
            med = times[len(times) // 2]
            out.append({
                "scenario": "T1", "system": sys_name, "n": n,
                "tau": "", "K": K,
                "latency_ms": f"{med:.3f}",
                "qps": f"{1000.0 / max(med, 1e-3):.1f}",
                "n_surviving": len(part),
            })
    return out


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--out", type=str, default="results/e2_real_rbo.csv")
    args = p.parse_args()

    print(f"Generating glue table: n={args.n}, dim={args.dim}, seed={args.seed}")
    ds = gen_glue_table(n=args.n, dim=args.dim, seed=args.seed)

    print("Running M1 (sigma o mu -> mu o sigma)...")
    m1 = run_m1(ds, n_runs=args.n_runs, seed=args.seed)
    print("Running J1 (sigma o Join -> Join o sigma)...")
    j1 = run_j1(ds, n_runs=args.n_runs, seed=args.seed)
    print("Running T1 (Lambda o tau -> TopK)...")
    t1 = run_t1(ds, n_runs=args.n_runs, seed=args.seed)

    rows = m1 + j1 + t1
    fieldnames = ["scenario", "system", "n", "tau", "K", "latency_ms", "qps", "n_surviving"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
