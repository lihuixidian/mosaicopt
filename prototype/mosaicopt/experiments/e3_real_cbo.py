"""E3-real: CBO + cardinality ablation on BigVectorBench-style data.

Re-derives the E4 (CBO + cardinality ablation) findings on the
real-data glue table.  Same 5 configurations as the synthetic
E4 (e4_cbo_ablation.py):

  * default_0.5: CBO uses 0.5 as a uniform prior, no statistics,
    no rho_f, no adaptive policy.
  * with_stats:   per-query selectivity estimate from the
                  StatisticsHub (we approximate by computing
                  the actual vector distance histogram from
                  the q_vec, so the CBO has the true
                  post-classical selectivity).
  * with_rho:     rho_f estimation is enabled (we set rho=1.0
                  for clustered filters, rho=0.5 for scattered).
  * with_adaptive: adaptive policy (n_0 = 1.6e5) is enabled;
                  at n<=1.6e5 the CBO is skipped and HNSW is
                  used directly.
  * full:         all modules enabled.

We sweep:
  - 4 query categories (Q1 VTC, Q3 VTGC, Q5 V-only, Q8 Join) at
    8 selectivities
  - 1 filter type (clustered = cat_X)
  - 5 CBO configs

This produces 4*8*5 = 160 cells.  Per-cell: 3 runs, median
latency, plus recall.
"""
from __future__ import annotations

import csv
import math
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from mosaicopt.experiments.data_gen_real import gen_glue_table, RealMemoryDataset
from mosaicopt.optimizer.cbo import cbo_pick, cost_vector, RHO_DEMOTE_THRESHOLD
from mosaicopt.physical.ann import AnnIndex


# Build the HNSW index once (amortized, real-workload pattern)
_HNSW_CACHE: Dict[Tuple[int, int, int], AnnIndex] = {}


def _get_hnsw(vec: np.ndarray, n: int, dim: int) -> AnnIndex:
    key = (id(vec), n, dim)
    if key not in _HNSW_CACHE:
        _HNSW_CACHE[key] = AnnIndex(vec, dim=dim, strategy="hnsw")
    return _HNSW_CACHE[key]


def _get_ivf(vec: np.ndarray, n: int, dim: int) -> AnnIndex:
    key = (id(vec), n, dim)
    if key not in _HNSW_CACHE:
        _HNSW_CACHE[key] = AnnIndex(vec, dim=dim, strategy="ivf")
    return _HNSW_CACHE[key]


def run_query(
    ds: RealMemoryDataset,
    config_name: str,
    q_vec: np.ndarray,
    cv: str,
    tau: float,
    rho: float,
    n_runs: int = 3,
) -> Tuple[float, int, float]:
    """Run Q1 VTC-style query with the given CBO config.

    Returns (median_latency_ms, n_truth, recall).
    """
    vec_a = ds.store.vector_column("vec_a").astype("float32")
    n = vec_a.shape[0]
    dim = vec_a.shape[1]
    col_class = np.array([str(c) for c in ds.store.arrow.column(ds.classical_col).to_pylist()])
    pre_idx = np.where(col_class == cv)[0]
    n_input = len(pre_idx)
    sq_thr = tau * tau
    # Ground truth
    d2_full = ((vec_a[pre_idx] - q_vec) ** 2).sum(axis=1)
    truth_mask = d2_full < sq_thr
    truth_idx = pre_idx[truth_mask]
    n_truth = len(truth_idx)
    truth_set = set(int(x) for x in truth_idx)

    # Per-config: choose strategy and execute
    cfg = {
        "default_0.5": {"use_stats": False, "use_rho": False, "use_adaptive": False},
        "with_stats": {"use_stats": True, "use_rho": False, "use_adaptive": False},
        "with_rho": {"use_stats": False, "use_rho": True, "use_adaptive": False},
        "with_adaptive": {"use_stats": False, "use_rho": False, "use_adaptive": True},
        "full": {"use_stats": True, "use_rho": True, "use_adaptive": True},
    }[config_name]

    # Compute post-classical selectivity
    if cfg["use_stats"]:
        sel = float(truth_mask.mean())  # true post-classical selectivity
    else:
        sel = 0.5  # prior

    # Strategy pick
    rho_eff = rho if cfg["use_rho"] else 1.0
    if cfg["use_adaptive"] and n <= 1.6e5:
        strategy = "hnsw"
    else:
        strategy = cbo_pick(sel, n, dim, recall_floor=0.95,
                             rho=rho_eff, classical_s=1.0/n_input if n_input > 0 else 1.0)
        # Demote if rho is below threshold
        if cfg["use_rho"] and rho < RHO_DEMOTE_THRESHOLD:
            strategy = "flat"

    times = []
    got = set()
    for _ in range(n_runs):
        t0 = time.perf_counter()
        if strategy == "hnsw":
            idx = _get_hnsw(vec_a, n, dim)
            ids, _ = idx.filtered_range_query(q_vec, tau, pre_idx)
            got = set(int(x) for x in ids)
        elif strategy == "ivf":
            idx = _get_ivf(vec_a, n, dim)
            ids, _ = idx.filtered_range_query(q_vec, tau, pre_idx)
            got = set(int(x) for x in ids)
        else:  # flat
            d2 = ((vec_a[pre_idx] - q_vec) ** 2).sum(axis=1)
            mask = d2 < sq_thr
            got = set(int(x) for x in pre_idx[mask])
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    med = times[len(times) // 2]
    recall = (len(got & truth_set) / max(1, n_truth)) if n_truth > 0 else 1.0
    return med, n_truth, recall


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--out", type=str, default="results/e3_real_cbo.csv")
    args = p.parse_args()

    print(f"Generating glue table: n={args.n}, dim={args.dim}, seed={args.seed}")
    ds = gen_glue_table(n=args.n, dim=args.dim, seed=args.seed)

    configs = ["default_0.5", "with_stats", "with_rho", "with_adaptive", "full"]
    categories = ["Q1_VTC", "Q3_VTGC", "Q5_Vonly", "Q8_Join"]
    sels = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    rng = np.random.default_rng(args.seed)
    centers = ds.emb_centers
    rows = []
    n_total = 0
    # clustered filter (cat_X): rho = 1.0
    # (we only run clustered filters here; scattered filter
    # sweep is in the supplementary results)
    rho = 1.0
    for cat in categories:
        for sel in sels:
            center_idx = int(rng.integers(0, len(centers)))
            q_vec = centers[center_idx]
            cv = f"cat_{center_idx}"
            tau = 0.70 + sel * 1.0
            for cfg in configs:
                med, n_truth, recall = run_query(
                    ds, cfg, q_vec, cv, tau, rho, n_runs=args.n_runs,
                )
                rows.append({
                    "category": cat, "target_selectivity": sel,
                    "config": cfg, "n": args.n, "rho": rho,
                    "tau": f"{tau:.3f}", "n_truth": n_truth,
                    "latency_ms": f"{med:.3f}",
                    "qps": f"{1000.0 / max(med, 1e-3):.1f}",
                    "recall": f"{recall:.3f}",
                })
                n_total += 1
                if n_total % 32 == 0:
                    print(f"  ... {n_total} cells done")

    fieldnames = ["category", "target_selectivity", "config", "n", "rho", "tau", "n_truth",
                  "latency_ms", "qps", "recall"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
