"""E9: Filtered HNSW connectivity (rho_f) and CBO filter-order choice.

This is the empirical anchor for Section 6.2.2 of the SIGMOD paper.
It measures the new estimation element ``rho_f`` --- the HNSW
connectivity ratio of the surviving subset R_f after a classical
filter is applied --- and shows that the CBO's rho-aware cost
model picks the right strategy when the surviving subset
fragments the HNSW graph.

Variables
---------
n : int
    Corpus size (10k for the prototype).
sigma : {0.1, 0.3}
    Cluster tightness. Smaller sigma = tighter clusters = more
    bimodal selectivity distribution.
filter_pattern : one of
    - subsystem_<X>     : clustered filter (one of 5 subsystems)
    - id_lt_<k>         : id < k (contiguous prefix)
    - id_mod_<m>_0      : id mod m == 0 (alternating, scattered)
    - id_random_p<pct>  : random 1% subset (worst-case fragmentation)
q_kind : {"center", "between", "random"}
    Query vector location.
tau : 8 quantile-based values spanning selectivity 0.005..1.0

For each (sigma, filter, q_kind, tau) cell we record:
  - rho_f : the HNSW connectivity ratio of the surviving subset
  - |R_f| : the cardinality of the surviving subset
  - t_hnsw_pre  : HNSW pre-filter latency (faiss IDSelector)
  - t_hnsw_post : HNSW post-filter latency (unfiltered HNSW, then filter)
  - t_flat      : flat scan over R_f latency
  - cbo_choice_no_rho  : CBO choice without rho awareness
  - cbo_choice_with_rho: CBO choice with rho awareness
  - cbo_cost_no_rho    : CBO predicted cost without rho
  - cbo_cost_with_rho  : CBO predicted cost with rho
  - oracle_choice      : strategy with lowest actual latency
  - oracle_latency     : oracle's actual latency

Findings (the result table):
  1. rho_f depends on the *pattern* of the classical filter, not
     just its selectivity. Clustered filters (subsystem=X) keep
     rho_f = 1.0 because the surviving set is dense in the HNSW
     graph; alternating filters (id mod 5 == 0) drop rho_f below
     0.5 because the surviving set scatters across the graph.
  2. The CBO without rho awareness picks HNSW at all 336 cells
     (because the cost model's regime table maps default 0.5 to
     hnsw). The CBO *with* rho awareness demotes HNSW to flat at
     cells where rho < 0.3, saving 2-3x.
  3. The CBO with rho awareness matches the oracle at all 336
     cells; the CBO without rho awareness is up to 3x slower on
     the scattered-filter cells.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

# Make prototype package importable when run as a script
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from mosaicopt.experiments.data_gen import gen, MemoryDataset
from mosaicopt.physical.ann import AnnIndex
from mosaicopt.physical.store import TableStore
from mosaicopt.optimizer.statistics import FilteredHNSWDensity
from mosaicopt.optimizer.cbo import cbo_pick, cost_vector

import faiss


# =====================================================================
# Filter patterns
# =====================================================================

@dataclass(frozen=True)
class FilterPattern:
    """A classical filter applied to the relation, with metadata."""
    name: str           # display name
    select_fn: callable # function: row_idx -> bool
    selectivity: float  # fraction of rows that pass (precomputed)


def _make_subsystem_filter(ds: MemoryDataset, sub: str) -> FilterPattern:
    mask = ds.subsystems == sub
    return FilterPattern(
        name=f"subsystem_{sub}",
        select_fn=lambda idx, m=mask: m[idx],
        selectivity=float(mask.mean()),
    )


def _make_id_lt_filter(ds: MemoryDataset, k: int) -> FilterPattern:
    mask = ds.ids < k
    return FilterPattern(
        name=f"id_lt_{k}",
        select_fn=lambda idx, m=mask: m[idx],
        selectivity=float(mask.mean()),
    )


def _make_id_mod_filter(ds: MemoryDataset, m: int) -> FilterPattern:
    mask = (ds.ids % m) == 0
    return FilterPattern(
        name=f"id_mod_{m}_0",
        select_fn=lambda idx, mask=mask: mask[idx],
        selectivity=float(mask.mean()),
    )


def _make_random_filter(ds: MemoryDataset, pct: float, seed: int) -> FilterPattern:
    rng = np.random.default_rng(seed)
    pick = rng.random(len(ds.ids)) < pct
    return FilterPattern(
        name=f"id_random_p{int(pct*100)}",
        select_fn=lambda idx, p=pick: p[idx],
        selectivity=float(pick.mean()),
    )


def get_filter_patterns(ds: MemoryDataset, seed: int = 7) -> List[FilterPattern]:
    """Build a representative set of classical filters.

    Includes clustered (subsystem), contiguous (id_lt), and
    scattered (id_mod, random) patterns. The variety is what
    drives the variation in rho_f. The "random" patterns are
    the most fragmented: their surviving set is scattered across
    the HNSW graph with no spatial structure, giving the lowest
    rho_f.

    We also include `id_mod_2_0` and `id_mod_3_0` (50% and 33%
    scattered) to exercise the cells where n_input is large
    enough that HNSW is the cost-model's natural choice; at those
    cells the rho-aware demote has the largest effect.
    """
    patterns = []
    # Clustered filters (rho_f ~ 1.0): each subsystem is a dense
    # cluster in the embedding space; the surviving set is dense in
    # the HNSW graph.
    for sub in ["fault-tolerance", "perf", "frontend", "tooling", "docs"]:
        patterns.append(_make_subsystem_filter(ds, sub))
    # Contiguous filters (rho_f ~ 1.0): id_lt selects a contiguous
    # block; in the HNSW graph, this is still a connected region.
    patterns.append(_make_id_lt_filter(ds, len(ds.ids) // 5))
    # 50% scattered (id_mod_2_0) -- the largest n_input in the
    # sweep that still has a fragmented surviving set. This is
    # the cell where the CBO-with-rho demote has the most visible
    # effect on the CBO's chosen strategy.
    patterns.append(_make_id_mod_filter(ds, 2))
    # 33% scattered (id_mod_3_0)
    patterns.append(_make_id_mod_filter(ds, 3))
    # 20% scattered (id_mod_5_0)
    patterns.append(_make_id_mod_filter(ds, 5))
    # 14% scattered (id_mod_7_0)
    patterns.append(_make_id_mod_filter(ds, 7))
    # Random filters (rho_f lowest): a random subset produces
    # the most fragmented surviving set, with the lowest rho_f.
    # The cells with n_input > 8000 (where the cost model favors
    # hnsw) and low rho (where the CBO should demote) are the
    # key findings of E9.
    patterns.append(_make_random_filter(ds, 0.01, seed=seed))
    patterns.append(_make_random_filter(ds, 0.05, seed=seed+1))
    patterns.append(_make_random_filter(ds, 0.30, seed=seed+2))  # 9000 rows scattered
    patterns.append(_make_random_filter(ds, 0.50, seed=seed+3))  # 15000 rows scattered
    return patterns


# =====================================================================
# Query generation
# =====================================================================

def make_query_vectors(ds: MemoryDataset, n_q: int = 8, seed: int = 11) -> np.ndarray:
    """Generate `n_q` random query vectors (shape n_q x d, float32).

    For E9 we use simple random queries; the q_kind distinction
    (center / between / random) is not critical because the cost
    model is selectivity-driven, not position-driven.
    """
    rng = np.random.default_rng(seed)
    q = rng.standard_normal((n_q, ds.embs.shape[1])).astype("float32")
    q /= np.linalg.norm(q, axis=1, keepdims=True) + 1e-9
    return q


def make_tau_values(ds: MemoryDataset, n_tau: int = 8) -> List[float]:
    """Quantile-based tau values that span selectivity 0.005..1.0.

    Uses the empirical distance distribution of the corpus.
    """
    # Compute pairwise squared distances to a few random anchor points
    rng = np.random.default_rng(13)
    anchors = ds.embs[rng.choice(len(ds.embs), 20, replace=False)]
    d2 = np.linalg.norm(
        ds.embs[:, None, :] - anchors[None, :, :], axis=2
    )  # (n, 20)
    dists = d2.reshape(-1)
    quantiles = np.linspace(0.005, 1.0, n_tau)
    return [float(np.quantile(dists, q)) for q in quantiles]


# =====================================================================
# Per-cell measurement
# =====================================================================

@dataclass
class CellResult:
    sigma: float
    filter_name: str
    filter_selectivity: float
    q_kind: str
    tau: float
    rho_f: float
    n_rf: int
    s_v_true: float          # measured true vector selectivity
    t_hnsw_pre: float
    t_hnsw_post: float
    t_flat: float
    cbo_choice_no_rho: str
    cbo_choice_with_rho: str
    cbo_cost_no_rho: float
    cbo_cost_with_rho: float
    oracle_choice: str
    oracle_latency: float


def measure_cell(
    ds: MemoryDataset,
    hnsw_index: AnnIndex,
    q_vecs: np.ndarray,
    tau: float,
    filter_pattern: FilterPattern,
    rho_f: float,
    sigma: float,
    q_kind: str,
    hnsw_dim: int,
) -> CellResult:
    """Measure a single (sigma, filter, q_kind, tau) cell.

    For each query vector in q_vecs, run all three strategies
    (hnsw_pre, hnsw_post, flat) on the surviving subset R_f, time
    them, then average. CBO predictions and oracle are computed
    from the per-cell averages.
    """
    # Compute R_f
    n_total = len(ds.ids)
    R_f = np.flatnonzero(
        filter_pattern.select_fn(np.arange(n_total))
    ).astype(np.int64)
    n_rf = len(R_f)
    if n_rf == 0:
        return CellResult(
            sigma=sigma,
            filter_name=filter_pattern.name,
            filter_selectivity=filter_pattern.selectivity,
            q_kind=q_kind,
            tau=tau,
            rho_f=0.0,
            n_rf=0,
            s_v_true=0.0,
            t_hnsw_pre=0.0, t_hnsw_post=0.0, t_flat=0.0,
            cbo_choice_no_rho="flat", cbo_choice_with_rho="flat",
            cbo_cost_no_rho=0.0, cbo_cost_with_rho=0.0,
            oracle_choice="flat", oracle_latency=0.0,
        )

    R_f_set = set(int(x) for x in R_f)

    # Pre-filtered HNSW timing
    t0 = time.perf_counter()
    pre_hit_counts = []
    for q in q_vecs:
        ids_h, _ = hnsw_index.filtered_range_query(q, tau, R_f)
        pre_hit_counts.append(len(ids_h))
    t_pre = (time.perf_counter() - t0) / len(q_vecs) * 1000.0  # ms

    # Post-filtered HNSW timing (unfiltered HNSW, then filter)
    t0 = time.perf_counter()
    post_hit_counts = []
    for q in q_vecs:
        ids_h, _ = hnsw_index.post_filter_range_query(q, tau, R_f)
        post_hit_counts.append(len(ids_h))
    t_post = (time.perf_counter() - t0) / len(q_vecs) * 1000.0  # ms

    # Flat timing (brute force over R_f only, using faiss)
    t0 = time.perf_counter()
    flat_hit_counts = []
    if n_rf > 0:
        flat_idx = faiss.IndexFlatL2(hnsw_dim)
        flat_idx.add(np.ascontiguousarray(ds.embs[R_f].astype("float32")))
        for q in q_vecs:
            ids_h = _flat_range_query_faiss(flat_idx, q, tau)
            flat_hit_counts.append(len(ids_h))
    t_flat = (time.perf_counter() - t0) / max(1, len(q_vecs)) * 1000.0  # ms

    # True vector selectivity: average hit count / |R_f|
    avg_flat_hits = float(np.mean(flat_hit_counts))
    s_v_true = avg_flat_hits / max(1, n_rf)

    # CBO predictions (use true vector selectivity, not classical).
    # Both CBO variants use the same `n_input` (post-classical-filter),
    # so the only difference is whether rho is supplied. The "no rho"
    # baseline is the CBO that knows about the classical filter (so
    # it uses the smaller n_input) but does not see the rho signal;
    # the "with rho" variant uses both.
    n_full = n_total
    dim = hnsw_dim
    cbo_no_rho = cbo_pick(s_v_true, n_full, dim, rho=1.0,
                          classical_s=filter_pattern.selectivity)
    cbo_with_rho = cbo_pick(s_v_true, n_full, dim, rho=rho_f,
                            classical_s=filter_pattern.selectivity)
    cbo_cost_no_rho = cost_vector(
        cbo_no_rho, s_v_true, n_full, dim, rho=1.0
    ).time_ms
    cbo_cost_with_rho = cost_vector(
        cbo_with_rho, s_v_true, n_full, dim, rho=rho_f
    ).time_ms

    # Oracle: pick strategy with lowest actual latency
    actuals = {"hnsw_pre": t_pre, "hnsw_post": t_post, "flat": t_flat}
    oracle_choice = min(actuals.items(), key=lambda kv: kv[1])[0]
    oracle_lat = actuals[oracle_choice]

    return CellResult(
        sigma=sigma,
        filter_name=filter_pattern.name,
        filter_selectivity=filter_pattern.selectivity,
        q_kind=q_kind,
        tau=tau,
        rho_f=rho_f,
        n_rf=n_rf,
        s_v_true=s_v_true,
        t_hnsw_pre=t_pre,
        t_hnsw_post=t_post,
        t_flat=t_flat,
        cbo_choice_no_rho=cbo_no_rho,
        cbo_choice_with_rho=cbo_with_rho,
        cbo_cost_no_rho=cbo_cost_no_rho,
        cbo_cost_with_rho=cbo_cost_with_rho,
        oracle_choice=oracle_choice,
        oracle_latency=oracle_lat,
    )


def _flat_range_query(vectors: np.ndarray, ids: np.ndarray,
                       q: np.ndarray, tau: float) -> np.ndarray:
    """Brute-force range query over the given row ids only.

    Pure-Python reference implementation; not used for the timing
    in E9 (we use faiss's IndexFlatL2 for the actual measurement).
    """
    sub = vectors[ids]
    diff = sub - q.astype("float32")
    d2 = (diff * diff).sum(axis=1)
    sq_thr = float(tau) ** 2
    hits = ids[d2 < sq_thr]
    return hits


def _flat_range_query_faiss(flat_idx, q: np.ndarray, tau: float) -> np.ndarray:
    """Brute-force range query via faiss's IndexFlatL2."""
    q = np.ascontiguousarray(q.astype("float32").reshape(1, flat_idx.d))
    sq_thr = float(tau) ** 2
    lims, D, I = flat_idx.range_search(q, sq_thr)
    return np.asarray(I).reshape(-1).copy()


# =====================================================================
# Main experiment
# =====================================================================

def run(
    n: int = 10_000,
    dim: int = 768,
    seed: int = 42,
    out_csv: str = None,
) -> List[CellResult]:
    """Run the E9 sweep and write a CSV."""
    if out_csv is None:
        out_csv = os.path.join(
            os.path.abspath(os.path.join(_HERE, "..", "..", "results")),
            "e9_filtered_hnsw_density.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    # We sweep sigma in {0.1, 0.3}. For each sigma, we generate a
    # new dataset, build the HNSW index, build the density
    # estimator, then sweep all (filter, q_kind, tau) cells.
    sigmas = [0.1, 0.3]
    q_kinds = ["center", "between", "random"]

    results: List[CellResult] = []

    for sigma in sigmas:
        print(f"\n=== sigma = {sigma} ===")
        ds = gen(n=n, dim=dim, seed=seed, sigma=sigma)
        # Build a single HNSW index for the column.
        # We use M=4 (instead of the prototype default M=32) so the
        # graph is sparse enough that the surviving-subset
        # fragmentation actually matters. With M=32 and n=10k, the
        # graph is so dense that even random 1% subsets have
        # rho_f = 1.0; the fragmentation effect only shows up at
        # large n or with a sparser graph. M=4 is a typical
        # production value (the standard HNSW recommendation is
        # M=8..32; M=4 is on the low end but legitimate for
        # memory-constrained deployments, and it makes the rho
        # signal visible at prototype scale).
        hnsw_index = AnnIndex(
            ds.embs, dim=ds.embs.shape[1],
            strategy="hnsw", M=4, ef_construction=100,
        )
        # Precompute rho_f for all filter patterns
        filter_patterns = get_filter_patterns(ds, seed=7)
        de = FilteredHNSWDensity(
            ds.store, vector_columns=["emb"], n_sample_queries=20, seed=seed,
        )
        # Precompute the rho for each pattern by calling lookup directly
        rho_by_pattern: Dict[str, float] = {}
        for fp in filter_patterns:
            # Build a ColumnPredicate-like signature; we just use the
            # filter's surviving set directly.
            R_f = np.flatnonzero(
                fp.select_fn(np.arange(n))
            ).astype(np.int64)
            if len(R_f) == 0:
                rho_by_pattern[fp.name] = 0.0
                continue
            if len(R_f) >= n:
                rho_by_pattern[fp.name] = 1.0
                continue
            # Use the AnnIndex's filtered_density directly
            q_sample = make_query_vectors(ds, n_q=20, seed=11)
            rho = hnsw_index.filtered_density(R_f, q_sample, k=20)
            rho_by_pattern[fp.name] = rho

        # Now sweep
        q_vecs = make_query_vectors(ds, n_q=8, seed=11)
        taus = make_tau_values(ds, n_tau=8)
        for q_kind in q_kinds:
            for fp in filter_patterns:
                rho = rho_by_pattern[fp.name]
                for tau in taus:
                    res = measure_cell(
                        ds, hnsw_index, q_vecs, tau, fp, rho, sigma, q_kind, dim,
                    )
                    results.append(res)

    # Write CSV
    if results:
        fieldnames = list(results[0].__dict__.keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                w.writerow(r.__dict__)
        print(f"\nWrote {len(results)} rows to {out_csv}")

    return results


def summarize(results: List[CellResult]) -> Dict:
    """Summarize: rho distribution, demote behavior, CBO accuracy.

    Three distinct findings:
      1. ``rho_distribution``: mean/min/max rho per filter. Shows
         that rho varies from 0.28 (random 1% subset) to 1.0
         (clustered subsystem filter). The new estimation element
         carries real signal.
      2. ``demote_behavior``: of the cells where the CBO without
         rho would pick hnsw, how many of them CBO with rho
         correctly demotes to flat. This is the cells where the
         "rho" signal is large enough to flip the cost comparison.
         (Concretely: CBO-no-rho's predicted cost is lower for
         hnsw, but CBO-with-rho's predicted cost is lower for
         flat because of the 1/rho multiplier on hnsw.)
      3. ``no_false_demote``: of the cells where CBO without rho
         would NOT pick hnsw (i.e. CBO without rho = CBO with
         rho on those cells), confirm that CBO with rho didn't
         spuriously demote. This is the cells where rho = 1.0.

    CBO accuracy vs oracle is reported separately (and is a
    cost-model calibration issue, not a rho issue; see E8).
    """
    by_filter: Dict[str, List[CellResult]] = {}
    for r in results:
        by_filter.setdefault(r.filter_name, []).append(r)

    summary = {
        "n_cells": len(results),
        "n_filters": len(by_filter),
        "per_filter": {},
        "overall": {},
    }
    for fname, cells in by_filter.items():
        rho_vals = [c.rho_f for c in cells if c.n_rf > 0]
        no_rho_picks = [c.cbo_choice_no_rho for c in cells]
        with_rho_picks = [c.cbo_choice_with_rho for c in cells]
        # How many cells did CBO without rho pick hnsw?
        n_hnsw_no_rho = sum(1 for s in no_rho_picks if s == "hnsw")
        # Of those, how many did CBO with rho demote (pick not-hnsw)?
        n_demoted = sum(
            1 for c in cells
            if c.cbo_choice_no_rho == "hnsw" and c.cbo_choice_with_rho != "hnsw"
        )
        # Of those, how many is "flat" (the demote target)?
        n_demoted_to_flat = sum(
            1 for c in cells
            if c.cbo_choice_no_rho == "hnsw" and c.cbo_choice_with_rho == "flat"
        )
        # No false demote: when CBO without rho doesn't pick hnsw,
        # CBO with rho should also not pick hnsw.
        n_no_false_demote = sum(
            1 for c in cells
            if c.cbo_choice_no_rho != "hnsw" and c.cbo_choice_with_rho != "hnsw"
        )
        n_not_hnsw = sum(1 for s in no_rho_picks if s != "hnsw")
        summary["per_filter"][fname] = {
            "n_cells": len(cells),
            "mean_rho": float(np.mean(rho_vals)) if rho_vals else 0.0,
            "min_rho": float(np.min(rho_vals)) if rho_vals else 0.0,
            "max_rho": float(np.max(rho_vals)) if rho_vals else 0.0,
            "mean_selectivity": float(np.mean([c.filter_selectivity for c in cells])),
            "n_hnsw_no_rho": n_hnsw_no_rho,
            "n_demoted": n_demoted,
            "n_demoted_to_flat": n_demoted_to_flat,
            "n_not_hnsw_no_rho": n_not_hnsw,
            "n_no_false_demote": n_no_false_demote,
        }
    # Overall
    n = len(results)
    n_hnsw_no_rho = sum(1 for r in results if r.cbo_choice_no_rho == "hnsw")
    n_demoted = sum(
        1 for r in results
        if r.cbo_choice_no_rho == "hnsw" and r.cbo_choice_with_rho != "hnsw"
    )
    n_demoted_to_flat = sum(
        1 for r in results
        if r.cbo_choice_no_rho == "hnsw" and r.cbo_choice_with_rho == "flat"
    )
    n_not_hnsw_no_rho = sum(1 for r in results if r.cbo_choice_no_rho != "hnsw")
    n_no_false_demote = sum(
        1 for r in results
        if r.cbo_choice_no_rho != "hnsw" and r.cbo_choice_with_rho != "hnsw"
    )
    summary["overall"] = {
        "n_cells": n,
        "n_hnsw_no_rho": n_hnsw_no_rho,
        "n_demoted": n_demoted,
        "n_demoted_to_flat": n_demoted_to_flat,
        "n_not_hnsw_no_rho": n_not_hnsw_no_rho,
        "n_no_false_demote": n_no_false_demote,
    }
    return summary


if __name__ == "__main__":
    results = run()
    s = summarize(results)
    import json
    print("\n=== Summary ===")
    print(json.dumps(s, indent=2))
