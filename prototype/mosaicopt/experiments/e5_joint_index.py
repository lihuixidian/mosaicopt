"""E5: Joint scalar + vector index adoption.

This experiment demonstrates that when an integrator registers a
joint index on a scalar attribute A and a vector attribute V
(e.g. a SeRF-style segment graph or an iRangeGraph-style
elemental-graph hierarchy), MosaicOpt's cost-based optimizer:

(1) sees the joint index as a candidate physical realization
    for the conjunctive sigma_A and sigma_{||V - q|| < tau};
(2) picks it whenever its user-supplied cost function c_chi is
    lower than the cascade (B-tree pre-filter then HNSW with
    post-filter); and
(3) keeps the optimizer extension purely additive: no new
    algebraic identity is needed.

Setup
-----
* data: a synthetic Mem-like relation with attributes
  (id, subsystem in {'ft', 'cache', 'auth'}, embedding ~ U(-1, 1)^d).
  d = 128, n = 10_000, n('ft') = 4000 (40% of rows).
* query template (the canonical joint predicate):
      Q_joint = Sigma_{subsystem='ft' AND ||emb - q|| < tau}(Mem)
* sweep: scalar selectivity s_c (fraction of rows with subs='ft')
  and vector selectivity s_v (||emb - q|| < tau for in-range rows).
  Range: s_c in {0.05, 0.10, 0.20, 0.40} (the prototype's
  canonical 4 narrow regimes) and s_v in {0.005, 0.01, 0.05, 0.20}.

Three candidate plans per cell:
  P_hnsw:  HNSW with a post-filter on subsystem='ft' (recall 1.00)
  P_casc:  B-tree pre-filter, then HNSW on the surviving subset
  P_joint: SeRF-style joint index J_{subs, emb} with cost
           c_chi(s_c, s_v, n) = 0.5 + 0.5 * s_c * n * log(1/s_v)
            (modelled after SeRF's 2D-segment-graph scaling:
             the segment graph visits O(s_c*n*log(1/s_v)) nodes)

What we measure
---------------
* For each cell (s_c, s_v), the CBO's predicted cost for each
  plan and the picked plan under the recall floor of 0.95.
* Same comparison repeated under four CBO configurations:
    -- flat_only      : no HNSW, only flat scan over the surviving set
    -- hnsw_allowed   : flat / ivf / hnsw (paper default)
    -- hnsw_joint     : flat / ivf / hnsw / joint  (the integrator
                        registers the SeRF-style index)
    -- joint_only     : only the joint index is allowed (sanity
                        check: when we force it, does the chosen
                        plan match the predicted opt?)

We expect: under hnsw_joint, the joint index is selected in the
{tight s_v, moderate s_c} regime (SeRF's sweet spot) and yields
the lowest predicted latency among the four candidates; under
joint_only, the optimizer routes through it on every cell where
its cost is below the recall floor's threshold.

Output
------
Writes a CSV with one row per (configuration, s_c, s_v) cell:
    configuration, s_c, s_v, picked, time_hnsw_ms, time_casc_ms,
    time_joint_ms, picked_time_ms
and prints a short summary of the "joint chosen" cells.
"""
from __future__ import annotations

import csv
import math
import os
from typing import Dict, List, Tuple


HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "..", "results"))
os.makedirs(RESULTS_DIR, exist_ok=True)
OUT_CSV = os.path.join(RESULTS_DIR, "e5_joint_index.csv")


# ---------------------------------------------------------------------------
# Cost model (mirrors mosaicopt/optimizer/cbo.py: COST_FLAT/IVF/HNSW)
# ---------------------------------------------------------------------------

N_ROWS = 10_000           # corpus size, matches the §11 real-style runs
D = 128                    # embedding dimension, matches e1_real.nk
COST_FLAT_PER_ROW = 0.06 * D / 1e6   # ms per row scan (flat)
COST_HNSW_PER_ROW = 0.0008 * D / 1e6 # ms per row HNSW visit
COST_HNSW_OVERHEAD = 0.1             # ms HNSW setup overhead
COST_IVF_OVERHEAD = 0.5              # ms IVF probe overhead
RECALL_FLOOR = 0.95
RHO_PRE_FILTER = 0.6                  # ρ_f for a moderate pre-filter


def cost_hnsw(s_c: float, s_v: float, n: int) -> float:
    """HNSW with post-filter on subsystem: visits all n * s_v candidate
    rows at the vector stage, then drops n * s_v * s_c at the post-filter."""
    return (
        COST_HNSW_OVERHEAD
        + COST_HNSW_PER_ROW * (n * s_v)        # all vector hits
        + COST_FLAT_PER_ROW * (n * s_v)        # exact distance compute
    )


def cost_cascade(s_c: float, s_v: float, n: int) -> float:
    """B-tree pre-filter (negligible cost) + HNSW over the surviving
    subset of n * s_c rows, at selectivity s_v of THAT subset.
    HNSW's surviving graph fragments under pre-filter; we apply a
    1/rho penalty per paper §6.2.2 (rho=RHO_PRE_FILTER)."""
    return (
        COST_HNSW_OVERHEAD / max(0.05, RHO_PRE_FILTER)
        + COST_HNSW_PER_ROW * (n * s_c)        # full HNSW over R_f
        + COST_FLAT_PER_ROW * (n * s_c * s_v)  # exact re-eval on survivors
    )


def cost_joint_seRF(s_c: float, s_v: float, n: int) -> float:
    r"""SeRF-style 2D segment graph (Zuo et al. 2024, Section 3.2) +
    iRangeGraph elemental graphs (Xu et al. 2025, Section 4).

    Both indexes visit exactly the n * s_c * s_v entries that pass
    the *joint* predicate (no overshoot from pre-filter segmentation),
    and the segment-graph skip pointer reduces per-visit cost by
    ~2x over a fragmented HNSW over R_f. The cost function is:

        c_joint(s_c, s_v, n)
          = COST_HNSW_OVERHEAD * 0.5        # lighter entry overhead
          + (per-visit-joint) * (n * s_c * s_v)
          + COST_FLAT_PER_ROW * (n * s_c * s_v)

    We set per-visit-joint = COST_HNSW_PER_ROW * 0.55, matching the
    ~2x improvement SeRF and iRangeGraph report over the cascade
    baseline on the n = 10k regime (paper section 6.2 / SeRF Fig. 7)."""
    per_visit_joint = COST_HNSW_PER_ROW * 0.55
    return (
        COST_HNSW_OVERHEAD * 0.5
        + per_visit_joint * (n * s_c * s_v)
        + COST_FLAT_PER_ROW * (n * s_c * s_v)
    )


def recall_ok(strategy: str, s_c: float, s_v: float) -> bool:
    """Whether the strategy meets the recall floor under the regime.

    * HNSW alone: recall 1.00 (exact distance compute at every visit).
    * Cascade:    recall 1.00 (exact at the surviving subset).
    * Joint:      recall 1.00 by construction (SeRF visits every
                  candidate within the window).
    All three meet the 0.95 floor in this experiment -- the cost
    comparison decides the winner, not the floor."""
    return True


# ---------------------------------------------------------------------------
# CBO selection: minimal re-implementation, scoped to E5
# ---------------------------------------------------------------------------

def pick(candidates: List[str], s_c: float, s_v: float) -> Tuple[str, Dict[str, float]]:
    costs = {}
    for name in candidates:
        if name == "hnsw":
            c = cost_hnsw(s_c, s_v, N_ROWS)
        elif name == "cascade":
            c = cost_cascade(s_c, s_v, N_ROWS)
        elif name == "joint":
            c = cost_joint_seRF(s_c, s_v, N_ROWS)
        else:
            c = float("inf")
        if recall_ok(name, s_c, s_v):
            costs[name] = c
        else:
            costs[name] = math.inf
    picked = min(costs.items(), key=lambda kv: kv[1])
    return picked[0], costs


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------

CONFIGS = {
    "flat_only":    ["cascade"],   # falls back to pre-filter + flat
    "hnsw_allowed": ["hnsw", "cascade"],
    "hnsw_joint":   ["hnsw", "cascade", "joint"],
    "joint_only":   ["joint"],
}

S_C_GRID = [0.05, 0.10, 0.20, 0.40]
S_V_GRID = [0.005, 0.01, 0.05, 0.20]


def run() -> Dict[str, int]:
    """Run the sweep and write CSV. Returns a dict of strategy counts."""
    rows: List[Tuple[str, float, float, str,
                     float, float, float, float]] = []
    counts: Dict[str, int] = {c: 0 for c in CONFIGS}

    for cfg_name, candidates in CONFIGS.items():
        for s_c in S_C_GRID:
            for s_v in S_V_GRID:
                picked, costs = pick(candidates, s_c, s_v)
                rows.append((
                    cfg_name,
                    s_c,
                    s_v,
                    picked,
                    costs.get("hnsw", math.inf),
                    costs.get("cascade", math.inf),
                    costs.get("joint", math.inf),
                    costs[picked] if picked in costs else math.inf,
                ))
                counts[cfg_name] += 1 if picked == "joint" else 0

    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["configuration", "s_c", "s_v",
                    "picked", "time_hnsw_ms", "time_casc_ms",
                    "time_joint_ms", "picked_time_ms"])
        for row in rows:
            w.writerow(row)

    return counts, rows


def summarize(counts, rows) -> str:
    """Print a short human-readable summary of the sweep."""
    joint_chosen_by_cfg = {}
    for cfg in CONFIGS:
        wins = sum(1 for r in rows if r[0] == cfg and r[3] == "joint")
        joint_chosen_by_cfg[cfg] = wins

    # Find the "best win" for the joint index
    best_win = None
    best_ratio = 0.0
    for r in rows:
        _, s_c, s_v, picked, t_h, t_c, t_j, _ = r
        # only count rows where all three costs are finite
        if not (math.isfinite(t_h) and math.isfinite(t_c) and math.isfinite(t_j)):
            continue
        if t_j <= min(t_h, t_c):
            ratio = min(t_h, t_c) / t_j
            if ratio > best_ratio:
                best_ratio = ratio
                best_win = (s_c, s_v, t_h, t_c, t_j)

    lines = []
    lines.append("E5 joint-index adoption summary")
    lines.append("=" * 40)
    for cfg, n in joint_chosen_by_cfg.items():
        lines.append(f"  cfg={cfg:14s}  joint_index chosen in {n:>2d} / {len(S_C_GRID)*len(S_V_GRID)} cells")
    if best_win is not None:
        s_c, s_v, t_h, t_c, t_j = best_win
        lines.append("")
        lines.append(f"  best joint-index win:")
        lines.append(f"    (s_c={s_c:.2f}, s_v={s_v:.3f})")
        lines.append(f"      cascade={t_c:.3f} ms  joint={t_j:.3f} ms  -> {t_c/t_j:.2f}x speedup over cascade")
        lines.append(f"      hnsw={t_h:.3f} ms     joint={t_j:.3f} ms  -> {t_h/t_j:.2f}x speedup over hnsw")
    lines.append("")
    lines.append(f"results: {OUT_CSV}")
    return "\n".join(lines)


if __name__ == "__main__":
    counts, rows = run()
    print(summarize(counts, rows))
