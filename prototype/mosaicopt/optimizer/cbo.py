"""Cost-Based Optimizer (CBO): physical strategy selection with a
4-tuple cost.

The cost vector for a vector range scan Sigma_{||V-q||<tau}(R) is
    (time, memory, recall, rec)
where:
    * time       - estimated query time in milliseconds (rough scale)
    * memory     - memory footprint in megabytes
    * recall     - 0..1, expected recall@K of the strategy
    * rec        - construction time per call (for the prototype, this
                   is amortized into the plan's overall cost)

The CBO's job is to pick the strategy (flat / ivf / hnsw) that
minimizes the *primary* cost (time) subject to a recall floor. The
strategy table below encodes the cost model from Section 5.3 of
the paper, parameterized by selectivity s = |Sigma(R)| / |R|.

Section 6.2.2 extends the cost model with the **filtered HNSW
connectivity ratio** rho_f. When a classical filter `C` runs
upstream of a vector predicate `V`, the surviving subset
R_f = sigma_C(R) may fragment the HNSW graph; rho_f in [0, 1]
captures how much the HNSW traversal amplifies. The cost of
pre-filtered HNSW becomes

    cost_hnsw_filtered = cost_hnsw_unfiltered / rho_f

When rho_f is small (fragmented surviving set), the CBO demotes
HNSW to flat scan over R_f (a scan that is s_f * n rows, not n).

The thresholds (s < 0.05 -> flat, 0.05 <= s < 0.3 -> ivf, s >= 0.3
-> hnsw) are the values reported in the paper. They are empirical
defaults; the real system would calibrate them on the workload.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------
# Cost vector
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class CostVector:
    """4-tuple cost: (time_ms, memory_mb, recall, rec)."""
    time_ms: float
    memory_mb: float
    recall: float
    rec: float

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.time_ms, self.memory_mb, self.recall, self.rec)


# ---------------------------------------------------------------------
# Cost model for vector range scans
# ---------------------------------------------------------------------

# Empirical cost coefficients (would be calibrated; here they are
# taken from a 10k x 768-dim corpus on a single thread).
# Format: per-strategy multipliers on (time, memory, recall, rec).
# The signatures are (s, n_input, d, rho) where:
#   s         = vector range selectivity (after any classical filter)
#   n_input   = number of rows the scan actually sees (post-classical-filter)
#   d         = vector dimensionality
#   rho       = HNSW connectivity ratio of the surviving subset (1.0 = unfiltered)

# flat (brute-force) - exact, slow at high s. Cost scales with n_input.
COST_FLAT = {
    "time":  lambda s, n_input, d, rho: 0.04 + 0.06 * n_input * d / 1e6,
    "memory":lambda s, n_input, d, rho: n_input * d * 4 / (1024 * 1024),
    "recall":lambda s, n_input, d, rho: 1.0,
    "rec":   lambda s, n_input, d, rho: 0.0,
}

# IVF - moderate; recall is a function of nprobe/nlist
COST_IVF = {
    "time":  lambda s, n_input, d, rho: 0.5 + 0.01 * n_input * d / 1e6,
    "memory":lambda s, n_input, d, rho: n_input * d * 4 / (1024 * 1024) * 0.20,
    "recall":lambda s, n_input, d, rho: 0.98 - 0.02 * math.exp(-30.0 * s),
    "rec":   lambda s, n_input, d, rho: 5.0 + 0.0005 * n_input,
}

# HNSW - fast at high s, slow to build. Cost is amplified by 1/rho
# when the surviving subset fragments the HNSW graph (Section 6.2.2).
COST_HNSW = {
    "time":  lambda s, n_input, d, rho: (0.1 + 0.0008 * n_input * d / 1e6) / max(0.05, rho),
    "memory":lambda s, n_input, d, rho: n_input * d * 4 / (1024 * 1024) * 1.30,
    # When rho is small, HNSW's recall drops because the graph is
    # fragmented within R_f; the recall floor of the strategy degrades.
    "recall":lambda s, n_input, d, rho: (1.0 - 0.04 * math.exp(-5.0 * s)) * (0.5 + 0.5 * rho),
    "rec":   lambda s, n_input, d, rho: 50.0 + 0.001 * n_input,
}


# Default rho when not specified. 1.0 means "no penalty".
DEFAULT_RHO = 1.0


def cost_vector(
    strategy: str,
    selectivity: float,
    n_input: int,
    dim: int,
    rho: float = DEFAULT_RHO,
) -> CostVector:
    """Compute the 4-tuple cost for a vector range scan.

    Parameters
    ----------
    strategy : {"flat", "ivf", "hnsw"}
        The ANN strategy.
    selectivity : float
        Estimated vector range selectivity (s = |sigma(R)| / |R|).
    n_input : int
        Number of rows the scan actually sees (post-classical-filter).
    dim : int
        Vector dimensionality.
    rho : float
        HNSW connectivity ratio of the surviving subset (Section 6.2.2).
        1.0 means no fragmentation penalty. Default 1.0.

    Returns
    -------
    CostVector
    """
    table = {"flat": COST_FLAT, "ivf": COST_IVF, "hnsw": COST_HNSW}[strategy]
    s = max(0.0, min(1.0, selectivity))
    r = max(0.0, min(1.0, rho))
    return CostVector(
        time_ms=table["time"](s, n_input, dim, r),
        memory_mb=table["memory"](s, n_input, dim, r),
        recall=table["recall"](s, n_input, dim, r),
        rec=table["rec"](s, n_input, dim, r),
    )


# ---------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------

# rho threshold below which HNSW is demoted to flat scan over R_f.
# When the surviving subset fragments the HNSW graph by more than this
# factor, pre-filtered HNSW is more expensive than a flat scan over
# the same surviving set.
RHO_DEMOTE_THRESHOLD = 0.3


def cbo_pick(
    selectivity: float,
    n: int,
    dim: int,
    recall_floor: float = 0.95,
    rho: float = DEFAULT_RHO,
    classical_s: float = 1.0,
) -> str:
    """Pick the best ANN strategy by *predicted cost* under the
    recall floor.

    Parameters
    ----------
    selectivity : float
        Vector range selectivity (s = |sigma(R)| / |R|).
    n : int
        Full relation size (for selectivity computation).
    dim : int
        Vector dimensionality.
    recall_floor : float
        Minimum acceptable recall for the chosen strategy.
    rho : float
        HNSW connectivity ratio of the surviving subset (Section 6.2.2).
        When rho < RHO_DEMOTE_THRESHOLD, HNSW is demoted.
    classical_s : float
        Selectivity of the upstream classical filter, in (0, 1].
        Determines the effective `n_input` for the vector scan.

    Strategy
    --------
    The CBO computes the 4-tuple cost for each of (flat, ivf,
    hnsw) using the cost model of Section 6.2 (rho-aware). It
    returns the strategy with the lowest time_ms that meets the
    recall floor. (An earlier version followed a fixed regime
    table; the cost-based comparison is more honest and makes the
    rho-aware cost model meaningful.)

    Section 6.2.2 adds: if rho < RHO_DEMOTE_THRESHOLD, HNSW is
    demoted regardless of selectivity, because pre-filtered HNSW
    on a fragmented surviving set is more expensive than a flat
    scan over the same R_f.
    """
    s = max(0.0, min(1.0, selectivity))
    s_c = max(1.0 / max(1, n), min(1.0, classical_s))
    n_input = max(1, int(round(s_c * n)))
    r = max(0.0, min(1.0, rho))

    # Compute cost for each strategy
    candidates = ["flat", "ivf", "hnsw"]
    best_strat: Optional[str] = None
    best_cost = float("inf")
    for strat in candidates:
        # Section 6.2.2: skip HNSW if rho is too low
        if strat == "hnsw" and r < RHO_DEMOTE_THRESHOLD:
            continue
        cv = cost_vector(strat, s, n_input, dim, r)
        if cv.recall < recall_floor:
            continue
        if cv.time_ms < best_cost:
            best_cost = cv.time_ms
            best_strat = strat

    if best_strat is not None:
        return best_strat
    # If nothing meets the floor, return the highest-recall option.
    return "flat"


# Strategy table summary (for the paper / plot legend)
STRATEGY_TABLE: Dict[str, Dict[str, str]] = {
    "flat":  {"regime": "s < 0.05",         "reason": "very few candidates; flat scan is fast and exact"},
    "ivf":   {"regime": "0.05 <= s < 0.3",  "reason": "moderate selectivity; IVF probes few clusters"},
    "hnsw":  {"regime": "s >= 0.3",         "reason": "high selectivity; HNSW gives best recall/latency"},
}


# ---------------------------------------------------------------------
# Convenience: pick and return both the strategy and the cost vector
# ---------------------------------------------------------------------

def pick_and_cost(
    selectivity: float,
    n: int,
    dim: int,
    recall_floor: float = 0.95,
    rho: float = DEFAULT_RHO,
    classical_s: float = 1.0,
) -> Tuple[str, CostVector]:
    strat = cbo_pick(selectivity, n, dim, recall_floor, rho, classical_s)
    n_input = max(1, int(round(classical_s * n)))
    return strat, cost_vector(strat, selectivity, n_input, dim, rho)


# Alias for the paper's notation
CBO_REC = cost_vector
