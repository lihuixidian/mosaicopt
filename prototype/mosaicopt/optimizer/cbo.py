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
    """3-tuple cost: (time_ms, memory_mb, recall).

    The deprecated 4-tuple ``(time, memory, recall, rec)`` carried a
    ``rec`` field for "construction time" that the CBO never actually
    compared on. That field has been removed; build-time reasoning
    belongs to :class:`mosaicopt.optimizer.index_manager.IndexManager`.
    """
    time_ms: float
    memory_mb: float
    recall: float

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.time_ms, self.memory_mb, self.recall)


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
# Cost: empirically ``n_input * d * 6e-6 ms`` per pass on this host.
# This is calibrated so that:
#   n_input = 10000,  d = 128  -> ~7.7 ms
#   n_input = 200000, d = 128 -> ~153 ms (real ~171 ms in our smoke run)
#   n_input = 10000,  d = 768 -> ~46 ms (matches old formula)
# Previously this was ``0.04 + 0.06 * n_input * d / 1e6`` which gave
# the wrong absolute at n=1M and was the cause of CBO picking IVF
# over flat under the (now-removed) build-on-demand assumption.
COST_FLAT = {
    "time":  lambda s, n_input, d, rho: 0.04 + 6.0e-6 * n_input * d,
    "memory":lambda s, n_input, d, rho: n_input * d * 4 / (1024 * 1024),
    "recall":lambda s, n_input, d, rho: 1.0,
}

# IVF - moderate recall at moderate selectivity. The CBO will only
# offer IVF when the IndexManager has built one (workload breaks even);
# the time formula here is per-query search cost assuming the index
# already exists (no build). nlist and nprobe are at their defaults
# (40, nlist/4) which gives the 0.98 recall form below.
COST_IVF = {
    "time":  lambda s, n_input, d, rho: 0.5 + 5.0e-6 * n_input * d,
    "memory":lambda s, n_input, d, rho: n_input * d * 4 / (1024 * 1024) * 0.20,
    "recall":lambda s, n_input, d, rho: 0.98 - 0.02 * math.exp(-30.0 * s),
}

# HNSW - fast at high s. Cost is amplified by 1/rho when the
# surviving subset fragments the HNSW graph (Section 6.2.2).
# Empirically ``0.1 + d * 0.5e-3 ms`` per query at this host. Build
# cost is *not* in this formula; the IndexManager decides whether
# an HNSW exists at all.
COST_HNSW = {
    "time":  lambda s, n_input, d, rho: (0.1 + 0.5e-3 * d) / max(0.05, rho),
    "memory":lambda s, n_input, d, rho: n_input * d * 4 / (1024 * 1024) * 1.30,
    # When rho is small, HNSW's recall drops because the graph is
    # fragmented within R_f; the recall floor of the strategy degrades.
    "recall":lambda s, n_input, d, rho: (1.0 - 0.04 * math.exp(-5.0 * s)) * (0.5 + 0.5 * rho),
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
    )


# ---------------------------------------------------------------------
# Strategy selection
# ---------------------------------------------------------------------

# rho threshold below which HNSW is demoted to flat scan over R_f.
# When the surviving subset fragments the HNSW graph by more than this
# factor, pre-filtered HNSW is more expensive than a flat scan over
# the same surviving set. This is a *per-query* decision that the CBO
# owns; the *build decision* (whether the HNSW even exists) belongs to
# IndexManager.
RHO_DEMOTE_THRESHOLD = 0.3


def cbo_pick(
    selectivity: float,
    n: int,
    dim: int,
    *,
    indexes_available=(),
    hnsw_exists: bool = False,
    recall_floor: float = 0.95,
    rho: float = DEFAULT_RHO,
    classical_s: float = 1.0,
) -> str:
    """Pick the best *already-existing* ANN strategy by predicted cost.

    Parameters
    ----------
    selectivity : float
        Vector range selectivity (s = |sigma(R)| / |R|).
    n : int
        Full relation size (for selectivity computation).
    dim : int
        Vector dimensionality.
    indexes_available : iterable of {"hnsw", "ivf"}
        Canonical new-style: which ANN indexes has the
        :class:`mosaicopt.optimizer.index_manager.IndexManager`
        already built for this column? The CBO's job is to pick
        among ``"flat"`` plus any indexes in this set. The CBO does
        NOT decide to build an index; that's IndexManager's domain.
        Default empty tuple (no indexes available; only "flat").
    hnsw_exists : bool, default False
        **Back-compat shim** for experiment scripts written before
        IndexManager existed. Equivalent to ``"hnsw" in indexes_available``.
        New code should pass ``indexes_available={"hnsw"}``.
    recall_floor : float
        Minimum acceptable recall for the chosen strategy.
    rho : float
        HNSW connectivity ratio of the surviving subset (Section 6.2.2).
        When rho < ``RHO_DEMOTE_THRESHOLD``, HNSW is demoted even if it
        exists.
    classical_s : float
        Selectivity of the upstream classical filter, in (0, 1].
        Determines the effective ``n_input`` for the vector scan.

    Returns
    -------
    strategy : {"flat", "ivf", "hnsw"}
        The strategy with the lowest ``time_ms`` that meets the
        recall floor and (if "hnsw") passes the rho demote test.
        Falls back to "flat" when no candidate qualifies.
    """
    s = max(0.0, min(1.0, selectivity))
    s_c = max(1.0 / max(1, n), min(1.0, classical_s))
    n_input = max(1, int(round(s_c * n)))
    r = max(0.0, min(1.0, rho))

    # Build the candidate set: every available index is a real option;
    # the CBO does NOT construct indexes on the fly. The flat strategy
    # is always available (no index build required).
    available = set(indexes_available)
    if hnsw_exists:
        available.add("hnsw")
    candidates = ["flat"]
    if "hnsw" in available:
        candidates.append("hnsw")
    if "ivf" in available:
        candidates.append("ivf")

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


# Strategy table summary (DEPRECATED as a CBO driver; kept for paper
# cross-reference). The current :func:`cbo_pick` is *cost-based* and
# does not consult this dict. The ``"ivf"`` entry is retained for
# historical reference but **the CBO will not offer IVF to a query
# unless the IndexManager has built an IVF index for that column**
# (i.e., the workload was large enough to amortize the build cost).
# In practice this means IVF is rarely picked in sweep scenarios and
# only matters in production workloads with workload > break_even.
STRATEGY_TABLE: Dict[str, Dict[str, str]] = {
    "flat":  {"regime": "s < 0.05",         "reason": "very few candidates; flat scan is fast and exact"},
    "ivf":   {"regime": "0.05 <= s < 0.3",  "reason": "moderate selectivity; IVF probes few clusters (rare in sweep; IndexManager gates)"},
    "hnsw":  {"regime": "s >= 0.3",         "reason": "high selectivity; HNSW gives best recall/latency when the IndexManager has built one"},
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
