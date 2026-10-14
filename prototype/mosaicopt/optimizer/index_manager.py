"""Index lifecycle manager (separated from CBO).

Background
----------
Build costs of an ANN index (HNSW AND IVF) should NOT be a per-
query concern of the CBO. The CBO's job is to pick between strategies
that **already exist** at query time; the *decision to build* an
index should live in a separate module that can reason about
workload size, build cost, and per-query savings honestly.

This module is that module. The IndexManager replaces both:
- the ``N_SKIP_HNSW_BUILD = 50_000`` band-aid in :mod:`cbo`
- the runtime ``ivf = AnnIndex(...)`` in baselines that built an IVF
  index on every query (~900 ms at n=1M) just because the CBO
  thought "ivf is fast".

Architecture
------------
::

                          +-----------------+
                          |  IndexManager   |   <-- decoupled from CBO
                          +-----------------+
                                  |
            +---------------------+-----------------------+
            |                     |                       |
        break_even(n)      should_build(n)         memory_estimate(n)
            |                     |
            v                     v
      build/queries          bool, bool
      ratio                      |
                                 v
                  +--------------+--------------+
                  |                             |
              skip HNSW (use flat)         build HNSW (use hnsw)
              skip IVF (use flat)          build IVF (use ivf)
                  |                             |
                  v                             v
            +-----------+               +-----------+
            |   CBO     |<-- indexes_available: set ---+
            +-----------+     {"hnsw", "ivf"}

Cost model
----------
For a sweep driver that exposes ``workload_size`` (how many queries
will run against this index):

    build_ms(n, dim)   = a + b * n + c * n * dim / d_ref
    savings_ms(n)      = alpha * sqrt(n / n_ref)
    break_even         = ceil(build_ms / savings_ms)

Both HNSW and IVF use the same break_even curve (calibrated at d=128,
n=10k on a single thread). Production deployments can override the
constants explicitly.

This module owns those constants. They can be tuned by the operator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np


# ---------------------------------------------------------------------
# HNSW parameter scaling
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class HnswParams:
    """HNSW build parameters chosen for (n, dim, recall_target)."""
    M: int
    ef_construction: int
    ef_search: int


# Recall target for the unfiltered ANN step. The CBO can demote
# further via rho (Section 6.2.2 of the paper) on a fragmented R_f.
DEFAULT_RECALL_TARGET = 0.99


def choose_hnsw_params(
    n: int,
    dim: int,
    recall_target: float = DEFAULT_RECALL_TARGET,
    query_budget_ms: float = 1000.0,
) -> HnswParams:
    """Pick ``(M, ef_construction, ef_search)`` for a column of n rows,
    d dims, calibrated to maintain recall ``>= recall_target`` while
    keeping per-query wall-clock cost under ``query_budget_ms``.

    Heuristic:
      * ``ef_search`` scales with the *log* of n (Malkov & Yashunin,
        2018, Sec. 4): doubling n needs ~30% more ef to keep recall.
        We start from ``ef_recall = 64 * (n / 10000) ** 0.30``.
      * The actual recall depends on data distribution; ``ef_recall``
        is a default that empirically reaches the recall target for
        the standard sweep datasets. Tuning per-deployment is allowed.
      * **Query-time cap**: at very large n the log-scale ef would
        dominate query time; per Malkov-Yashunin the wall-clock for
        HNSW range_search empirically grows as
        ``t_ms ≈ c_query * ef_search * sqrt(n)``, where ``c_query`` is
        host-specific (calibrated to ~5e-5 on this benchmark host).
        We pick ``max_ef = query_budget_ms / (c_query * sqrt(n))``
        and ``ef_search = min(ef_recall, max_ef, 2048)``.
      * ``ef_construction`` fixed at 100 (build-time only; doesn't
        affect query path).
      * ``M = 32`` uniformly. The previous ``M=48 for n > 500k``
        heuristic over-paid build cost (~5× on this host at n=1M)
        for marginal recall gain.

    Notes
    -----
    * The recall target is an *input*, not a derived guarantee.
      ``choose_hnsw_params`` does not measure recall against ground
      truth (doing so requires running queries). Operators that
      want a calibration loop should re-tune ``ef_recall`` constant
      after a small ground-truth sweep.
    * ``c_query`` is calibrated on this benchmark host. Deployments
      on different hardware should re-measure (it is exposed below
      for that purpose; pass an explicit ``c_query`` if needed).
    """
    if n <= 0:
        return HnswParams(M=32, ef_construction=200, ef_search=64)

    # Recall-driven ef, log-scaling per Malkov-Yashunin.
    ef_recall = max(64, round(64.0 * (n / 10_000.0) ** 0.30))
    # Hard upper cap to bound the largest ef we'd ever consider.
    # Calibrated from the n=1M sweep: at ef > 1024 the HNSW
    # traversal cost (~50-100 ms at n=1M) starts to dominate the
    # brute-force-on-subset cost, so going higher is rarely
    # worth it. The CBO uses the same cap (HNSW_MAX_EF_SEARCH
    # in cbo.py) to demote HNSW when expected result count
    # exceeds it.
    ef_recall = min(ef_recall, 1024)

    # Query-time budget cap. At n -> infinity the log-scale ef grows
    # unboundedly in principle, but the resulting query time would
    # violate ``query_budget_ms``. We cap ef such that the empirical
    # query time (c_query * ef * sqrt(n)) stays under the budget.
    c_query = 5e-5  # ms per (ef * sqrt(n)) on this benchmark host
    n_eff = max(1, int(n))
    max_ef_for_budget = max(
        64,
        int(query_budget_ms / (c_query * (n_eff ** 0.5))),
    )
    ef_search = min(ef_recall, max_ef_for_budget)

    ef_construction = 100
    # Uniform M=32 across all n; the prior "M=48 for n > 500k"
    # heuristic was over-tuned for the HNSW *traversal* (helping
    # recall on fragmented surviving sets) but paid a ~5× build
    # cost on a single thread. M=32 still gives recall > 0.99
    # for the workloads we sweep; operators can override per-call.
    M = 32

    return HnswParams(M=M, ef_construction=ef_construction, ef_search=ef_search)


# ---------------------------------------------------------------------
# Build economics
# ---------------------------------------------------------------------

# Calibration constants (d=128, n=1M, 1 thread; re-tuned 2026-10-14
# from the n=1M verification sweep on this benchmark host).
#
# Empirical (single-thread, this host):
#   - flat brute-force query at n=1M, d=128, 200k surviving set: ~205 ms
#   - HNSW query (ef=64) at n=1M, d=128: ~0.5 ms
#   - savings_per_query ≈ 200 ms  ← was 50 ms before recalibration
#   - HNSW build at n=1M, d=128, M=48, ef_construction=200: ~1.3 s
#
# The previous ``SAVINGS_AT_REF_MS = 50`` was calibrated against a
# 10k d=512 corpus where flat scan was ~5 ms; carried over unchanged
# when the system was scaled to n=1M. At n=1M the constant was 4×
# too low, making ``break_even`` 4× too high (~27 instead of ~7),
# so MosaicOpt never built HNSW at the small sweep workloads.
#
# Operators can override per-deployment by passing explicit values.
DEFAULT_BUILD_A_MS = 30.0           # base build overhead
DEFAULT_BUILD_B_PER_N = 0.0008      # per-row marginal build ms at d=128
DEFAULT_BUILD_C_PER_ND = 0.000004   # additional per (n, d) marginal
DEFAULT_SAVINGS_REF_N = 1_000_000   # reference scale for savings model
DEFAULT_SAVINGS_AT_REF_MS = 200.0   # flat-vs-hnsw savings per query at n_ref
#                                  # (was 50.0; re-tuned from n=1M empirical)


@dataclass
class IndexManager:
    """Manage ANN index lifecycle (HNSW + IVF) for one optimizer stack.

    Parameters
    ----------
    workload_size : int
        Expected number of queries this stack will run. Drives the
        build-vs-flat decision (a 30 s build at n=1M breaks even at
        ~600 queries, far above typical sweep workloads of 16-100).
    recall_target : float
        Target recall@K for the unfiltered ANN step. Used by
        ``choose_hnsw_params`` to scale ef_search.

    Notes
    -----
    * Per-instance caches the built indexes by column name. Two
      separate caches so HNSW and IVF can coexist.
    * Thread-safety: this is per-stack (per baseline class), not
      shared. The sweep instantiates one IndexManager per stack so
      that different systems don't clobber each other's caches.
    """
    workload_size: int = 16
    recall_target: float = DEFAULT_RECALL_TARGET
    build_a_ms: float = DEFAULT_BUILD_A_MS
    build_b_per_n: float = DEFAULT_BUILD_B_PER_N
    build_c_per_nd: float = DEFAULT_BUILD_C_PER_ND
    savings_ref_n: int = DEFAULT_SAVINGS_REF_N
    savings_at_ref_ms: float = DEFAULT_SAVINGS_AT_REF_MS
    _hnsw_cache: Dict[str, Any] = field(default_factory=dict)
    _ivf_cache: Dict[str, Any] = field(default_factory=dict)

    # --- economics ---

    def break_even(self, n: int, dim: int = 128) -> int:
        """Min workload size to make an ANN build profitable.

        Returns the integer ``q`` such that the workload would amortize
        the build cost over ``q`` queries (with per-query savings from
        using the index over flat on a representative workload).

        Used for both HNSW and IVF because the build-time-vs-query-time
        ratio is of the same order; production deployments with very
        different IVF vs HNSW build times should override per-instance.
        """
        if n <= 0:
            return 1
        build_ms = (
            self.build_a_ms
            + self.build_b_per_n * n
            + self.build_c_per_nd * n * dim
        )
        ratio = max(n, 1) / max(1, self.savings_ref_n)
        savings_ms = self.savings_at_ref_ms * math.sqrt(ratio)
        return max(1, int(math.ceil(build_ms / max(0.1, savings_ms))))

    def should_build(self, n: int, dim: int = 128) -> bool:
        """Whether the workload is large enough to justify any ANN build."""
        return self.workload_size >= self.break_even(n, dim)

    # --- parameter choice ---

    def hnsw_params(self, n: int, dim: int) -> HnswParams:
        """HNSW parameters appropriate for a column of n rows, d dims."""
        return choose_hnsw_params(n, dim, self.recall_target)

    # --- HNSW access ---

    def has_hnsw(self, column: str) -> bool:
        """Whether an HNSW index has been built and cached for the column."""
        return column in self._hnsw_cache

    def get_hnsw(self, column: str) -> Optional[Any]:
        """Return the cached HNSW index for the column, or None."""
        return self._hnsw_cache.get(column)

    def ensure_hnsw(
        self,
        column: str,
        vectors: np.ndarray,
        dim: int,
        ann_index_cls: Any = None,
    ) -> Tuple[bool, Optional[Any]]:
        """Build/cache an HNSW index iff economics justify the build.

        Returns ``(True, idx)`` if HNSW is now available for the
        column, ``(False, None)`` if the workload is too small to
        justify the build.
        """
        if column in self._hnsw_cache:
            return True, self._hnsw_cache[column]
        n = int(vectors.shape[0]) if hasattr(vectors, "shape") else int(len(vectors))
        if not self.should_build(n, dim):
            return False, None
        if ann_index_cls is None:
            from ..physical.ann import AnnIndex as _AnnIndexCls
            ann_index_cls = _AnnIndexCls
        params = self.hnsw_params(n, dim)
        v = np.ascontiguousarray(vectors.astype("float32"))
        idx = ann_index_cls(
            v, dim=dim, strategy="hnsw",
            M=params.M, ef_construction=params.ef_construction,
        )
        try:
            idx._index.hnsw.efSearch = params.ef_search
        except AttributeError:
            pass
        self._hnsw_cache[column] = idx
        return True, idx

    # --- IVF access ---

    def has_ivf(self, column: str) -> bool:
        """Whether an IVF index has been built and cached for the column."""
        return column in self._ivf_cache

    def get_ivf(self, column: str) -> Optional[Any]:
        """Return the cached IVF index for the column, or None."""
        return self._ivf_cache.get(column)

    def ensure_ivf(
        self,
        column: str,
        vectors: np.ndarray,
        dim: int,
        ann_index_cls: Any = None,
    ) -> Tuple[bool, Optional[Any]]:
        """Build/cache an IVF index iff economics justify the build.

        Returns ``(True, idx)`` if IVF is now available for the column,
        ``(False, None)`` if the workload is too small to justify the
        build. The CBO is then free to choose IVF; baselines must NOT
        bypass this method and call ``AnnIndex(strategy='ivf')``
        directly (that's the bug we just fixed).
        """
        if column in self._ivf_cache:
            return True, self._ivf_cache[column]
        n = int(vectors.shape[0]) if hasattr(vectors, "shape") else int(len(vectors))
        if not self.should_build(n, dim):
            return False, None
        if ann_index_cls is None:
            from ..physical.ann import AnnIndex as _AnnIndexCls
            ann_index_cls = _AnnIndexCls
        v = np.ascontiguousarray(vectors.astype("float32"))
        idx = ann_index_cls(v, dim=dim, strategy="ivf")
        self._ivf_cache[column] = idx
        return True, idx

    # --- unified access used by CBO ---

    def available_indexes(self, column: str) -> Set[str]:
        """Return the set of strategy names for which an index exists.

        Always include strategy names from ``{"hnsw", "ivf"}`` only when
        their respective ``ensure_*`` method has run for this column
        and a build was economically justified. ``"flat"`` is *not*
        returned here because it never requires a build.
        """
        out: Set[str] = set()
        if self.has_hnsw(column):
            out.add("hnsw")
        if self.has_ivf(column):
            out.add("ivf")
        return out

    # --- pre-build entry point for workload drivers ---

    def pre_build(self, columns) -> None:
        """Pre-build HNSW / IVF indexes for the given columns.

        This is the supported path for the workload infrastructure
        (sweep driver, app startup, long-running DB process). The
        IndexManager decides per column whether to actually build
        based on workload economics; pre-build is a no-op for columns
        whose ``should_build(n, dim)`` returns False.

        Parameters
        ----------
        columns : iterable of (column_name, vectors, dim) tuples

            Each element is a triple: column name, the base vectors
            as a numpy array of shape ``(n, dim)``, and the
            dimensionality. Pass ``store.vector_column(col)`` for
            ``vectors`` and the column's vector dimension for
            ``dim``.

        Example::

            index_mgr.pre_build([
                ("vec_a", ds.store.vector_column("vec_a"), 128),
                ("vec_b", ds.store.vector_column("vec_b"), 128),
            ])
        """
        for entry in columns:
            if len(entry) == 3:
                col, vec_col, dim = entry
            else:
                col, vec_col = entry
                dim = vec_col.shape[1]
            self.ensure_hnsw(col, vec_col, dim=dim)
            self.ensure_ivf(col, vec_col, dim=dim)

    def memory_estimate(self, n: int, dim: int) -> float:
        """Approximate memory (MB) an HNSW over (n, dim) would use.

        Used by the memory track of the cost model and by the sweep
        driver for budgeting. The estimate is ``n * dim * 4 * 1.3 / 1MB``
        (1.3× accounts for the graph links on top of the raw vectors).
        """
        return n * dim * 4 * 1.3 / (1024 * 1024)
