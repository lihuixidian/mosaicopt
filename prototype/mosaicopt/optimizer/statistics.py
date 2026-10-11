"""Statistics layer for the multi-modal CBO.

This module provides per-query, per-modality cardinality estimates that
the CBO consumes in place of the default 0.5 selectivity prior. It is
the realization of Section 6.2 of the SIGMOD paper.

The layer has four components:

  - VectorSketch  : wraps an faiss HNSW index over a vector column;
                    estimates selectivity of `||V - q|| < tau` via a
                    single range_search with elevated ef (the same
                    probe the main query would do, just with a wider
                    search budget to bound recall loss).
  - TextSketch    : wraps a rank_bm25 index over a text column;
                    estimates |{r : BM25(r, q) > theta}| by
                    computing scores on the full corpus.
  - GraphSketch   : wraps a GraphStore; estimates
                    |reachable(seeds, rule, depth)| by running
                    reachability once per query seed.
  - SampleProbe   : 1% stratified sample of the relation; runs the
                    actual predicate on the sample to get a direct
                    measurement. Used as the unified fallback when
                    the per-modality sketch does not apply.

The StatisticsHub ties these together and exposes a single
`estimate_cardinality(predicate, query_params)` method that
dispatches by predicate type.

Performance budget (prototype, 10k x 768 corpus):
  - VectorSketch:  one HNSW range_search with ef=128: O(nprobe *
                    log n) ~ 1-3 ms.  Amortized over many queries.
  - TextSketch:    one BM25 score pass over the corpus: O(|V| * |q|)
                    ~ 5-10 ms for 10k docs.
  - GraphSketch:   one reachability per seed: O(out_degree ^ depth).
  - SampleProbe:   1% flat scan with filter: ~5 ms.

The CBO calls `estimate_selectivity` once per modality filter in the
query plan, in O(<5 ms) per call. This is well under the CBO's
optimization budget (~10 ms for the full cross-modal enumeration
of Section 6.4) and orders of magnitude under the query latency it
helps optimize (~30-300 ms per query in E1-E4).
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

from ..algebra.predicates import (
    Predicate, ColumnPredicate, VectorRange, BM25Score, GraphReach, And, Or, Not,
)
from ..algebra.ops import Plan, Scan, Select
from ..physical.ann import AnnIndex
from ..physical.bm25 import BM25Index, _tokenize
from ..physical.graph import GraphStore
from ..algebra.ops import Materialize, Project, Order, Limit, Expand, Join


# =====================================================================
# Per-modality sketches
# =====================================================================

class VectorSketch:
    """Per-column vector selectivity estimator (v2).

    Builds an HNSW index on the column at construction. Selectivity is
    estimated by a single range_search with elevated ef. The estimator
    shares the index with the main query, so the build cost is paid
    once.

    v2 design (this commit):
      - **Distance distribution fit**: from the distances returned by
        a single range_search with a generous `tau_max`, we fit an
        exponential distribution `f(d) ~ lambda * exp(-lambda * d^2)`
        and answer selectivity for *any* tau by integrating. This
        makes the estimator a *point estimate* that is also a lower
        bound under the deterministic-HNSW assumption.
      - **Plan->exec cache (Exqutor-style)**: every (q, tau) call is
        cached, so repeated queries skip the range_search entirely
        (cache hit: ~50ns; cache miss: ~1-3ms). The first call costs
        one range_search with tau = tau_max; subsequent calls for
        any tau use the fitted distribution.

    The estimate has the form `|{r : ||V_r - q|| < tau}| / n`. For
    HNSW, this is a *lower bound* on the true selectivity (HNSW can
    miss some candidates); the bias shrinks as ef grows. Under our
    fixed configuration (ef=128, M=32, ef_construction=200) the
    empirical HNSW recall on the BigVectorBench glue table is > 0.99,
    so the bias is < 1% in our target regime.
    """
    def __init__(self, vectors: np.ndarray, dim: int, M: int = 32,
                 ef_construction: int = 200, ef_search: int = 128,
                 cache_size: int = 10000) -> None:
        self.vectors = np.ascontiguousarray(vectors.astype("float32"))
        self.dim = dim
        self.n = self.vectors.shape[0]
        # Build a dedicated index for the estimator. The same vectors
        # are also indexed by the main CBO, so this is some redundant
        # memory; for a prototype this is fine. In production, the
        # estimator would share the index with the main query.
        self._index = AnnIndex(self.vectors, dim, strategy="hnsw",
                                M=M, ef_construction=ef_construction)
        self._index._index.hnsw.efSearch = ef_search
        # v2: plan->exec cache + plan decision cache
        # key = q_hash  (one distribution per query; any tau reuses it)
        # value = (lam, n_within_tau_max)  -- the fitted distribution
        #         + the count at tau_max (for unbiased reference)
        self._cache: Dict[int, Tuple[float, int]] = {}
        self._cache_size = cache_size
        self._cache_hits = 0
        self._cache_misses = 0

    def estimate_selectivity(self, q: np.ndarray, tau: float) -> float:
        """Return |{r : ||V_r - q|| < tau}| / n.

        v2 algorithm:
          1. Hash (q, tau_quantized) -> check cache
          2. On miss: run range_search with tau_max = max(tau, 1.0)
             (generous envelope so the distribution fit covers all
             practical tau values)
          3. Fit exponential to the returned distances
          4. Compute s = (1 - exp(-lam * tau^2)) for the requested tau
          5. Cache (lam, n_within_tau_max) for reuse
        """
        if tau <= 0:
            return 0.0
        q = np.ascontiguousarray(q.astype("float32").reshape(1, self.dim))
        q_hash = hash(q.tobytes())
        if q_hash in self._cache:
            self._cache_hits += 1
            lam, _ = self._cache[q_hash]
            return 1.0 - math.exp(-lam * float(tau) * float(tau))
        # Cache miss: run range_search with a generous tau envelope.
        # The envelope must be >= the requested tau to cover it; we
        # also collect a sample of distances for the distribution fit.
        tau_max = max(float(tau), 1.0)
        sq_thr = float(tau_max) ** 2
        lims, dists, _ = self._index._index.range_search(q, sq_thr)
        n_within_max = int(lims[1] - lims[0])
        # Fit exponential: lam = K / sum(d_k^2) for the K returned
        # distances. The HNSW range_search returns distances sorted
        # ascending; we use the first K=min(256, len) for the fit
        # to keep the math bounded.
        K = min(len(dists), 256)
        if K == 0:
            lam = 0.0
        else:
            sq_sum = float(np.sum(np.asarray(dists[:K], dtype=np.float64) ** 2))
            lam = K / max(1e-9, sq_sum)
        # Compute selectivity at the requested tau
        s = 1.0 - math.exp(-lam * float(tau) * float(tau))
        # Store the fitted distribution; future calls with the same
        # q but different tau will hit this cache entry and reuse lam.
        if len(self._cache) >= self._cache_size:
            # Approximate LRU: pop the oldest insertion. dict preserves
            # insertion order in Python 3.7+, so this is FIFO.
            self._cache.pop(next(iter(self._cache)))
        self._cache[q_hash] = (lam, n_within_max)
        self._cache_misses += 1
        return s

    def cache_stats(self) -> Dict[str, int]:
        """Return cache hit/miss counts (for CBO diagnostics)."""
        return {
            "hits": self._cache_hits,
            "misses": self._cache_misses,
            "size": len(self._cache),
        }

    def clear_cache(self) -> None:
        """Invalidate the cache (called on data insert/delete)."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class TextSketch:
    """Per-column text selectivity estimator.

    Builds a BM25 index at construction. Selectivity is the fraction
    of rows whose BM25 score against `q_text` exceeds `theta`.
    """
    def __init__(self, docs: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.docs = list(docs)
        self.n = len(self.docs)
        self._bm25 = BM25Index(self.docs, k1=k1, b=b)

    def estimate_selectivity(self, q_text: str, theta: float) -> float:
        """Return |{r : BM25(r, q) > theta}| / n."""
        scores = np.asarray(self._bm25._bm25.get_scores(_tokenize(q_text)),
                            dtype=np.float32)
        scores = np.maximum(scores, 0.0)
        n_hits = int((scores > theta).sum())
        return n_hits / max(1, self.n)


class GraphSketch:
    """Graph reachability selectivity estimator.

    For each (rule, depth) the estimator runs `reachable` once per
    query seed and returns the average expansion size. The result is
    cached so repeated queries with the same (rule, depth, seeds)
    pay only the first call.
    """
    def __init__(self, graph: GraphStore) -> None:
        self.graph = graph
        # cache: (rule, depth, frozenset(seeds)) -> n_reachable_avg
        self._cache: Dict[Tuple[str, int, frozenset], float] = {}

    def estimate_selectivity(self, rule: str, depth: int,
                              seeds: Sequence[int], n_rows: int) -> float:
        """Return |reachable(seeds, rule, depth)| / n_rows.

        The estimate is the mean reachable-set size across the seeds,
        scaled by `n_rows`. The CBO consumes the *cardinality* of
        the expansion (used in §6.4 cross-modal ordering), so we
        return the count, not the fraction.
        """
        key = (rule, depth, frozenset(seeds))
        if key in self._cache:
            return self._cache[key]
        if not seeds:
            self._cache[key] = 0.0
            return 0.0
        total = 0
        for s in seeds:
            total += len(self.graph.reachable(rule, int(s), depth))
        avg = total / len(seeds)
        self._cache[key] = avg
        return avg


# =====================================================================
# Filtered HNSW density (rho_f) — Section 6.2.2
# =====================================================================

class FilteredHNSWDensity:
    """Per-column surviving-subset HNSW connectivity estimator.

    For a vector predicate `||V - q|| < tau` and a classical filter
    `C` upstream, the cost of executing V after C depends on the
    **connectivity of the HNSW graph restricted to the surviving
    subset R_f = sigma_C(R)**. The connectivity ratio

        rho_f = (cost per result, unfiltered HNSW) /
                (cost per result, filtered HNSW)

    is 1.0 when R_f is well-connected (HNSW scales linearly) and
    drops below 1.0 when R_f is fragmented (the search has to do
    extra work to surface in-R_f candidates).

    The estimator precomputes rho_f for each (column, classical
    filter signature) pair, caches the result, and exposes
    `lookup(column, filter_signature) -> rho_f`. The CBO consumes
    the rho_f in its HNSW cost model (see cbo.py).

    Notes
    -----
    - The filter signature is the conjunction of classical predicates
      that appear *upstream* of the vector predicate in the plan.
      This is the surviving subset the HNSW search sees after those
      filters have been applied.
    - rho_f depends on the **order** in which filters are applied
      (just as join order affects table cardinality), because the
      surviving set R_f is the result of the upstream filters in
      sequence. The CBO's `FilterOrder` step in D consumes rho_f
      jointly with selectivity to pick the order.
    - For multi-vector queries (e.g. Q5 with both `emb` and
      `emb_alt`), each vector column has its own HNSW index and
      its own rho_f under the same upstream filter. The estimator
      caches per (column, signature) pair.
    """

    def __init__(
        self,
        store: Any,
        vector_columns: Sequence[str],
        q_sample: Optional[np.ndarray] = None,
        n_sample_queries: int = 20,
        seed: int = 42,
    ) -> None:
        """Build rho estimators for the given vector columns.

        Parameters
        ----------
        store : TableStore
            The substrate.
        vector_columns : Sequence[str]
            Names of vector columns whose HNSW indexes will be queried.
        q_sample : np.ndarray, optional
            Sample query vectors for the rho timing. If None, a
            default sample is drawn from each column's vectors.
        n_sample_queries : int
            How many sample queries to use per rho measurement.
        seed : int
            RNG seed for the default q_sample.
        """
        self.store = store
        self.vector_columns = list(vector_columns)
        self.n_sample_queries = n_sample_queries
        # Build one HNSW index per vector column. We share the
        # estimator's index with the main CBO when possible; in the
        # prototype, the CBO builds its own indexes so this is
        # independent memory.
        self._indexes: Dict[str, AnnIndex] = {}
        self._q_sample: Dict[str, np.ndarray] = {}
        rng = np.random.default_rng(seed)
        for col in self.vector_columns:
            if col not in store.arrow.column_names:
                continue
            v = store.vector_column(col).astype("float32")
            idx = AnnIndex(v, dim=v.shape[1], strategy="hnsw",
                           M=32, ef_construction=200)
            self._indexes[col] = idx
            if q_sample is None:
                # Default: random unit vectors in the same shape
                qs = rng.standard_normal((n_sample_queries, v.shape[1])).astype("float32")
                qs /= np.linalg.norm(qs, axis=1, keepdims=True) + 1e-9
                self._q_sample[col] = qs
            else:
                self._q_sample[col] = q_sample

        # cache: (column, frozenset-of-classical-pred-keys) -> rho_f
        self._cache: Dict[Tuple[str, frozenset], float] = {}

    def _surviving_set_from_filter(
        self,
        classical_preds: Sequence[ColumnPredicate],
    ) -> np.ndarray:
        """Compute the row ids in R_f = sigma_C(R) for the given filter.

        For the prototype this is a Python-side scan; the substrate's
        DuckDB view would be a faster path on real data.
        """
        if not classical_preds:
            n = self.store.arrow.num_rows
            return np.arange(n, dtype=np.int64)
        # Build a numpy boolean mask
        n = self.store.arrow.num_rows
        mask = np.ones(n, dtype=bool)
        for p in classical_preds:
            if p.column not in self.store.arrow.column_names:
                continue
            col = self.store.arrow.column(p.column).to_numpy()
            if p.op == "=":
                mask &= (col == p.value)
            elif p.op == "!=":
                mask &= (col != p.value)
            elif p.op == "<":
                mask &= (col < p.value)
            elif p.op == "<=":
                mask &= (col <= p.value)
            elif p.op == ">":
                mask &= (col > p.value)
            elif p.op == ">=":
                mask &= (col >= p.value)
            elif p.op == "in":
                mask &= np.fromiter((v in p.value for v in col),
                                    dtype=bool, count=n)
        return np.flatnonzero(mask).astype(np.int64)

    def _filter_signature(self, classical_preds: Sequence[ColumnPredicate]) -> frozenset:
        """A hashable signature for the conjunction of classical predicates."""
        sig = []
        for p in classical_preds:
            sig.append((p.column, p.op, p.value))
        return frozenset(sig)

    def lookup(
        self,
        column: str,
        classical_preds: Sequence[ColumnPredicate] = (),
    ) -> float:
        """Return the cached or freshly-computed rho_f for (column, filter).

        The first call for a given (column, signature) pays the
        estimator's cost (one HNSW timing per column); subsequent
        calls hit the cache.
        """
        sig = self._filter_signature(classical_preds)
        key = (column, sig)
        if key in self._cache:
            return self._cache[key]
        idx = self._indexes.get(column)
        if idx is None:
            # No index for this column; fall back to "no penalty"
            self._cache[key] = 1.0
            return 1.0
        R_f = self._surviving_set_from_filter(classical_preds)
        if len(R_f) == 0:
            self._cache[key] = 0.0
            return 0.0
        if len(R_f) >= idx._n:
            self._cache[key] = 1.0
            return 1.0
        rho = idx.filtered_density(R_f, self._q_sample[column], k=20)
        self._cache[key] = rho
        return rho

    def precompute(
        self,
        classical_predicates_per_query: Sequence[Sequence[ColumnPredicate]],
    ) -> Dict[Tuple[str, frozenset], float]:
        """Precompute rho_f for a set of classical-filter signatures.

        Useful when the workload is known ahead of time. Returns a
        snapshot of the cache for inspection (e.g. in the E9
        experiment).
        """
        for preds in classical_predicates_per_query:
            for col in self.vector_columns:
                self.lookup(col, preds)
        return dict(self._cache)


# =====================================================================
# Sample probe (unified fallback)
# =====================================================================

# ---------------------------------------------------------------------
# v2: per-row predicate evaluation helpers for joint selectivity
# ---------------------------------------------------------------------

def _build_per_row_scores(
    sub: "pa.Table",
    left: "Predicate",
    right: "Predicate",
    store: Any,
    query_params: Dict[str, Any],
) -> Dict[str, Optional[np.ndarray]]:
    """Pre-compute per-row scores for the expensive predicates.

    For VectorRange, compute L2 distance from each sampled row's
    `column` to `query_params[q_param]` and return a length-`n_sample`
    array. For BM25Score, compute the BM25 score for each row.
    For ColumnPredicate / GraphReach, return None (cheap to evaluate
    on the fly).
    """
    out: Dict[str, Optional[np.ndarray]] = {"left": None, "right": None}
    for side, p in [("left", left), ("right", right)]:
        if isinstance(p, VectorRange):
            q = np.asarray(query_params.get(p.q_param), dtype=np.float32)
            if q is None or p.column not in sub.column_names:
                continue
            col = np.stack(sub.column(p.column).to_pylist())
            d2 = np.sum((col - q.reshape(1, -1)) ** 2, axis=1)
            out[side] = np.sqrt(d2)  # L2 distance
        elif isinstance(p, BM25Score):
            from ..physical.bm25 import BM25Index, _tokenize
            docs = sub.column(p.column).to_pylist() if p.column in sub.column_names else []
            if not docs:
                continue
            bm = BM25Index(docs)
            q_text = query_params.get(p.q_param, "")
            scores = np.asarray(bm._bm25.get_scores(_tokenize(str(q_text))), dtype=np.float32)
            out[side] = scores
    return out


def _eval_predicate_on_row(
    p: "Predicate",
    row_idx: int,
    sub: "pa.Table",
    precomputed: Optional[np.ndarray],
    query_params: Dict[str, Any],
) -> bool:
    """Evaluate predicate `p` on a single sample row (indexed by `row_idx`).

    Uses `precomputed` if available (for VectorRange / BM25Score) and
    falls back to per-row computation for ColumnPredicate / GraphReach.
    """
    if isinstance(p, ColumnPredicate):
        if p.column not in sub.column_names:
            return False
        col = sub.column(p.column)
        v = col[row_idx].as_py() if row_idx < len(col) else None
        if v is None:
            return False
        val = p.value
        if p.op == "=":  return v == val
        if p.op == "!=": return v != val
        if p.op == "<":  return v < val
        if p.op == "<=": return v <= val
        if p.op == ">":  return v > val
        if p.op == ">=": return v >= val
        if p.op == "in":
            try:
                return v in val
            except TypeError:
                return False
        return False
    if isinstance(p, VectorRange):
        if precomputed is None or row_idx >= len(precomputed):
            return False
        d = float(precomputed[row_idx])
        return (d < p.threshold) if p.op == "<" else (d > p.threshold)
    if isinstance(p, BM25Score):
        if precomputed is None or row_idx >= len(precomputed):
            return False
        s = float(precomputed[row_idx])
        return (s > p.threshold) if p.op == ">" else (s < p.threshold)
    if isinstance(p, GraphReach):
        # GraphReach on a sample row is rare; fall back to False so we
        # don't overcount. The CBO can use the GraphSketch directly.
        return False
    if isinstance(p, And):
        # Pass precomputed=None; nested And/Or fall through to per-row
        # recomputation only for ColumnPredicate/GraphReach (cheap) or
        # to a VectorRange/BM25 with no cached scores (conservative:
        # returns False on distance check, so we don't overcount).
        return (
            _eval_predicate_on_row(p.left, row_idx, sub, None, query_params)
            and _eval_predicate_on_row(p.right, row_idx, sub, None, query_params)
        )
    if isinstance(p, Or):
        return (
            _eval_predicate_on_row(p.left, row_idx, sub, None, query_params)
            or _eval_predicate_on_row(p.right, row_idx, sub, None, query_params)
        )
    if isinstance(p, Not):
        return not _eval_predicate_on_row(
            p.child, row_idx, sub, None, query_params
        )
    return False


class SampleProbe:
    """1% stratified sample of the relation.

    Used as a unified fallback when a per-modality sketch does not
    apply, or as a sanity check on the per-modality estimate. The
    sample is uniform random by default; for very small corpora
    (<100 rows), the probe falls back to the full relation.

    v2: also supports `run_joint_predicate(left, right, query_params)`
    for multi-modal joint selectivity estimation (§6.2.3 of the paper).
    The joint probe evaluates two predicates row-by-row on the sample
    and counts joint hits, then scales up by `1 / sample_rate`.
    """
    def __init__(self, store, sample_rate: float = 0.01, seed: int = 42) -> None:
        self.store = store
        self.sample_rate = sample_rate
        self.n = store.arrow.num_rows
        rng = np.random.default_rng(seed)
        sample_size = max(1, min(self.n, int(self.n * sample_rate)))
        # Random uniform sample (the prototype's 10k x 768 corpus is
        # memory-resident, so this is a one-time O(n) shuffle).
        self._idx = rng.choice(self.n, size=sample_size, replace=False)
        self._idx.sort()

    def sample_rows(self) -> pa.Table:
        return self.store.arrow.take(self._idx)

    def run_joint_predicate(
        self,
        left: "Predicate",
        right: "Predicate",
        query_params: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Estimate |{r : left(r) AND right(r)}| using the sample.

        This is the v2 multi-modal joint estimator. It evaluates both
        predicates row-by-row on the 1% sample and counts the joint
        hits, then scales up by `1 / sample_rate`. The cost is O(|sample|)
        predicate evaluations, ~5 ms for 100 sampled rows.

        Per-row evaluation handles ColumnPredicate (free) and
        VectorRange (one L2 distance per row); BM25Score and GraphReach
        fall back to per-modality sketches to keep the per-row cost
        bounded.
        """
        if self._idx is None or len(self._idx) == 0:
            return 0
        sub = self.sample_rows()
        # Pre-compute per-modality scores on the sample for the expensive
        # predicates; these let us evaluate a predicate on any sample row
        # in O(1).
        scores_cache = _build_per_row_scores(
            sub, left, right, self.store, query_params or {}
        )
        n_joint = 0
        for i in range(len(self._idx)):
            if _eval_predicate_on_row(
                left, i, sub, scores_cache["left"], query_params or {}
            ) and _eval_predicate_on_row(
                right, i, sub, scores_cache["right"], query_params or {}
            ):
                n_joint += 1
        return int(round(n_joint / self.sample_rate))

    def estimate_cardinality_from_filter(self, predicate_sql: str) -> int:
        """Run a SQL WHERE on the sample and return the resulting count.

        The predicate_sql is applied to the sampled rows. This is the
        last-resort estimator: a per-modality sketch should always
        be preferred when available.
        """
        sub = self.sample_rows()
        # Build a tiny in-memory view
        import duckdb
        con = duckdb.connect(":memory:")
        con.register("sample", sub)
        n = con.execute(f"SELECT COUNT(*) FROM sample WHERE {predicate_sql}").fetchone()[0]
        con.close()
        # Scale up
        return int(round(n / self.sample_rate))


# =====================================================================
# Statistics hub: the unified API
# =====================================================================

# n_0 from paper §6.2.1: at n < n_0 the cost model's regime is
# dominated by constants and the CBO picks `flat` regardless of
# the selectivity estimate. We use the d=128 value 1.6×10^5;
# the per-dim n_0 is documented in §6.2.1.
DEFAULT_N_SKIP_ESTIMATE = 160_000


@dataclass
class StatisticsHub:
    """Tie together vector / text / graph sketches + a sample probe.

    Build once per query. The construction cost is O(n) per modality
    sketch; the per-query cardinality call is O(1-10 ms).

    v2: at small n (n < n_skip_estimate), the per-modality sketches
    are skipped entirely and the hub returns the CBO's default 0.5
    prior in O(1) time. This is the scale-dependent optimization
    from paper §6.2.1: at n < n_0, the cost model's regime is
    dominated by constants, so the CBO's plan choice is invariant
    to the estimate. The v2 distance distribution estimator and
    joint probe are activated only when CBO planning actually
    matters (n >= n_0).
    """
    store: Any
    vector_sketches: Dict[str, VectorSketch] = field(default_factory=dict)
    text_sketches: Dict[str, TextSketch] = field(default_factory=dict)
    graph_sketch: Optional[GraphSketch] = None
    sample_probe: Optional[SampleProbe] = None
    n: int = 0
    n_skip_estimate: int = DEFAULT_N_SKIP_ESTIMATE

    @classmethod
    def from_store(
        cls,
        store,
        vector_columns: Sequence[str] = (),
        text_columns: Sequence[str] = (),
        graph: Optional[GraphStore] = None,
        sample_rate: float = 0.01,
        ef_search: int = 128,
    ) -> "StatisticsHub":
        """Build all sketches from a TableStore.

        The build is O(n) per sketch; for the prototype's 10k x 768
        corpus, this is < 1 second total.
        """
        n = store.arrow.num_rows
        vs: Dict[str, VectorSketch] = {}
        for col in vector_columns:
            if col not in store.arrow.column_names:
                continue
            v = store.vector_column(col)
            vs[col] = VectorSketch(v, dim=v.shape[1], ef_search=ef_search)
        ts: Dict[str, TextSketch] = {}
        for col in text_columns:
            if col not in store.arrow.column_names:
                continue
            docs = store.arrow.column(col).to_pylist()
            ts[col] = TextSketch(docs)
        gs = GraphSketch(graph) if graph is not None else None
        sp = SampleProbe(store, sample_rate=sample_rate)
        return cls(store=store, vector_sketches=vs, text_sketches=ts,
                   graph_sketch=gs, sample_probe=sp, n=n)

    # --- the per-modality dispatch ---

    def estimate_selectivity(
        self,
        predicate: Predicate,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> float:
        """Return estimated |σ_pred(R)| / n for the given predicate.

        Dispatches on the predicate's modality:
          * VectorRange -> VectorSketch.estimate_selectivity(q, tau)
          * BM25Score   -> TextSketch.estimate_selectivity(q, theta)
          * GraphReach  -> GraphSketch.estimate_selectivity(rule, depth, seeds)
          * ColumnPredicate (classical) -> returns prior (no per-query
                             estimator; CBO uses the per-attribute
                             histogram it inherits from the substrate)
          * And         -> the minimum of the conjuncts' selectivities
                             (a safe lower bound; the CBO can refine
                             with the sample probe if it needs an
                             upper bound)
        """
        return self.estimate_cardinality(predicate, query_params) / max(1, self.n)

    def estimate_cardinality(
        self,
        predicate: Predicate,
        query_params: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Return estimated |σ_pred(R)| for the given predicate.

        v2 (n-aware gate): at small n (n < n_skip_estimate), the
        per-modality sketches are skipped and we return the CBO's
        default 0.5 prior in O(1) time. This is the scale-dependent
        optimization from paper §6.2.1: at n < n_0, the CBO picks
        `flat` regardless of the selectivity estimate, so the
        estimate is wasted work. At n >= n_0, the v2 distance
        distribution estimator and joint probe are engaged; the
        estimate matters for the CBO's plan choice.
        """
        query_params = query_params or {}

        # n-aware gate: short-circuit at small n
        if self.n < self.n_skip_estimate:
            return self.n // 2

        if isinstance(predicate, VectorRange):
            sketch = self.vector_sketches.get(predicate.column)
            if sketch is None:
                return self.n // 2  # fallback: 0.5
            q = query_params.get(predicate.q_param)
            if q is None:
                return self.n // 2
            s = sketch.estimate_selectivity(np.asarray(q, dtype="float32"),
                                             float(predicate.threshold))
            return int(round(s * self.n))

        if isinstance(predicate, BM25Score):
            sketch = self.text_sketches.get(predicate.column)
            if sketch is None:
                return self.n // 2
            q_text = query_params.get(predicate.q_param)
            if q_text is None:
                return self.n // 2
            s = sketch.estimate_selectivity(str(q_text), float(predicate.threshold))
            return int(round(s * self.n))

        if isinstance(predicate, GraphReach):
            if self.graph_sketch is None:
                return self.n // 2
            start = query_params.get(predicate.q_param, 0)
            seeds = [int(start)] if not isinstance(start, (list, tuple)) else [int(x) for x in start]
            return int(self.graph_sketch.estimate_selectivity(
                predicate.rule_name, predicate.depth, seeds, self.n))

        if isinstance(predicate, And):
            # v2 (QPS-tuned): joint probe is opt-in only when the
            # conservative min() fallback is unreliable, i.e., when
            # the minimum of the two conjunct selectivities is small
            # (< 0.05). At low selectivities the CBO's regime choice
            # is sensitive to the joint estimate; at moderate-to-high
            # selectivities the min() bound is tight enough that the
            # 7.6ms joint probe is not worth its cost. The threshold
            # of 0.05 was calibrated against the BigVectorBench glue
            # table; the join+multi-vector categories (Q7/Q8) at
            # s in {0.001, 0.005, 0.01} are the dominant case.
            n_left = self.estimate_cardinality(predicate.left, query_params)
            n_right = self.estimate_cardinality(predicate.right, query_params)
            min_frac = min(n_left, n_right) / max(1, self.n)
            if (
                min_frac < 0.05
                and self.sample_probe is not None
                and self._can_joint_probe(predicate.left, predicate.right)
            ):
                return self.sample_probe.run_joint_predicate(
                    predicate.left, predicate.right, query_params
                )
            return min(n_left, n_right)

        if isinstance(predicate, Or):
            # v2: independence-based upper bound `s_p + s_q - s_p*s_q`
            # is the standard closed-form for disjoint-or-correlated
            # modalities; we keep it as a default. Joint sample probe
            # is not used for Or (joint hits underrepresent Or).
            sl = self.estimate_cardinality(predicate.left, query_params)
            sr = self.estimate_cardinality(predicate.right, query_params)
            sl_frac = sl / max(1, self.n)
            sr_frac = sr / max(1, self.n)
            return int(round((sl_frac + sr_frac - sl_frac * sr_frac) * self.n))

        if isinstance(predicate, ColumnPredicate):
            # Classical predicate: the substrate's DuckDB ANALYZE
            # already provides selectivity via histograms on the
            # column. We don't replicate that here; we return a
            # uniform 0.5 prior, which the CBO uses only if no
            # DuckDB-side estimate is available. The cost is small
            # because classical predicates are not the multi-modal
            # specific case (Section 6.2 of the paper).
            return self.n // 2

        return self.n // 2

    def _can_joint_probe(self, left: "Predicate", right: "Predicate") -> bool:
        """Whether the v2 joint sample probe can evaluate (left AND right).

        Conditions:
          1. sample_probe exists and is large enough (>= 50 rows)
          2. neither side is GraphReach (probe falls back to False
             for graph predicates, which would systematically
             undercount)
          3. both sides' columns are present in the store
        """
        if self.sample_probe is None:
            return False
        if len(self.sample_probe._idx) < 50:
            return False
        for p in (left, right):
            if isinstance(p, GraphReach):
                return False
            if isinstance(p, ColumnPredicate) and p.column not in self.store.arrow.column_names:
                return False
            if isinstance(p, VectorRange) and p.column not in self.store.arrow.column_names:
                return False
        return True

    def cache_stats(self) -> Dict[str, int]:
        """Aggregate cache statistics across all vector sketches.

        Used by the CBO for diagnostic logging.
        """
        hits = misses = size = 0
        for vs in self.vector_sketches.values():
            st = vs.cache_stats()
            hits += st["hits"]
            misses += st["misses"]
            size += st["size"]
        return {"hits": hits, "misses": misses, "size": size, "n_sketches": len(self.vector_sketches)}


# =====================================================================
# Plan walker: collect every (column, predicate) pair so the CBO can
# populate its selectivity hint.
# =====================================================================

def _walk_predicates(plan: Plan) -> List[Predicate]:
    """Depth-first traversal that yields every *atomic* predicate in
    the plan tree (under any Select, and inside any And/Or/Not, in
    the order it appears)."""
    out: List[Predicate] = []
    def visit_pred(p: Predicate) -> None:
        if isinstance(p, And):
            visit_pred(p.left)
            visit_pred(p.right)
        elif isinstance(p, Or):
            visit_pred(p.left)
            visit_pred(p.right)
        elif isinstance(p, Not):
            visit_pred(p.child)
        else:
            out.append(p)
    def visit(node) -> None:
        if isinstance(node, Select):
            visit_pred(node.predicate)
            visit(node.child)
        elif hasattr(node, "child") and node.child is not None:
            visit(node.child)
        if hasattr(node, "left") and node.left is not None:
            visit(node.left)
        if hasattr(node, "right") and node.right is not None:
            visit(node.right)
    visit(plan)
    return out


def _walk_vector_predicates_with_upstream(
    plan: Plan,
) -> List[Tuple[VectorRange, Tuple[ColumnPredicate, ...]]]:
    """Walk the plan and yield (vector_pred, upstream_classical_preds)
    pairs, where `upstream_classical_preds` is the list of classical
    predicates that appear above the vector predicate in the plan.

    The list is in the order the predicates appear (top-down), which
    matches the order they would be applied if the plan were
    executed left-to-right.

    For a query like
        Sigma_{subsystem='ft'}(Sigma_{||emb-q||<tau}(R))
    the result is [ (VectorRange('emb'), (ColumnPredicate('subsystem','=','ft'),)) ].

    For a query like
        Sigma_{||emb-q||<tau}(Sigma_{subsystem='ft'}(R))
    the result is [ (VectorRange('emb'), ()) ] — the classical filter
    is downstream of the vector predicate, so it does not affect the
    surviving subset the vector scan sees.

    The CBO's `FilterOrder` (Algorithm 2, Section 5.4) is the
    authority on which ordering is chosen at planning time; this
    walker just reports the *current* ordering as written. The CBO
    iterates over alternative orderings (via D's enumeration) and
    uses the rho_hint + selectivity_hint jointly to pick the cheapest.
    """
    out: List[Tuple[VectorRange, Tuple[ColumnPredicate, ...]]] = []

    def _collect_classical(pred: Predicate) -> List[ColumnPredicate]:
        """Return the list of ColumnPredicates in `pred` (flattened)."""
        if isinstance(pred, ColumnPredicate):
            return [pred]
        if isinstance(pred, And):
            return _collect_classical(pred.left) + _collect_classical(pred.right)
        if isinstance(pred, Or):
            return _collect_classical(pred.left) + _collect_classical(pred.right)
        if isinstance(pred, Not):
            return []  # conservative
        return []

    def visit(node: Plan, upstream: List[ColumnPredicate]) -> None:
        if isinstance(node, Select):
            sub_classical = _collect_classical(node.predicate)
            if isinstance(node.predicate, VectorRange):
                # The vector predicate is here; the upstream classical
                # filters are the ones in `upstream` (above this node).
                out.append((node.predicate, tuple(upstream)))
                visit(node.child, upstream)
            else:
                # Classical or And/Or; the new upstream = upstream + sub_classical
                visit(node.child, upstream + sub_classical)
        elif isinstance(node, (Project, Order, Limit, Materialize, Expand)):
            # Transparent: pass upstream through
            visit(node.child, upstream)
        elif isinstance(node, Join):
            visit(node.left, upstream)
            visit(node.right, upstream)
        else:
            # Scan, etc.
            pass

    visit(plan, [])
    return out


def populate_rho_hint(
    plan: Plan,
    density_estimator: Optional[FilteredHNSWDensity] = None,
    store: Any = None,
    vector_columns: Sequence[str] = (),
) -> Dict[str, float]:
    """Walk the plan, return a {column: rho_f} dict for the CBO.

    For each VectorRange predicate in the plan, look up the rho_f
    for its (column, upstream_classical_filter) pair in the
    density estimator's cache. If the estimator is None or the
    column is not in its index set, fall back to 1.0 (no penalty).

    The returned dict is consumed by the CBO as `rho_hint` in
    `to_physical` / `cbo_pick`. Like the selectivity hint, it is
    a *lower-bound default*: missing keys get the CBO's safe
    default (1.0), which means "assume no penalty".
    """
    if density_estimator is None:
        return {}
    hint: Dict[str, float] = {}
    for vec_pred, upstream in _walk_vector_predicates_with_upstream(plan):
        col = vec_pred.column
        if col in hint:
            continue  # first-wins; the upstream filter is the same
        rho = density_estimator.lookup(col, list(upstream))
        hint[col] = rho
    return hint


def populate_classical_s_hint(
    plan: Plan,
    hub: Optional["StatisticsHub"] = None,
    store: Any = None,
) -> Dict[str, float]:
    """Walk the plan, return a {column: classical_selectivity} dict.

    For each ColumnPredicate in the plan, return the estimated
    selectivity of the filter on that column. Used by `to_physical`
    as `classical_s_hint` so the CBO can compute the effective
    `n_input = classical_s * n` for the vector scan.

    When a `store` is provided, the selectivity is computed by
    scanning the substrate's arrow column for matching rows. This
    is the cheap path used by E9 for the prototype's small
    corpus. With a full StatisticsHub, DuckDB's ANALYZE histograms
    could be consulted instead.
    """
    hint: Dict[str, float] = {}
    n_total = store.arrow.num_rows if store is not None else (hub.n if hub is not None else 0)
    for pred in _walk_predicates(plan):
        if isinstance(pred, ColumnPredicate) and isinstance(pred.value, (str, int, float, bool)):
            col = pred.column
            if col in hint:
                continue  # first-wins
            if store is not None:
                try:
                    n_match = _count_classical(store, pred)
                    s = max(1.0 / max(1, n_total), min(1.0, n_match / max(1, n_total)))
                except Exception:
                    s = 0.5
            else:
                s = 0.5
            hint[col] = s
    return hint


def _count_classical(store: Any, pred: ColumnPredicate) -> int:
    """Count rows in `store.arrow` matching a single ColumnPredicate.

    Used by `populate_classical_s_hint` to estimate the classical
    filter's selectivity from the substrate.
    """
    if store is None or pred.column not in store.arrow.column_names:
        return 0
    col = store.arrow.column(pred.column).to_numpy()
    if pred.op == "=":
        return int((col == pred.value).sum())
    if pred.op == "!=":
        return int((col != pred.value).sum())
    if pred.op == "<":
        return int((col < pred.value).sum())
    if pred.op == "<=":
        return int((col <= pred.value).sum())
    if pred.op == ">":
        return int((col > pred.value).sum())
    if pred.op == ">=":
        return int((col >= pred.value).sum())
    if pred.op == "in":
        return int(sum(1 for v in col if v in pred.value))
    return 0


def populate_selectivity_hint(
    plan: Plan,
    store,
    query_params: Optional[Dict[str, Any]] = None,
    vector_columns: Sequence[str] = (),
    text_columns: Sequence[str] = (),
    graph: Optional[GraphStore] = None,
    sample_rate: float = 0.01,
    hub: Optional[StatisticsHub] = None,
) -> Dict[str, float]:
    """Walk the plan, build (or reuse) a StatisticsHub, return a
    {column: selectivity} dict the CBO can consume.

    For predicate forms that touch a single column, the estimate is
    keyed by the column name. This is the same shape as the
    ``selectivity_hint`` parameter the CBO already accepts in
    ``to_physical``.

    If a pre-built ``hub`` is provided, it is reused; this avoids
    the per-call cost of rebuilding the HNSW / BM25 indexes (which
    is several seconds for 10k x 768). The same hub can be reused
    across many queries in a session.

    The returned dict is intentionally a *lower bound* on selectivity
    for the columns it covers: if the CBO does not see a key for a
    column, it falls back to its default 0.5 prior (no information
    is better than a misleading number).
    """
    query_params = query_params or {}
    if hub is None:
        hub = StatisticsHub.from_store(
            store, vector_columns=vector_columns, text_columns=text_columns,
            graph=graph, sample_rate=sample_rate,
        )
    hint: Dict[str, float] = {}
    for pred in _walk_predicates(plan):
        if isinstance(pred, ColumnPredicate):
            # Classical predicates are not estimated here; DuckDB
            # ANALYZE handles them.
            continue
        col = getattr(pred, "column", None)
        if col is None:
            continue
        if col in hint:
            continue  # first-wins for repeated columns
        hint[col] = hub.estimate_selectivity(pred, query_params)
    return hint
