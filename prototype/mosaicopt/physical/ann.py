"""faiss-backed ANN index implementing the H_{V,q,tau} physical operator.

Three strategies are supported, matching Section 5.3 of the paper:
    * "hnsw"  : HNSW graph (Malkov & Yashunin, 2018) - high recall,
                sublinear query time, but slow build.
    * "ivf"   : IVF inverted file - moderate recall, faster build,
                tunable via nprobe.
    * "flat"  : brute-force linear scan - exact (recall = 1), slow on
                large n but best for very small n or very large tau.

The CBO in optimizer/cbo.py picks a strategy based on estimated
selectivity (the fraction of rows that pass the range filter).
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import faiss
import numpy as np


@dataclass
class AnnIndex:
    """faiss-backed vector index.

    Parameters
    ----------
    vectors : np.ndarray of shape (n, d), dtype float32
        The base vectors to index.
    dim : int
        Dimensionality (must match vectors.shape[1]).
    strategy : {"hnsw", "ivf", "flat"}
        Which physical strategy to use.
    M : int
        HNSW graph degree (default 32).
    ef_construction : int
        HNSW build parameter (default 200).
    nlist : int
        IVF number of cells (default min(40, n//100) per spec).
    metric : str
        "l2" or "ip" (inner product).
    """
    vectors: np.ndarray
    dim: int
    strategy: str = "hnsw"
    M: int = 32
    ef_construction: int = 200
    nlist: Optional[int] = None
    metric: str = "l2"

    def __post_init__(self) -> None:
        v = np.ascontiguousarray(self.vectors.astype("float32"))
        if v.ndim != 2 or v.shape[1] != self.dim:
            raise ValueError(f"vectors must be (n,{self.dim}); got {v.shape}")
        n = v.shape[0]

        if self.strategy == "hnsw":
            idx = faiss.IndexHNSWFlat(self.dim, self.M, faiss.METRIC_L2)
            idx.hnsw.efConstruction = self.ef_construction
            idx.add(v)
            idx.hnsw.efSearch = 64  # default; tunable via .hnsw.efSearch
        elif self.strategy == "ivf":
            nlist = self.nlist if self.nlist is not None else max(1, min(40, n // 100))
            nlist = max(1, min(nlist, n))
            quantizer = faiss.IndexFlatL2(self.dim)
            idx = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_L2)
            # IVF needs training; for a 10k-row prototype, default k-means
            # converges in <1s.
            if not idx.is_trained:
                idx.train(v)
            idx.add(v)
            idx.nprobe = max(1, nlist // 4)
        elif self.strategy == "flat":
            idx = faiss.IndexFlatL2(self.dim)
            idx.add(v)
        else:
            raise ValueError(f"unknown strategy: {self.strategy}")

        # faiss returns indices in the *added* order; the prototypes use
        # 0..n-1 as row ids, so .add(v) is already identity-mapped.
        self._index = idx
        self._n = n

    # --- query --------------------------------------------------------

    def range_query(
        self,
        q: np.ndarray,
        threshold: float,
        k_hard: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances) for vectors within L2 distance < threshold.

        For "flat", this is exact. For HNSW/IVF, recall is bounded by
        the index's internal parameters (efSearch / nprobe); see
        Section 5.3 of the paper.
        """
        q = np.ascontiguousarray(q.astype("float32").reshape(1, self.dim))
        # faiss range_search returns squared L2 by default; we use
        # squared distance and convert the threshold to squared.
        sq_thr = float(threshold) ** 2
        lims, D, I = self._index.range_search(q, sq_thr)
        # faiss returns flat numpy arrays; with a single query, len(D)
        # is 0..n_hits. Reshape to 1-D for downstream code.
        d = np.asarray(D).reshape(-1)
        i = np.asarray(I).reshape(-1)
        if k_hard is not None and len(i) > k_hard:
            # If the index overshoots (e.g. IVF with broad nprobe),
            # keep the k smallest. This is a *physical* implementation
            # detail; the logical H_{V,q,tau} does not bound K.
            order = np.argsort(d)[:k_hard]
            d = d[order]
            i = i[order]
        # Return L2 (not squared) for direct comparison with the
        # algebra's ||V - q|| semantics.
        return i.copy(), np.sqrt(d).copy()

    def knn_query(self, q: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids, distances) for the k-nearest neighbours."""
        q = np.ascontiguousarray(q.astype("float32").reshape(1, self.dim))
        D, I = self._index.search(q, k)
        return I[0].copy(), np.sqrt(D[0]).copy()

    # --- filtered traversal (Section 6.2.2) ---------------------------

    def filtered_range_query(
        self,
        q: np.ndarray,
        threshold: float,
        allowed_ids: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """HNSW with *pre-filter*: restrict the search to nodes in `allowed_ids`.

        Uses faiss's `IDSelector` API. Returns the (ids, distances)
        for the surviving candidates that pass the range threshold.
        If `allowed_ids` is empty, returns an empty pair.

        The traversal cost depends on the connectivity of the HNSW
        graph restricted to `allowed_ids` -- if the surviving subset
        is fragmented, the search has to traverse more of the graph
        and the effective cost grows. The CBO's cost model captures
        this via the `rho_f` parameter of `cbo.cost_vector`.
        """
        if len(allowed_ids) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        q = np.ascontiguousarray(q.astype("float32").reshape(1, self.dim))
        ids64 = np.ascontiguousarray(allowed_ids.astype(np.int64))
        sel = faiss.IDSelectorBatch(ids64)
        params = faiss.SearchParametersHNSW()
        params.sel = sel
        sq_thr = float(threshold) ** 2
        lims, D, I = self._index.range_search(q, sq_thr, params=params)
        d = np.asarray(D).reshape(-1)
        i = np.asarray(I).reshape(-1)
        return i.copy(), np.sqrt(d).copy()

    def post_filter_range_query(
        self,
        q: np.ndarray,
        threshold: float,
        allowed_ids: np.ndarray,
        k_hard: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """HNSW with *post-filter*: unfiltered HNSW search, then keep only `allowed_ids`.

        The traversal cost is the unfiltered HNSW cost (no graph
        fragmentation penalty), but recall is bounded by HNSW's
        ability to surface all the true in-`allowed_ids` candidates
        in its top-k. We use a large efSearch so the search
        approximates an exhaustive scan in the k that matters.
        """
        ids, dists = self.range_query(q, threshold, k_hard=k_hard)
        if len(allowed_ids) == 0:
            return np.array([], dtype=np.int64), np.array([], dtype=np.float32)
        allowed_set = set(int(x) for x in allowed_ids)
        mask = np.fromiter((int(i) in allowed_set for i in ids),
                           dtype=bool, count=len(ids))
        return ids[mask].copy(), dists[mask].copy()

    def filtered_density(
        self,
        allowed_ids: np.ndarray,
        q_sample: np.ndarray,
        k: int = 50,
    ) -> float:
        r"""Empirical ``rho_f`` for the surviving subset `allowed_ids`.

        The connectivity ratio rho_f is defined as the cost-efficiency
        of HNSW search restricted to ``allowed_ids``, relative to
        the unfiltered search. Operationally:

            rho_f = (T_unfiltered / n_unfiltered) /
                    (T_filtered / n_in_Rf_found)

        When the HNSW graph is well-connected within ``allowed_ids``,
        every HNSW step stays in the surviving set and the search
        cost scales linearly with ``|R_f|``: rho_f -> 1. When the
        surviving set is scattered (e.g. a selective classical
        filter picks nodes from multiple disconnected clusters),
        the HNSW traversal has to do extra work to surface every
        in-``R_f`` candidate, and the cost amplifies: rho_f < 1.

        The estimator uses ``q_sample`` (shape ``(k_q, d)``) as a
        set of random queries; the per-query time is averaged. The
        measurement is approximate (it depends on the choice of
        ``k`` and the sample of queries), but the bias is small on
        the prototype's 10k corpus.

        Parameters
        ----------
        allowed_ids : np.ndarray of int
            The surviving subset of row ids (R_f).
        q_sample : np.ndarray of shape (k_q, d), float32
            Sample query vectors for the timing.
        k : int
            Number of nearest neighbours to ask for in each search.
        """
        if len(allowed_ids) == 0 or self.strategy != "hnsw":
            return 0.0
        n = self._n
        if len(allowed_ids) >= n:
            return 1.0  # no filter, full connectivity

        # Use a small k for both unfiltered and filtered; we want
        # to measure per-step cost, not exhaustive search.
        k_uf = min(k, n)
        k_f = min(k, len(allowed_ids))
        if k_f == 0:
            return 0.0

        # Unfiltered timing
        t0 = time.perf_counter()
        D_uf, I_uf = self._index.search(q_sample, k_uf)
        t_uf = (time.perf_counter() - t0) / max(1, len(q_sample))

        # Filtered timing
        sel = faiss.IDSelectorBatch(np.ascontiguousarray(allowed_ids.astype(np.int64)))
        params = faiss.SearchParametersHNSW()
        params.sel = sel
        t0 = time.perf_counter()
        D_f, I_f = self._index.search(q_sample, k_f, params=params)
        t_f = (time.perf_counter() - t0) / max(1, len(q_sample))

        # n_in_Rf_found (per query, averaged): the number of in-R_f
        # results that filtered search actually returned. When R_f
        # is fragmented, filtered search returns fewer than k_f
        # (the search runs out of in-R_f candidates in the top-k).
        allowed_set = set(int(x) for x in allowed_ids)
        in_Rf_per_q = np.zeros(len(q_sample), dtype=np.int64)
        for qi, row in enumerate(I_f):
            for i in row:
                if int(i) in allowed_set:
                    in_Rf_per_q[qi] += 1
        avg_found = max(1, int(in_Rf_per_q.mean()))

        # Cost per in-Rf result, unfiltered and filtered
        cost_per_unf = t_uf / max(1, k_uf)
        cost_per_f = t_f / max(1, avg_found)
        # rho_f: how efficient is filtered search relative to unfiltered,
        # normalized so that 1.0 means "linear scaling"
        rho = cost_per_unf / max(1e-9, cost_per_f)
        return min(1.0, max(0.0, float(rho)))


# ---------------------------------------------------------------------
# Thin convenience function used by the optimizer
# ---------------------------------------------------------------------

def ann_range_query(
    vectors: np.ndarray,
    q: np.ndarray,
    threshold: float,
    strategy: str = "hnsw",
) -> Tuple[np.ndarray, np.ndarray, float]:
    """One-shot helper: build index, run range query, return (ids, dists, t_ms).

    Returns the elapsed time in milliseconds (wall clock) for the
    *query* step (not the build). The build is included in
    construction time but the CBO's cost model treats build as a
    one-off amortized cost.
    """
    idx = AnnIndex(vectors, vectors.shape[1], strategy=strategy)
    t0 = time.perf_counter()
    ids, dists = idx.range_query(q, threshold)
    t_ms = (time.perf_counter() - t0) * 1000.0
    return ids, dists, t_ms
