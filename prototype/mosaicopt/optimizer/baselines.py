"""Baseline systems for the multi-modal query comparison.

This module provides two simplified Python implementations of
multi-modal query systems that we compare against MosaicOpt in
the §11 evaluation:

* :class:`NaviXStyle` --- a single-CBO unified engine that picks
  the best per-modality physical operator independently
  (vector HNSW / text BM25 / graph BFS) and intersects the
  results.  This approximates the design of
  NaviX / kuzu (VLDB 2025), which also uses a single CBO over
  vector + text + graph modalities.  NaviX supports a
  prefilter / postfilter flag per modality; for the fair
  comparison we always prefilter on each modality's primary
  index and intersect.

* :class:`ACORNStyle` --- a vector-first pipeline that runs
  HNSW with a pre-filter (the ACORN-σ with σ=1.0 simplification
  via faiss's ``IDSelector``), then post-filters the surviving
  set with BM25 for text and a BFS reachability check for
  graph.  This is the natural adaptation of ACORN
  (arXiv 2022), which natively supports attribute filters but
  not text or graph; we add the post-filters as the minimal
  extension.  Where ACORN cannot natively support a modality
  (graph, fuzzy-text), we fall back to a post-filter pass.

Both systems take a :class:`mosaicopt.algebra.ops.Plan` and a
:class:`mosaicopt.physical.store.TableStore`, run the query
end-to-end, and return a :class:`BaselineResult` with the wall-
clock latency, the recall against the exact result, and the
number of candidates produced at each stage.

The baselines intentionally *do not* use MosaicOpt's
RBO, CBO, or statistics layer.  The comparison measures the
end-to-end latency of three different optimizer stacks on the
same logical plan.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

from ..algebra.ops import (
    Plan, Scan, Project, Select, Materialize, Order, Limit,
    Expand, Join,
)
from ..algebra.predicates import (
    Predicate, ColumnPredicate, VectorRange, BM25Score,
    GraphReach, And, Or, Not,
)
from ..physical.ann import AnnIndex
from ..physical.bm25 import BM25Index, _tokenize
from ..physical.graph import GraphStore
from ..physical.store import TableStore
from .index_manager import IndexManager


# ---------------------------------------------------------------------
# Common result type
# ---------------------------------------------------------------------

@dataclass
class BaselineResult:
    """End-to-end result from a baseline system."""
    system: str                       # "mosaicopt" / "navix" / "acorn"
    query_id: str
    n_input: int                      # rows in the input relation
    n_predicate_selectivity: float    # |true result| / n_input
    n_candidate: int                  # rows produced at the end
    wall_clock_ms: float              # end-to-end latency
    recall: float                     # recall vs exact ground truth
    stages: List[Dict[str, Any]] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------
# Helper: walk the plan and collect predicates by modality
# ---------------------------------------------------------------------

def _collect_predicates(plan: Plan) -> Tuple[
    List[ColumnPredicate], List[VectorRange],
    List[BM25Score], List[GraphReach]
]:
    """Walk the plan tree, return (classical, vector, text, graph) lists.

    ColumnPredicates with op='contains' are split out into a separate
    fuzzy list (returned as part of `classical` with a marker).  Use
    :func:`_collect_predicates_v2` for the v2 split.
    """
    classical, vectors, texts, graphs, _fuzzy = _collect_predicates_v2(plan)
    return classical, vectors, texts, graphs


def _collect_predicates_v2(plan: Plan) -> Tuple[
    List[ColumnPredicate], List[VectorRange],
    List[BM25Score], List[GraphReach], List[ColumnPredicate]
]:
    """Walk the plan tree, return (classical, vector, text, graph, fuzzy) lists.

    ColumnPredicates with op='contains' are routed to the fuzzy list;
    other ColumnPredicates go to the classical list.
    """
    classical: List[ColumnPredicate] = []
    vectors: List[VectorRange] = []
    texts: List[BM25Score] = []
    graphs: List[GraphReach] = []
    fuzzy: List[ColumnPredicate] = []
    def visit_pred(p: Predicate) -> None:
        if isinstance(p, ColumnPredicate):
            if p.op == "contains":
                fuzzy.append(p)
            else:
                classical.append(p)
        elif isinstance(p, VectorRange):
            vectors.append(p)
        elif isinstance(p, BM25Score):
            texts.append(p)
        elif isinstance(p, GraphReach):
            graphs.append(p)
        elif isinstance(p, (And, Or)):
            visit_pred(p.left)
            visit_pred(p.right)
        elif isinstance(p, Not):
            visit_pred(p.child)
    def visit(node: Plan) -> None:
        if isinstance(node, Select):
            visit_pred(node.predicate)
            visit(node.child)
        elif isinstance(node, (Project, Materialize, Order, Limit)):
            visit(node.child)
        elif isinstance(node, Expand):
            visit(node.child)
        elif isinstance(node, Join):
            visit(node.left)
            visit(node.right)
        else:
            pass
    visit(plan)
    return classical, vectors, texts, graphs, fuzzy


def _apply_classical_filter(
    table: pa.Table,
    preds: List[ColumnPredicate],
    col_cache: Optional[Dict[Tuple[int, str], np.ndarray]] = None,
) -> np.ndarray:
    """Apply classical predicates to a table, return surviving row indices.

    ``col_cache`` is an optional per-system dict that maps
    ``(id(table), column_name)`` to a materialized numpy column.
    ``table.column(...).to_numpy()`` on a 1M-row string column costs
    ~70 ms; sharing the materialized array across all queries
    in a sweep removes that per-query overhead.
    """
    n = table.num_rows
    if not preds:
        return np.arange(n, dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    for p in preds:
        if p.column not in table.column_names:
            continue
        cache_key = (id(table), p.column)
        if col_cache is not None and cache_key in col_cache:
            col = col_cache[cache_key]
        else:
            col = table.column(p.column).to_numpy()
            if col_cache is not None:
                col_cache[cache_key] = col
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
            # Vectorized ``in`` (was a Python generator — slow at n=1M).
            mask &= np.isin(col, np.asarray(p.value))
    return np.flatnonzero(mask).astype(np.int64)


def _apply_classical_filter_on(
    table: pa.Table,
    surviving_ids: np.ndarray,
    preds: List[ColumnPredicate],
) -> np.ndarray:
    """Apply classical predicates restricted to a surviving set.

    Returns the surviving row indices after the predicates.
    """
    if not preds or len(surviving_ids) == 0:
        return surviving_ids
    keep = np.ones(len(surviving_ids), dtype=bool)
    for p in preds:
        if p.column not in table.column_names:
            continue
        col = table.column(p.column)
        for i, rid in enumerate(surviving_ids):
            v = col[int(rid)].as_py()
            if p.op == "=":
                if str(v) != str(p.value):
                    keep[i] = False
            elif p.op == "!=":
                if str(v) == str(p.value):
                    keep[i] = False
            elif p.op == "<":
                if not (v < p.value):
                    keep[i] = False
            elif p.op == "<=":
                if not (v <= p.value):
                    keep[i] = False
            elif p.op == ">":
                if not (v > p.value):
                    keep[i] = False
            elif p.op == ">=":
                if not (v >= p.value):
                    keep[i] = False
    return surviving_ids[keep]


def _apply_fuzzy_filter(
    table: pa.Table,
    surviving_ids: np.ndarray,
    fuzzy_preds: List[ColumnPredicate],
) -> np.ndarray:
    """Apply LIKE-style fuzzy text match: ColumnPredicate(col, 'contains', term).

    Returns the surviving row indices after the fuzzy predicate.
    """
    if not fuzzy_preds or len(surviving_ids) == 0:
        return surviving_ids
    keep = np.ones(len(surviving_ids), dtype=bool)
    for p in fuzzy_preds:
        if p.column not in table.column_names:
            continue
        term = str(p.value).lower()
        col_data = table.column(p.column)
        for i, rid in enumerate(surviving_ids):
            text = str(col_data[int(rid)].as_py()).lower()
            if term not in text:
                keep[i] = False
    return surviving_ids[keep]


def _apply_bm25_filter(
    table: pa.Table,
    surviving_ids: np.ndarray,
    text_preds: List[BM25Score],
    query_params: Dict[str, Any],
    bm25_cache: Optional[Dict[str, Tuple[List[str], Any]]] = None,
) -> np.ndarray:
    """Post-filter: keep rows whose BM25 score exceeds threshold.

    Uses a column-level BM25 index that is cached at the
    system level so successive queries don't pay the
    ~hundreds-of-ms build cost. The cache stores, per text
    column, the full list of document strings plus the
    built ``rank_bm25`` index; per-query we just call
    ``get_scores`` and look up the surviving_ids positions.

    Falls back to the legacy per-query BM25 build when no
    cache is provided (single-shot callers).
    """
    if not text_preds or len(surviving_ids) == 0:
        return surviving_ids
    text_col = text_preds[0].column
    keep = np.ones(len(surviving_ids), dtype=bool)
    for tp in text_preds:
        q_text = query_params.get(tp.q_param, "")
        if not q_text:
            continue
        scores: np.ndarray
        if bm25_cache is not None and text_col in bm25_cache:
            docs, idx = bm25_cache[text_col]
            all_scores = np.asarray(
                idx.get_scores(_tokenize(str(q_text))),
                dtype=np.float32,
            )
            all_scores = np.maximum(all_scores, 0.0)
            # Subset to surviving_ids positions
            sub_idx = np.asarray(surviving_ids, dtype=np.int64)
            # Clamp to avoid out-of-range when datasets expose
            # id gaps
            sub_idx = np.minimum(sub_idx, len(all_scores) - 1)
            scores = all_scores[sub_idx]
        else:
            # Slow path: build a small BM25 index over the
            # surviving subset only (used by tests / single-call
            # callers).
            docs = [str(table.column(text_col)[i].as_py())
                    for i in surviving_ids]
            bm25 = BM25Index(docs)
            scores = np.asarray(
                bm25._bm25.get_scores(_tokenize(str(q_text))),
                dtype=np.float32,
            )
            scores = np.maximum(scores, 0.0)
        keep &= (scores > float(tp.threshold))
    return surviving_ids[keep]


def build_bm25_cache(table: pa.Table, text_columns: List[str]
                     ) -> Dict[str, Tuple[List[str], Any]]:
    """Build a per-text-column BM25 cache for the *whole* table.

    Pay once (~a few seconds at n = 500k for the largest text
    columns) to make every subsequent ``_apply_bm25_filter`` call
    O(|surviving_ids|) instead of O(|surviving_ids| * doc_len).
    """
    cache: Dict[str, Tuple[List[str], Any]] = {}
    for col in text_columns:
        try:
            arr = table.column(col)
        except KeyError:
            continue
        n = len(arr)
        # Pull the docs as Python strings; this is the dominating
        # one-time cost and is amortized across all 64 sweep queries.
        docs = [str(arr[i].as_py()) for i in range(n)]
        bm25 = BM25Index(docs)
        cache[col] = (docs, bm25._bm25)
    return cache


def _apply_graph_filter(
    table: pa.Table,
    surviving_ids: np.ndarray,
    graph_preds: List[GraphReach],
    graph_store: Optional[GraphStore],
    query_params: Dict[str, Any],
) -> np.ndarray:
    """Post-filter: keep rows whose graph_node reaches the query seed."""
    if not graph_preds or len(surviving_ids) == 0 or graph_store is None:
        return surviving_ids
    keep = np.ones(len(surviving_ids), dtype=bool)
    for gp in graph_preds:
        seed_param = query_params.get(gp.q_param, 0)
        if not isinstance(seed_param, (list, tuple)):
            seeds = [int(seed_param)]
        else:
            seeds = [int(s) for s in seed_param]
        node_col = np.array([table.column(gp.column)[i].as_py()
                             for i in surviving_ids], dtype=np.int64)
        in_reach = np.zeros(len(surviving_ids), dtype=bool)
        for s in seeds:
            reachable = graph_store.reachable(gp.rule_name, s, gp.depth)
            in_reach |= np.isin(node_col, reachable)
        keep &= in_reach
    return surviving_ids[keep]


def _apply_vector_filter(
    vectors: np.ndarray,
    surviving_ids: np.ndarray,
    vector_preds: List[VectorRange],
    hnsw_index: AnnIndex,
    query_params: Dict[str, Any],
    use_pre_filter: bool,
) -> np.ndarray:
    """Apply vector range predicates.  If ``use_pre_filter`` is True,
    use HNSW with faiss IDSelector; otherwise post-filter."""
    if not vector_preds or len(surviving_ids) == 0:
        return surviving_ids
    keep = np.ones(len(surviving_ids), dtype=bool)
    for vp in vector_preds:
        q = query_params.get(vp.q_param)
        if q is None:
            continue
        q_arr = np.asarray(q, dtype="float32")
        if use_pre_filter:
            ids, _ = hnsw_index.filtered_range_query(
                q_arr, float(vp.threshold), surviving_ids
            )
        else:
            ids, _ = hnsw_index.range_query(q_arr, float(vp.threshold))
            ids_set = set(int(x) for x in ids)
            keep &= np.fromiter(
                (int(surviving_ids[i]) in ids_set
                 for i in range(len(surviving_ids))),
                dtype=bool, count=len(surviving_ids),
            )
            continue
        ids_set = set(int(x) for x in ids)
        keep &= np.fromiter(
            (int(surviving_ids[i]) in ids_set
             for i in range(len(surviving_ids))),
            dtype=bool, count=len(surviving_ids),
        )
    return surviving_ids[keep]


# ---------------------------------------------------------------------
# MosaicOpt (the proposed system, end-to-end via to_physical)
# ---------------------------------------------------------------------

class MosaicOptSystem:
    """End-to-end MosaicOpt: RBO + CBO + statistics + adaptive policy.

    Uses the integrated :func:`optimizer.plan.to_physical`
    mapping, which applies the RBO, the cross-modal plan
    enumeration (D), the CBO with statistics, and the
    $\\rho_f$-aware cost model.
    """
    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None,
                 workload_size: int = 16) -> None:
        self.store = store
        self.graph = graph
        # Lazy cache for built HNSW indexes; the build-vs-skip
        # decision is delegated to the IndexManager (instead of the
        # previous ``N_SKIP_HNSW_BUILD = 50_000`` hardcoded override in
        # :mod:`cbo`). ``workload_size`` is the expected number of
        # queries that will run against this stack; the sweep driver
        # sets it explicitly. Default 16 ≈ small benchmark.
        self.hnsw_cache: Dict[str, Any] = {}
        self.index_mgr = IndexManager(workload_size=workload_size)
        # Cache BM25 indexes per text column; built lazily on
        # first text query, reused for the rest of the sweep.
        # Pays the one-time build cost in exchange for not
        # rebuilding over the surviving subset per query (the
        # latter was the dominant per-query cost at n = 500k).
        self.bm25_cache: Optional[Dict[str, Tuple[List[str], Any]]] = None
        # Per-system cache for the Arrow->numpy column conversion.
        # ``brute_force_filtered_range`` is called once per vector
        # predicate per query (>= 1 per query); without the cache
        # the conversion (~100 ms at n = 500k) is paid per call.
        self.vec_cache: Dict[str, np.ndarray] = {}
        # Per-system cache for the Arrow->numpy classical-column
        # conversion. ``_apply_classical_filter`` materializes the
        # predicate column once per call; for a string column at
        # n=1M the materialization is ~70 ms. The cache reuses
        # the materialized numpy array across all queries that
        # share the same ``table`` object.
        self.classical_cache: Dict[Tuple[int, str], np.ndarray] = {}

    def _maybe_ensure_hnsw(self, plan) -> None:
        """DEPRECATED per-query helper — kept as a thin alias for
        backwards compatibility. The supported path is
        ``self.index_mgr.pre_build([(col, vec, dim), ...])`` invoked
        once from the workload driver (sweep setup / app startup).
        This method should not be called from a per-query hot path.
        """
        from .baselines import _collect_predicates_v2 as _c
        try:
            _, vectors, _, _, _ = _c(plan)
        except Exception:
            vectors = []
        cols = []
        for vp in vectors:
            if vp.column not in self.store.arrow.column_names:
                continue
            vec = self.store.vector_column(vp.column)
            cols.append((vp.column, vec, vec.shape[1]))
        if cols:
            self.index_mgr.pre_build(cols)

    def _get_hnsw(self, vec_col, column_name):
        """Back-compat thin wrapper around IndexManager.

        Returns the cached or freshly-built HNSW index for the
        column, or ``None`` if the IndexManager decided the workload
        is too small to justify the build (the caller should then
        fall back to a brute-force exact scan). The efSearch used is
        now picked by the IndexManager (``choose_hnsw_params``),
        not a hard-coded 512.

        Use ``self.index_mgr.ensure_hnsw`` directly in new code; this
        thin wrapper is retained for the PostFilterStyle call site
        that still uses ``self._get_hnsw``.
        """
        has_hnsw, hnsw = self.index_mgr.ensure_hnsw(
            column_name, vec_col, dim=vec_col.shape[1],
        )
        return hnsw

    def run(
        self,
        plan: Plan,
        query_params: Optional[Dict[str, Any]] = None,
        query_id: str = "q",
        ground_truth_ids: Optional[np.ndarray] = None,
    ) -> BaselineResult:
        from .cbo import cbo_pick
        query_params = query_params or {}
        # Walk the plan tree ONCE; previous revisions called this twice
        # (once for classical+fuzzy, once for vectors/texts/graphs).
        # Cost at n=1M was ~0.04 ms total — small but worth removing
        # the duplicate work + the implied "we don't know what we
        # need" code smell.
        classical, vectors, texts, graphs, fuzzy = _collect_predicates_v2(plan)
        # Historical note: prior revisions built a fresh StatisticsHub +
        # FilteredHNSWDensity + populate_*_hint dicts at the start of
        # every ``run`` call, then ran ``to_physical`` for a "physical
        # plan" that was never actually executed — the cascade below
        # is the real executor. Profiling at n=200k showed the dead
        # block cost ~9 ms / query; at n=1M it cost ~50-100 ms
        # because ``populate_classical_s_hint`` re-counts classical
        # rows the executor was about to filter anyway. The exact cs
        # is now computed from ``surviving_ids.size / n_total`` after
        # the cascade, which is both cheaper and more accurate.
        table = self.store.arrow
        n_total = table.num_rows
        stages = []
        t0 = time.perf_counter()
        # 1. Classical pre-filter. Defer the ``np.arange(n_total)``
        #    allocation until we know whether we actually need it:
        #    if classical is non-empty, ``_apply_classical_filter``
        #    returns a fresh int64 array of the surviving indices
        #    and the arange would be immediately discarded.
        surviving_ids: np.ndarray
        if classical:
            t1 = time.perf_counter()
            surviving_ids = _apply_classical_filter(
                table, classical, col_cache=self.classical_cache,
            )
            t_classical = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": "classical", "kept": len(surviving_ids),
                           "ms": t_classical})
        else:
            surviving_ids = np.arange(n_total, dtype=np.int64)
        # 1b. Fuzzy text pre-filter (cheap; reduces the search space)
        if fuzzy and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            surviving_ids = _apply_fuzzy_filter(table, surviving_ids, fuzzy)
            t_fz = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"fuzzy({len(fuzzy)} preds)",
                           "kept": len(surviving_ids), "ms": t_fz})
        # 2. Vector (CBO with ρ + classical_s)
        if vectors and len(surviving_ids) > 0:
            # HNSW / IVF indexes are **pre-built** by the workload
            # infrastructure (sweep driver / app startup) which calls
            # ``self.index_mgr.pre_build(plan)`` once before any queries
            # run. The per-query path is read-only w.r.t. indexes: the
            # CBO sees only ``available_indexes(column)`` and routes
            # accordingly without triggering builds.
            for vp in vectors:
                q = query_params.get(vp.q_param)
                if q is None:
                    continue
                q_arr = np.asarray(q, dtype="float32")
                available = self.index_mgr.available_indexes(vp.column)
                # s_v_estimate is only worth its ~125ms cost at large
                # n when CBO has >= 2 indexed strategies to compare.
                # When no HNSW/IVF exist, the CBO has only ``flat`` as
                # a non-trivial candidate, and the cost is driven by
                # ``classical_s`` alone — a selectivity estimate adds
                # nothing. Skip it to save ~125ms/query.
                if available:
                    s = float(s_v_estimate(
                        self.store, surviving_ids, vp, q_arr, hnsw=None,
                        vec_cache=self.vec_cache,
                    ))
                else:
                    s = 1.0  # CBO sees candidates=["flat"]; cs drives the rest.
                # rho = HNSW connectivity ratio of the surviving subset
                # (Section 6.2.2). No per-query estimator is wired into
                # MosaicOptSystem.run; the legacy ``populate_rho_hint``
                # path was removed together with the dead to_physical
                # block. The CBO's DEFAULT_RHO=1.0 (no fragmentation
                # penalty) is the right default here — the surviving
                # subset's fragmentation cost is bounded by classical_s
                # which already scales n_input. If a per-query rho
                # estimator is added later, replace this with the dict
                # lookup from that estimator.
                rho = 1.0
                cs = 1.0
                if surviving_ids.size > 0:
                    cs = len(surviving_ids) / max(1, n_total)
                strategy = cbo_pick(s, n_total, 128, indexes_available=available,
                                    rho=rho, classical_s=cs)
                # Execute the strategy
                if strategy == "hnsw":
                    hnsw = self.index_mgr.get_hnsw(vp.column)
                    if hnsw is None:
                        # Defensive fallback. Should not happen because
                        # ``available`` was derived from the cache, but
                        # be safe.
                        strategy = "flat"
                        new_ids = brute_force_filtered_range(
                            self.store, vp.column, q_arr,
                            float(vp.threshold), surviving_ids,
                            vec_cache=self.vec_cache,
                        )
                    else:
                        new_ids, _ = hnsw.filtered_range_query(
                            q_arr, float(vp.threshold), surviving_ids
                        )
                elif strategy == "ivf":
                    # Same architecture: IVF comes from the cache.
                    # Building IVF on demand (~900 ms at n=1M) was the
                    # bug; now it only runs once per workload.
                    ivf = self.index_mgr.get_ivf(vp.column)
                    if ivf is None:
                        strategy = "flat"
                        new_ids = brute_force_filtered_range(
                            self.store, vp.column, q_arr,
                            float(vp.threshold), surviving_ids,
                            vec_cache=self.vec_cache,
                        )
                    else:
                        new_ids, _ = ivf.range_query(q_arr, float(vp.threshold))
                else:
                    # flat: brute force over the surviving subset (no
                    # ANN build; recall = 1.0 by construction).
                    new_ids = brute_force_filtered_range(
                        self.store, vp.column, q_arr,
                        float(vp.threshold), surviving_ids,
                        vec_cache=self.vec_cache,
                    )
                # Vectorized intersection. The previous Python loop
                # ``[i for i in surviving_ids if int(i) in new_set]``
                # was O(|surviving_ids|) pure-Python overhead — ~50 ms
                # at n=1M with a tight vector filter. ``np.isin`` does
                # the same work in C with sorted-set semantics; for
                # 1M survivors vs ~2k new_ids it runs in ~5 ms.
                if new_ids.size == 0:
                    surviving_ids = np.empty(0, dtype=np.int64)
                else:
                    mask = np.isin(surviving_ids, new_ids,
                                   assume_unique=True)
                    surviving_ids = surviving_ids[mask]
            t_vec = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"vector({len(vectors)} preds)",
                           "kept": len(surviving_ids), "ms": t_vec})
        # 3. Text (post-filter)
        if texts and len(surviving_ids) > 0:
            if self.bm25_cache is None:
                self.bm25_cache = build_bm25_cache(
                    self.store.arrow,
                    [tp.column for tp in texts
                     if tp.column in self.store.arrow.column_names],
                )
            t1 = time.perf_counter()
            surviving_ids = _apply_bm25_filter(
                self.store.arrow, surviving_ids, texts, query_params,
                bm25_cache=self.bm25_cache,
            )
            t_txt = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"text({len(texts)} preds)",
                           "kept": len(surviving_ids), "ms": t_txt})
        # 4. Graph (post-filter)
        if graphs and len(surviving_ids) > 0 and self.graph is not None:
            t1 = time.perf_counter()
            surviving_ids = _apply_graph_filter(
                self.store.arrow, surviving_ids, graphs, self.graph,
                query_params
            )
            t_gr = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"graph({len(graphs)} preds)",
                           "kept": len(surviving_ids), "ms": t_gr})
        t_total = (time.perf_counter() - t0) * 1000.0
        # Compute recall
        recall = 0.0
        if ground_truth_ids is not None and len(ground_truth_ids) > 0:
            got = set(int(x) for x in surviving_ids)
            ref = set(int(x) for x in ground_truth_ids)
            recall = len(got & ref) / max(1, len(ref))
        sel = (len(ground_truth_ids) / max(1, n_total)) if ground_truth_ids is not None else 0.0
        return BaselineResult(
            system="mosaicopt", query_id=query_id,
            n_input=n_total, n_predicate_selectivity=sel,
            n_candidate=len(surviving_ids), wall_clock_ms=t_total,
            recall=recall, stages=stages,
        )


def _vector_column_cache(
    store: TableStore, column: str,
    cache: Dict[str, np.ndarray],
) -> np.ndarray:
    """Return the (n, d) float32 numpy view of ``store.column``.

    Caches the conversion (Arrow list-of-float32 -> contiguous
    2-D float32 array) per (store id, column name); the
    conversion itself costs 100+ ms at n = 500k and dominates
    ``brute_force_filtered_range`` if redone per call. The
    cache lifetime is the calling system's lifetime, so it
    is the caller's responsibility to drop the cache dict
    when the table is replaced.
    """
    key = (id(store), column)
    arr = cache.get(key)
    if arr is None:
        arr = store.vector_column(column)
        if not arr.flags["C_CONTIGUOUS"]:
            arr = np.ascontiguousarray(arr, dtype="float32")
        else:
            arr = arr.astype("float32", copy=False)
        cache[key] = arr
    return arr


def brute_force_filtered_range(
    store: TableStore, column: str, q_arr: np.ndarray,
    threshold: float, allowed_ids: np.ndarray,
    vec_cache: Optional[Dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """Exact L2 range query restricted to ``allowed_ids``.

    Drop-in for ``AnnIndex.filtered_range_query`` (which, after
    the faiss-IDSelector fix in ``physical/ann.py``, *is* brute
    force anyway). This free-function avoids instantiating any
    HNSW index, which is essential at n = 500k+ where HNSW
    build alone is 200+ s. Cost is O(|allowed_ids| * d) numpy
    ops; at n = 500k with no classical pre-filter this is
    ~30 ms; with a 5% classical pre-filter it is ~1.5 ms.

    Pass ``vec_cache`` to amortize the Arrow->numpy column
    conversion (~100 ms at n = 500k) across many calls.
    """
    if len(allowed_ids) == 0:
        return np.array([], dtype=np.int64)
    q_f = np.ascontiguousarray(q_arr.astype("float32").reshape(1, -1))
    ids64 = np.ascontiguousarray(allowed_ids.astype(np.int64))
    if vec_cache is not None:
        full = _vector_column_cache(store, column, vec_cache)
    else:
        full = store.vector_column(column)
    sub = full[ids64]
    diff = sub - q_f
    d2 = (diff * diff).sum(axis=1)
    sq_thr = float(threshold) ** 2
    mask = d2 < sq_thr
    return ids64[mask]


def s_v_estimate(
    store: TableStore, surviving_ids: np.ndarray,
    vp: VectorRange, q_arr: np.ndarray,
    hnsw: Optional["AnnIndex"] = None,
    vec_cache: Optional[Dict[str, np.ndarray]] = None,
) -> float:
    """Estimate the vector predicate selectivity against the
    *current* surviving set.

    Used by MosaicOptSystem to feed the CBO.

    Performance path:
      - If ``hnsw`` is provided (recommended), use the
        ``filtered_range_query`` interface (which itself routes
        through the AnnIndex's cached vectors for an exact
        brute-force scan over ``allowed_ids``).
      - Otherwise fall back to a free-function brute force,
        which avoids instantiating an AnnIndex when the CBO
        is going to choose ``flat`` or ``ivf`` anyway.
    """
    if len(surviving_ids) == 0:
        return 0.0
    if hnsw is not None:
        ids, _ = hnsw.filtered_range_query(
            q_arr, float(vp.threshold), surviving_ids
        )
        return len(ids) / max(1, len(surviving_ids))
    # No HNSW provided: cheap, no-allocation brute force over
    # the surviving subset (single contiguous numpy op).
    ids = brute_force_filtered_range(
        store, vp.column, q_arr, float(vp.threshold), surviving_ids,
        vec_cache=vec_cache,
    )
    return len(ids) / max(1, len(surviving_ids))


# ---------------------------------------------------------------------
# NaviX-style baseline
# ---------------------------------------------------------------------

class NaviXStyle:
    """Single-CBO unified engine (NaviX / kuzu-style approximation).

    Picks the best per-modality physical operator independently:
    HNSW for vector (pre-filtered), BM25 for text, BFS for
    graph.  Intersects the surviving sets.  No cross-modal
    cost model; no ρ-aware amplifier; no adaptive re-planning.

    This is a fair approximation of the NaviX paper's design
    (VLDB 2025): a single CBO selects each modality's primary
    index and runs them in parallel, intersecting the
    candidates.
    """
    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None,
                 workload_size: int = 16) -> None:
        self.store = store
        self.graph = graph
        # Lazy cache for built HNSW indexes; the build-vs-skip
        # decision is delegated to the IndexManager (instead of the
        # previous ``N_SKIP_HNSW_BUILD = 50_000`` hardcoded override in
        # :mod:`cbo`). ``workload_size`` is the expected number of
        # queries that will run against this stack; the sweep driver
        # sets it explicitly. Default 16 ≈ small benchmark.
        self.hnsw_cache: Dict[str, Any] = {}
        self.index_mgr = IndexManager(workload_size=workload_size)
        # Cache BM25 indexes per text column; built lazily on
        # first text query, reused for the rest of the sweep.
        # Pays the one-time build cost in exchange for not
        # rebuilding over the surviving subset per query (the
        # latter was the dominant per-query cost at n = 500k).
        self.bm25_cache: Optional[Dict[str, Tuple[List[str], Any]]] = None
        # Per-system cache for the Arrow->numpy column conversion.
        # ``brute_force_filtered_range`` is called once per vector
        # predicate per query (>= 1 per query); without the cache
        # the conversion (~100 ms at n = 500k) is paid per call.
        self.vec_cache: Dict[str, np.ndarray] = {}
        # Per-system cache for the Arrow->numpy classical-column
        # conversion (avoids re-materializing the same string
        # column on every query — ~70 ms at n=1M).
        self.classical_cache: Dict[Tuple[int, str], np.ndarray] = {}

    def _maybe_ensure_hnsw(self, plan) -> None:
        """DEPRECATED per-query helper — kept as a thin alias for
        backwards compatibility. The supported path is
        ``self.index_mgr.pre_build([(col, vec, dim), ...])`` invoked
        once from the workload driver (sweep setup / app startup).
        This method should not be called from a per-query hot path.
        """
        from .baselines import _collect_predicates_v2 as _c
        try:
            _, vectors, _, _, _ = _c(plan)
        except Exception:
            vectors = []
        cols = []
        for vp in vectors:
            if vp.column not in self.store.arrow.column_names:
                continue
            vec = self.store.vector_column(vp.column)
            cols.append((vp.column, vec, vec.shape[1]))
        if cols:
            self.index_mgr.pre_build(cols)

    def _get_hnsw(self, vec_col, column_name):
        """Back-compat thin wrapper around IndexManager.

        Returns the cached or freshly-built HNSW index for the
        column, or ``None`` if the IndexManager decided the workload
        is too small to justify the build (the caller should then
        fall back to a brute-force exact scan). The efSearch used is
        now picked by the IndexManager (``choose_hnsw_params``),
        not a hard-coded 512.

        Use ``self.index_mgr.ensure_hnsw`` directly in new code; this
        thin wrapper is retained for the PostFilterStyle call site
        that still uses ``self._get_hnsw``.
        """
        has_hnsw, hnsw = self.index_mgr.ensure_hnsw(
            column_name, vec_col, dim=vec_col.shape[1],
        )
        return hnsw

    def run(
        self,
        plan: Plan,
        query_params: Optional[Dict[str, Any]] = None,
        query_id: str = "q",
        ground_truth_ids: Optional[np.ndarray] = None,
    ) -> BaselineResult:
        query_params = query_params or {}
        table = self.store.arrow
        n_total = table.num_rows
        classical, vectors, texts, graphs, fuzzy = _collect_predicates_v2(plan)
        stages = []
        t0 = time.perf_counter()
        # 1. Classical pre-filter (always -- cheap)
        surviving_ids = np.arange(n_total, dtype=np.int64)
        if classical:
            t1 = time.perf_counter()
            surviving_ids = _apply_classical_filter(table, classical, col_cache=self.classical_cache)
            t_classical = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": "classical", "kept": len(surviving_ids),
                           "ms": t_classical})
        # 1b. Fuzzy pre-filter
        if fuzzy and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            surviving_ids = _apply_fuzzy_filter(table, surviving_ids, fuzzy)
            t_fz = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"fuzzy({len(fuzzy)} preds)",
                           "kept": len(surviving_ids), "ms": t_fz})
        # 2. Vector pre-filter (brute force over surviving subset)
        # The previous HNSW-based path is equivalent in result after
        # the faiss-IDSelector fix in ``physical/ann.py`` but paid
        # a 200+ s HNSW build at n = 500k. Direct brute-force numpy
        # over the cached column avoids the build entirely.
        if vectors and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            for vp in vectors:
                q = query_params.get(vp.q_param)
                if q is None:
                    continue
                q_arr = np.asarray(q, dtype="float32")
                new_ids = brute_force_filtered_range(
                    self.store, vp.column, q_arr,
                    float(vp.threshold), surviving_ids,
                    vec_cache=self.vec_cache,
                )
                # Vectorized np.isin (replaces ~50 ms Python loop at n=1M)
                mask = np.isin(surviving_ids, np.asarray(new_ids, dtype=np.int64),
                               assume_unique=True)
                surviving_ids = surviving_ids[mask]
            t_vec = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"vector_hnsw({len(vectors)} preds)",
                           "kept": len(surviving_ids), "ms": t_vec})
        # 3. Text BM25 (post-filter over surviving)
        if texts and len(surviving_ids) > 0:
            if self.bm25_cache is None:
                self.bm25_cache = build_bm25_cache(
                    table,
                    [tp.column for tp in texts
                     if tp.column in table.column_names],
                )
            t1 = time.perf_counter()
            surviving_ids = _apply_bm25_filter(
                table, surviving_ids, texts, query_params,
                bm25_cache=self.bm25_cache,
            )
            t_txt = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"text_bm25({len(texts)} preds)",
                           "kept": len(surviving_ids), "ms": t_txt})
        # 4. Graph BFS (post-filter)
        if graphs and len(surviving_ids) > 0 and self.graph is not None:
            t1 = time.perf_counter()
            surviving_ids = _apply_graph_filter(
                table, surviving_ids, graphs, self.graph, query_params
            )
            t_gr = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"graph_bfs({len(graphs)} preds)",
                           "kept": len(surviving_ids), "ms": t_gr})
        t_total = (time.perf_counter() - t0) * 1000.0
        recall = 0.0
        if ground_truth_ids is not None and len(ground_truth_ids) > 0:
            got = set(int(x) for x in surviving_ids)
            ref = set(int(x) for x in ground_truth_ids)
            recall = len(got & ref) / max(1, len(ref))
        sel = (len(ground_truth_ids) / max(1, n_total)) if ground_truth_ids is not None else 0.0
        return BaselineResult(
            system="navix", query_id=query_id,
            n_input=n_total, n_predicate_selectivity=sel,
            n_candidate=len(surviving_ids), wall_clock_ms=t_total,
            recall=recall, stages=stages,
        )


# ---------------------------------------------------------------------
# ACORN-style baseline
# ---------------------------------------------------------------------

class ACORNStyle:
    """ACORN-σ with σ=1.0 + post-BM25 + post-graph-filter.

    ACORN (arXiv 2022) supports attribute filters via
    predicate-agnostic HNSW with extra edges.  ACORN does
    *not* natively support BM25 text or graph
    reachability; we add the minimal post-filter passes.
    The "vector first, others post-filter" order is the
    natural ACORN pipeline when text and graph are
    unavailable.
    """
    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None,
                 workload_size: int = 16) -> None:
        self.store = store
        self.graph = graph
        # Lazy cache for built HNSW indexes; the build-vs-skip
        # decision is delegated to the IndexManager (instead of the
        # previous ``N_SKIP_HNSW_BUILD = 50_000`` hardcoded override in
        # :mod:`cbo`). ``workload_size`` is the expected number of
        # queries that will run against this stack; the sweep driver
        # sets it explicitly. Default 16 ≈ small benchmark.
        self.hnsw_cache: Dict[str, Any] = {}
        self.index_mgr = IndexManager(workload_size=workload_size)
        # Cache BM25 indexes per text column; built lazily on
        # first text query, reused for the rest of the sweep.
        # Pays the one-time build cost in exchange for not
        # rebuilding over the surviving subset per query (the
        # latter was the dominant per-query cost at n = 500k).
        self.bm25_cache: Optional[Dict[str, Tuple[List[str], Any]]] = None
        # Per-system cache for the Arrow->numpy column conversion.
        # ``brute_force_filtered_range`` is called once per vector
        # predicate per query (>= 1 per query); without the cache
        # the conversion (~100 ms at n = 500k) is paid per call.
        self.vec_cache: Dict[str, np.ndarray] = {}
        # Per-system cache for the Arrow->numpy classical-column
        # conversion (avoids re-materializing the same string
        # column on every query — ~70 ms at n=1M).
        self.classical_cache: Dict[Tuple[int, str], np.ndarray] = {}

    def _maybe_ensure_hnsw(self, plan) -> None:
        """DEPRECATED per-query helper — kept as a thin alias for
        backwards compatibility. The supported path is
        ``self.index_mgr.pre_build([(col, vec, dim), ...])`` invoked
        once from the workload driver (sweep setup / app startup).
        This method should not be called from a per-query hot path.
        """
        from .baselines import _collect_predicates_v2 as _c
        try:
            _, vectors, _, _, _ = _c(plan)
        except Exception:
            vectors = []
        cols = []
        for vp in vectors:
            if vp.column not in self.store.arrow.column_names:
                continue
            vec = self.store.vector_column(vp.column)
            cols.append((vp.column, vec, vec.shape[1]))
        if cols:
            self.index_mgr.pre_build(cols)

    def _get_hnsw(self, vec_col, column_name):
        """Back-compat thin wrapper around IndexManager.

        Returns the cached or freshly-built HNSW index for the
        column, or ``None`` if the IndexManager decided the workload
        is too small to justify the build (the caller should then
        fall back to a brute-force exact scan). The efSearch used is
        now picked by the IndexManager (``choose_hnsw_params``),
        not a hard-coded 512.

        Use ``self.index_mgr.ensure_hnsw`` directly in new code; this
        thin wrapper is retained for the PostFilterStyle call site
        that still uses ``self._get_hnsw``.
        """
        has_hnsw, hnsw = self.index_mgr.ensure_hnsw(
            column_name, vec_col, dim=vec_col.shape[1],
        )
        return hnsw

    def run(
        self,
        plan: Plan,
        query_params: Optional[Dict[str, Any]] = None,
        query_id: str = "q",
        ground_truth_ids: Optional[np.ndarray] = None,
    ) -> BaselineResult:
        query_params = query_params or {}
        table = self.store.arrow
        n_total = table.num_rows
        classical, vectors, texts, graphs, fuzzy = _collect_predicates_v2(plan)
        stages = []
        t0 = time.perf_counter()
        # ACORN: no classical pre-filter (only attribute-supported
        # in original paper; we extend by also pre-filtering
        # classical since it's a free win).
        surviving_ids = np.arange(n_total, dtype=np.int64)
        if classical:
            t1 = time.perf_counter()
            surviving_ids = _apply_classical_filter(table, classical, col_cache=self.classical_cache)
            t_classical = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": "classical", "kept": len(surviving_ids),
                           "ms": t_classical})
        # 1b. Fuzzy pre-filter
        if fuzzy and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            surviving_ids = _apply_fuzzy_filter(table, surviving_ids, fuzzy)
            t_fz = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"fuzzy({len(fuzzy)} preds)",
                           "kept": len(surviving_ids), "ms": t_fz})
        # 1. Vector pre-filter (brute force over surviving subset)
        # Same reasoning as NaviXStyle: HNSW equivalent after the
        # faiss-IDSelector fix, but avoids the 200+ s HNSW build
        # at n = 500k by going through a cached vector column.
        if vectors and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            for vp in vectors:
                q = query_params.get(vp.q_param)
                if q is None:
                    continue
                q_arr = np.asarray(q, dtype="float32")
                new_ids = brute_force_filtered_range(
                    self.store, vp.column, q_arr,
                    float(vp.threshold), surviving_ids,
                    vec_cache=self.vec_cache,
                )
                # Vectorized np.isin (replaces ~50 ms Python loop at n=1M)
                mask = np.isin(surviving_ids, np.asarray(new_ids, dtype=np.int64),
                               assume_unique=True)
                surviving_ids = surviving_ids[mask]
            t_vec = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"acorn_hnsw({len(vectors)} preds)",
                           "kept": len(surviving_ids), "ms": t_vec})
        # 2. BM25 post-filter (text)
        if texts and len(surviving_ids) > 0:
            if self.bm25_cache is None:
                self.bm25_cache = build_bm25_cache(
                    table,
                    [tp.column for tp in texts
                     if tp.column in table.column_names],
                )
            t1 = time.perf_counter()
            surviving_ids = _apply_bm25_filter(
                table, surviving_ids, texts, query_params,
                bm25_cache=self.bm25_cache,
            )
            t_txt = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"post_bm25({len(texts)} preds)",
                           "kept": len(surviving_ids), "ms": t_txt})
        # 3. Graph post-filter
        if graphs and len(surviving_ids) > 0 and self.graph is not None:
            t1 = time.perf_counter()
            surviving_ids = _apply_graph_filter(
                table, surviving_ids, graphs, self.graph, query_params
            )
            t_gr = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"post_graph({len(graphs)} preds)",
                           "kept": len(surviving_ids), "ms": t_gr})
        t_total = (time.perf_counter() - t0) * 1000.0
        recall = 0.0
        if ground_truth_ids is not None and len(ground_truth_ids) > 0:
            got = set(int(x) for x in surviving_ids)
            ref = set(int(x) for x in ground_truth_ids)
            recall = len(got & ref) / max(1, len(ref))
        sel = (len(ground_truth_ids) / max(1, n_total)) if ground_truth_ids is not None else 0.0
        return BaselineResult(
            system="acorn", query_id=query_id,
            n_input=n_total, n_predicate_selectivity=sel,
            n_candidate=len(surviving_ids), wall_clock_ms=t_total,
            recall=recall, stages=stages,
        )


class PreFilterStyle:
    """Pre-filter HNSW (pgvector / Milvus / Weaviate default).

    Represents the *de facto industry* approach: run a
    boolean filter on the classical column first to get a
    candidate set R_pre, then run an HNSW range query
    restricted to R_pre via faiss ``IDSelector`` (the
    pre-filter variant of ACORN's pre-filter).  Text and
    graph predicates are *post*-filter passes on the
    surviving set, because pre-filter would require a
    per-predicate index that pgvector / Milvus do not
    expose.

    No RBO, no CBO, no ρ-aware amplifier, no adaptive
    re-planning.  Vector strategy is always HNSW.
    """

    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None,
                 workload_size: int = 16) -> None:
        self.store = store
        self.graph = graph
        # Lazy cache for built HNSW indexes; the build-vs-skip
        # decision is delegated to the IndexManager (instead of the
        # previous ``N_SKIP_HNSW_BUILD = 50_000`` hardcoded override in
        # :mod:`cbo`). ``workload_size`` is the expected number of
        # queries that will run against this stack; the sweep driver
        # sets it explicitly. Default 16 ≈ small benchmark.
        self.hnsw_cache: Dict[str, Any] = {}
        self.index_mgr = IndexManager(workload_size=workload_size)
        # Cache BM25 indexes per text column; built lazily on
        # first text query, reused for the rest of the sweep.
        # Pays the one-time build cost in exchange for not
        # rebuilding over the surviving subset per query (the
        # latter was the dominant per-query cost at n = 500k).
        self.bm25_cache: Optional[Dict[str, Tuple[List[str], Any]]] = None
        # Per-system cache for the Arrow->numpy column conversion.
        # ``brute_force_filtered_range`` is called once per vector
        # predicate per query (>= 1 per query); without the cache
        # the conversion (~100 ms at n = 500k) is paid per call.
        self.vec_cache: Dict[str, np.ndarray] = {}
        # Per-system cache for the Arrow->numpy classical-column
        # conversion (avoids re-materializing the same string
        # column on every query — ~70 ms at n=1M).
        self.classical_cache: Dict[Tuple[int, str], np.ndarray] = {}

    def _maybe_ensure_hnsw(self, plan) -> None:
        """DEPRECATED per-query helper — kept as a thin alias for
        backwards compatibility. The supported path is
        ``self.index_mgr.pre_build([(col, vec, dim), ...])`` invoked
        once from the workload driver (sweep setup / app startup).
        This method should not be called from a per-query hot path.
        """
        from .baselines import _collect_predicates_v2 as _c
        try:
            _, vectors, _, _, _ = _c(plan)
        except Exception:
            vectors = []
        cols = []
        for vp in vectors:
            if vp.column not in self.store.arrow.column_names:
                continue
            vec = self.store.vector_column(vp.column)
            cols.append((vp.column, vec, vec.shape[1]))
        if cols:
            self.index_mgr.pre_build(cols)

    def _get_hnsw(self, vec_col, column_name):
        """Back-compat thin wrapper around IndexManager.

        Returns the cached or freshly-built HNSW index for the
        column, or ``None`` if the IndexManager decided the workload
        is too small to justify the build (the caller should then
        fall back to a brute-force exact scan). The efSearch used is
        now picked by the IndexManager (``choose_hnsw_params``),
        not a hard-coded 512.

        Use ``self.index_mgr.ensure_hnsw`` directly in new code; this
        thin wrapper is retained for the PostFilterStyle call site
        that still uses ``self._get_hnsw``.
        """
        has_hnsw, hnsw = self.index_mgr.ensure_hnsw(
            column_name, vec_col, dim=vec_col.shape[1],
        )
        return hnsw

    def run(
        self,
        plan: Plan,
        query_params: Optional[Dict[str, Any]] = None,
        query_id: str = "q",
        ground_truth_ids: Optional[np.ndarray] = None,
    ) -> BaselineResult:
        query_params = query_params or {}
        table = self.store.arrow
        n_total = table.num_rows
        classical, vectors, texts, graphs, fuzzy = _collect_predicates_v2(plan)
        stages: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        surviving = np.arange(n_total, dtype=np.int64)
        # 1. Classical pre-filter (always)
        if classical:
            t1 = time.perf_counter()
            surviving = _apply_classical_filter(table, classical, col_cache=self.classical_cache)
            stages.append({"stage": "classical_pre",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 2. Fuzzy pre-filter (Q4 only; otherwise keep as no-op)
        if fuzzy and len(surviving) > 0:
            t1 = time.perf_counter()
            surviving = _apply_fuzzy_filter(table, surviving, fuzzy)
            stages.append({"stage": f"fuzzy_pre({len(fuzzy)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 3. HNSW pre-filter on the surviving set
        if vectors and len(surviving) > 0:
            t1 = time.perf_counter()
            for vp in vectors:
                q = query_params.get(vp.q_param)
                if q is None:
                    continue
                q_arr = np.asarray(q, dtype="float32")
                new_ids = brute_force_filtered_range(
                    self.store, vp.column, q_arr,
                    float(vp.threshold), surviving,
                    vec_cache=self.vec_cache,
                )
                # Vectorized np.isin (replaces ~50 ms Python loop at n=1M)
                mask = np.isin(surviving, np.asarray(new_ids, dtype=np.int64),
                               assume_unique=True)
                surviving = surviving[mask]
            stages.append({"stage": f"hnsw_pre({len(vectors)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 4. BM25 post-filter (text — no index, scan surviving)
        if texts and len(surviving) > 0:
            if self.bm25_cache is None:
                self.bm25_cache = build_bm25_cache(
                    table,
                    [tp.column for tp in texts
                     if tp.column in table.column_names],
                )
            t1 = time.perf_counter()
            surviving = _apply_bm25_filter(
                table, surviving, texts, query_params,
                bm25_cache=self.bm25_cache,
            )
            stages.append({"stage": f"post_bm25({len(texts)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 5. Graph BFS post-filter
        if graphs and len(surviving) > 0 and self.graph is not None:
            t1 = time.perf_counter()
            surviving = _apply_graph_filter(
                table, surviving, graphs, self.graph, query_params
            )
            stages.append({"stage": f"post_graph({len(graphs)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        t_total = (time.perf_counter() - t0) * 1000.0
        recall = 0.0
        if ground_truth_ids is not None and len(ground_truth_ids) > 0:
            got = set(int(x) for x in surviving)
            ref = set(int(x) for x in ground_truth_ids)
            recall = len(got & ref) / max(1, len(ref))
        sel = (len(ground_truth_ids) / max(1, n_total)) if ground_truth_ids is not None else 0.0
        return BaselineResult(
            system="prefilter", query_id=query_id,
            n_input=n_total, n_predicate_selectivity=sel,
            n_candidate=len(surviving), wall_clock_ms=t_total,
            recall=recall, stages=stages,
        )


class PostFilterStyle:
    """Post-filter HNSW (naive baseline).

    Represents the *naive* approach: run HNSW on all ``n``
    rows first, then post-filter the surviving set with
    classical, BM25, and BFS.  Vector strategy is always
    HNSW (built once and reused).  No RBO, no CBO, no
    ρ-aware amplifier.  This is the "no plan enumeration,
    no pre-filter" baseline.
    """

    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None,
                 workload_size: int = 16) -> None:
        self.store = store
        self.graph = graph
        # Lazy cache for built HNSW indexes; the build-vs-skip
        # decision is delegated to the IndexManager (instead of the
        # previous ``N_SKIP_HNSW_BUILD = 50_000`` hardcoded override in
        # :mod:`cbo`). ``workload_size`` is the expected number of
        # queries that will run against this stack; the sweep driver
        # sets it explicitly. Default 16 ≈ small benchmark.
        self.hnsw_cache: Dict[str, Any] = {}
        self.index_mgr = IndexManager(workload_size=workload_size)
        # Cache BM25 indexes per text column; built lazily on
        # first text query, reused for the rest of the sweep.
        # Pays the one-time build cost in exchange for not
        # rebuilding over the surviving subset per query (the
        # latter was the dominant per-query cost at n = 500k).
        self.bm25_cache: Optional[Dict[str, Tuple[List[str], Any]]] = None
        # Per-system cache for the Arrow->numpy column conversion.
        # ``brute_force_filtered_range`` is called once per vector
        # predicate per query (>= 1 per query); without the cache
        # the conversion (~100 ms at n = 500k) is paid per call.
        self.vec_cache: Dict[str, np.ndarray] = {}
        # Per-system cache for the Arrow->numpy classical-column
        # conversion (avoids re-materializing the same string
        # column on every query — ~70 ms at n=1M).
        self.classical_cache: Dict[Tuple[int, str], np.ndarray] = {}

    def _maybe_ensure_hnsw(self, plan) -> None:
        """DEPRECATED per-query helper — kept as a thin alias for
        backwards compatibility. The supported path is
        ``self.index_mgr.pre_build([(col, vec, dim), ...])`` invoked
        once from the workload driver (sweep setup / app startup).
        This method should not be called from a per-query hot path.
        """
        from .baselines import _collect_predicates_v2 as _c
        try:
            _, vectors, _, _, _ = _c(plan)
        except Exception:
            vectors = []
        cols = []
        for vp in vectors:
            if vp.column not in self.store.arrow.column_names:
                continue
            vec = self.store.vector_column(vp.column)
            cols.append((vp.column, vec, vec.shape[1]))
        if cols:
            self.index_mgr.pre_build(cols)

    def _get_hnsw(self, vec_col, column_name):
        """Back-compat thin wrapper around IndexManager.

        Returns the cached or freshly-built HNSW index for the
        column, or ``None`` if the IndexManager decided the workload
        is too small to justify the build (the caller should then
        fall back to a brute-force exact scan). The efSearch used is
        now picked by the IndexManager (``choose_hnsw_params``),
        not a hard-coded 512.

        Use ``self.index_mgr.ensure_hnsw`` directly in new code; this
        thin wrapper is retained for the PostFilterStyle call site
        that still uses ``self._get_hnsw``.
        """
        has_hnsw, hnsw = self.index_mgr.ensure_hnsw(
            column_name, vec_col, dim=vec_col.shape[1],
        )
        return hnsw

    def run(
        self,
        plan: Plan,
        query_params: Optional[Dict[str, Any]] = None,
        query_id: str = "q",
        ground_truth_ids: Optional[np.ndarray] = None,
    ) -> BaselineResult:
        query_params = query_params or {}
        table = self.store.arrow
        n_total = table.num_rows
        classical, vectors, texts, graphs, fuzzy = _collect_predicates_v2(plan)
        stages: List[Dict[str, Any]] = []
        t0 = time.perf_counter()
        # 1. HNSW on all n (no classical pre-filter)
        surviving = np.arange(n_total, dtype=np.int64)
        if vectors and len(surviving) > 0:
            t1 = time.perf_counter()
            for vp in vectors:
                q = query_params.get(vp.q_param)
                if q is None:
                    continue
                q_arr = np.asarray(q, dtype="float32")
                vec_col = self.store.vector_column(vp.column)
                hnsw = self._get_hnsw(vec_col, vp.column)
                if hnsw is None:
                    # IndexManager decided the workload is too small
                    # to justify HNSW build; fall back to the same
                    # exact brute force the other baselines use.
                    ids = brute_force_filtered_range(
                        self.store, vp.column, q_arr,
                        float(vp.threshold), surviving,
                        vec_cache=self.vec_cache,
                    )
                else:
                    ids, _ = hnsw.range_query(q_arr, float(vp.threshold))
                surviving = np.asarray(ids, dtype=np.int64)
            stages.append({"stage": f"hnsw_full({len(vectors)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 2. Classical post-filter
        if classical and len(surviving) > 0:
            t1 = time.perf_counter()
            surviving = _apply_classical_filter_on(table, surviving, classical)
            stages.append({"stage": "classical_post",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 3. Fuzzy post-filter
        if fuzzy and len(surviving) > 0:
            t1 = time.perf_counter()
            surviving = _apply_fuzzy_filter(table, surviving, fuzzy)
            stages.append({"stage": f"fuzzy_post({len(fuzzy)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 4. BM25 post-filter
        if texts and len(surviving) > 0:
            if self.bm25_cache is None:
                self.bm25_cache = build_bm25_cache(
                    table,
                    [tp.column for tp in texts
                     if tp.column in table.column_names],
                )
            t1 = time.perf_counter()
            surviving = _apply_bm25_filter(
                table, surviving, texts, query_params,
                bm25_cache=self.bm25_cache,
            )
            stages.append({"stage": f"post_bm25({len(texts)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 5. Graph BFS post-filter
        if graphs and len(surviving) > 0 and self.graph is not None:
            t1 = time.perf_counter()
            surviving = _apply_graph_filter(
                table, surviving, graphs, self.graph, query_params
            )
            stages.append({"stage": f"post_graph({len(graphs)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        t_total = (time.perf_counter() - t0) * 1000.0
        recall = 0.0
        if ground_truth_ids is not None and len(ground_truth_ids) > 0:
            got = set(int(x) for x in surviving)
            ref = set(int(x) for x in ground_truth_ids)
            recall = len(got & ref) / max(1, len(ref))
        sel = (len(ground_truth_ids) / max(1, n_total)) if ground_truth_ids is not None else 0.0
        return BaselineResult(
            system="postfilter", query_id=query_id,
            n_input=n_total, n_predicate_selectivity=sel,
            n_candidate=len(surviving), wall_clock_ms=t_total,
            recall=recall, stages=stages,
        )


# ---------------------------------------------------------------------
# Convenience: get all systems for a sweep
# ---------------------------------------------------------------------

def all_systems(
    store: TableStore, graph: Optional[GraphStore] = None
) -> Dict[str, Any]:
    """Return the five systems as a dict."""
    return {
        "mosaicopt": MosaicOptSystem(store, graph),
        "navix": NaviXStyle(store, graph),
        "acorn": ACORNStyle(store, graph),
        "prefilter": PreFilterStyle(store, graph),
        "postfilter": PostFilterStyle(store, graph),
    }
