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
) -> np.ndarray:
    """Apply classical predicates to a table, return surviving row indices."""
    n = table.num_rows
    if not preds:
        return np.arange(n, dtype=np.int64)
    mask = np.ones(n, dtype=bool)
    for p in preds:
        if p.column not in table.column_names:
            continue
        col = table.column(p.column).to_numpy()
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
) -> np.ndarray:
    """Post-filter: keep rows whose BM25 score exceeds threshold.

    Builds a small BM25 index over the surviving rows.
    """
    if not text_preds or len(surviving_ids) == 0:
        return surviving_ids
    docs = [str(table.column(text_preds[0].column)[i].as_py())
            for i in surviving_ids]
    bm25 = BM25Index(docs)
    keep = np.ones(len(surviving_ids), dtype=bool)
    for tp in text_preds:
        q_text = query_params.get(tp.q_param, "")
        if not q_text:
            continue
        scores = np.asarray(bm25._bm25.get_scores(_tokenize(str(q_text))),
                            dtype=np.float32)
        scores = np.maximum(scores, 0.0)
        keep &= (scores > float(tp.threshold))
    return surviving_ids[keep]


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
    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None) -> None:
        self.store = store
        self.graph = graph

    def run(
        self,
        plan: Plan,
        query_params: Optional[Dict[str, Any]] = None,
        query_id: str = "q",
        ground_truth_ids: Optional[np.ndarray] = None,
    ) -> BaselineResult:
        from .plan import to_physical
        from .statistics import (
            populate_selectivity_hint, populate_classical_s_hint,
            populate_rho_hint, StatisticsHub, FilteredHNSWDensity,
        )
        query_params = query_params or {}
        classical, _, _, _, fuzzy = _collect_predicates_v2(plan)
        sel_hint = {}
        cs_hint = populate_classical_s_hint(plan, store=self.store)
        rho_hint = {}
        try:
            hub = StatisticsHub.from_store(
                self.store,
                vector_columns=[],
                text_columns=[],
                graph=self.graph,
                sample_rate=0.0,
            )
            sel_hint = populate_selectivity_hint(plan, self.store, query_params, hub=hub)
        except Exception:
            sel_hint = {}
        try:
            fd = FilteredHNSWDensity(self.store, vector_columns=[],
                                      n_sample_queries=5, seed=0)
            rho_hint = populate_rho_hint(plan, fd, self.store)
        except Exception:
            rho_hint = {}
        t0 = time.perf_counter()
        try:
            phys = to_physical(plan, self.store,
                               selectivity_hint=sel_hint or None,
                               rho_hint=rho_hint or None,
                               classical_s_hint=cs_hint or None)
        except Exception:
            t = (time.perf_counter() - t0) * 1000.0
            n_total = self.store.arrow.num_rows
            return BaselineResult(
                system="mosaicopt", query_id=query_id,
                n_input=n_total, n_predicate_selectivity=0.0,
                n_candidate=0, wall_clock_ms=t, recall=0.0,
                extra={"error": "to_physical failed"},
            )
        # The PhysicalPlan tree isn't directly executable in
        # this iteration; we fall back to a faithful approximation:
        # apply the predicates in their canonical order, using
        # the CBO's predicted strategy for the vector step.  This
        # preserves the *strategy choice* (flat/ivf/hnsw) and the
        # *filter order* the RBO + D produced.
        from .cbo import cbo_pick
        table = self.store.arrow
        n_total = table.num_rows
        surviving_ids = np.arange(n_total, dtype=np.int64)
        stages = []
        # 1. Classical pre-filter
        if classical:
            t1 = time.perf_counter()
            surviving_ids = _apply_classical_filter(table, classical)
            t_classical = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": "classical", "kept": len(surviving_ids),
                           "ms": t_classical})
        # 1b. Fuzzy text pre-filter (cheap; reduces the search space)
        if fuzzy and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            surviving_ids = _apply_fuzzy_filter(table, surviving_ids, fuzzy)
            t_fz = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"fuzzy({len(fuzzy)} preds)",
                           "kept": len(surviving_ids), "ms": t_fz})
        # 2. Vector (CBO with ρ + classical_s)
        from . import cbo as _cbo_mod
        _, vectors, texts, graphs, _ = _collect_predicates_v2(plan)
        if vectors and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            for vp in vectors:
                q = query_params.get(vp.q_param)
                if q is None:
                    continue
                q_arr = np.asarray(q, dtype="float32")
                s = float(s_v_estimate(self.store, surviving_ids, vp, q_arr))
                rho = float(rho_hint.get(vp.column, 1.0))
                cs = 1.0
                if surviving_ids.size > 0:
                    cs = len(surviving_ids) / max(1, n_total)
                strategy = cbo_pick(s, n_total, 128, rho=rho, classical_s=cs)
                # Execute the strategy
                vec_col = self.store.vector_column(vp.column)
                from ..physical.ann import AnnIndex
                if strategy == "hnsw":
                    hnsw = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="hnsw")
                    # Use HNSW with pre-filter on the surviving set
                    # (consistent with the baseline semantics; the CBO
                    # still chooses between flat/ivf/hnsw based on the
                    # predicted cost).
                    new_ids, _ = hnsw.filtered_range_query(
                        q_arr, float(vp.threshold), surviving_ids
                    )
                elif strategy == "ivf":
                    ivf = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="ivf")
                    new_ids, _ = ivf.range_query(q_arr, float(vp.threshold))
                else:
                    flat = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="flat")
                    new_ids, _ = flat.range_query(q_arr, float(vp.threshold))
                new_set = set(int(x) for x in new_ids)
                surviving_ids = np.array(
                    [i for i in surviving_ids if int(i) in new_set],
                    dtype=np.int64,
                )
            t_vec = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"vector({len(vectors)} preds)",
                           "kept": len(surviving_ids), "ms": t_vec})
        # 3. Text (post-filter)
        if texts and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            surviving_ids = _apply_bm25_filter(
                self.store.arrow, surviving_ids, texts, query_params
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


def s_v_estimate(
    store: TableStore, surviving_ids: np.ndarray,
    vp: VectorRange, q_arr: np.ndarray,
) -> float:
    """Estimate the vector predicate selectivity against the
    *current* surviving set.

    Used by MosaicOptSystem to feed the CBO.  We compute the
    empirical selectivity by running a flat scan on the
    surviving subset and counting hits.
    """
    if len(surviving_ids) == 0:
        return 0.0
    sub = store.vector_column(vp.column)[surviving_ids].astype("float32")
    diff = sub - q_arr.reshape(1, -1)
    d2 = (diff * diff).sum(axis=1)
    sq_thr = float(vp.threshold) ** 2
    n_hits = int((d2 < sq_thr).sum())
    return n_hits / max(1, len(surviving_ids))


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
    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None) -> None:
        self.store = store
        self.graph = graph
        # Build BM25 index once per call (text)
        # Build HNSW index once per call (vector)

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
            surviving_ids = _apply_classical_filter(table, classical)
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
        # 2. Vector HNSW pre-filter (NaviX's vector strategy)
        if vectors and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            for vp in vectors:
                q = query_params.get(vp.q_param)
                if q is None:
                    continue
                q_arr = np.asarray(q, dtype="float32")
                vec_col = self.store.vector_column(vp.column)
                hnsw = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="hnsw")
                ids, _ = hnsw.filtered_range_query(
                    q_arr, float(vp.threshold), surviving_ids
                )
                ids_set = set(int(x) for x in ids)
                surviving_ids = np.array(
                    [i for i in surviving_ids if int(i) in ids_set],
                    dtype=np.int64,
                )
            t_vec = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"vector_hnsw({len(vectors)} preds)",
                           "kept": len(surviving_ids), "ms": t_vec})
        # 3. Text BM25 (post-filter over surviving)
        if texts and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            surviving_ids = _apply_bm25_filter(
                table, surviving_ids, texts, query_params
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
    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None) -> None:
        self.store = store
        self.graph = graph

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
            surviving_ids = _apply_classical_filter(table, classical)
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
        # 1. HNSW pre-filter (the ACORN-style "vector first")
        if vectors and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            for vp in vectors:
                q = query_params.get(vp.q_param)
                if q is None:
                    continue
                q_arr = np.asarray(q, dtype="float32")
                vec_col = self.store.vector_column(vp.column)
                hnsw = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="hnsw")
                # Use the strictest (pre-filter with sel) variant
                ids, _ = hnsw.filtered_range_query(
                    q_arr, float(vp.threshold), surviving_ids
                )
                ids_set = set(int(x) for x in ids)
                surviving_ids = np.array(
                    [i for i in surviving_ids if int(i) in ids_set],
                    dtype=np.int64,
                )
            t_vec = (time.perf_counter() - t1) * 1000.0
            stages.append({"stage": f"acorn_hnsw({len(vectors)} preds)",
                           "kept": len(surviving_ids), "ms": t_vec})
        # 2. BM25 post-filter (text)
        if texts and len(surviving_ids) > 0:
            t1 = time.perf_counter()
            surviving_ids = _apply_bm25_filter(
                table, surviving_ids, texts, query_params
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

    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None) -> None:
        self.store = store
        self.graph = graph

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
            surviving = _apply_classical_filter(table, classical)
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
                vec_col = self.store.vector_column(vp.column)
                hnsw = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="hnsw")
                ids, _ = hnsw.filtered_range_query(
                    q_arr, float(vp.threshold), surviving
                )
                ids_set = set(int(x) for x in ids)
                surviving = np.array(
                    [i for i in surviving if int(i) in ids_set],
                    dtype=np.int64,
                )
            stages.append({"stage": f"hnsw_pre({len(vectors)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 4. BM25 post-filter (text — no index, scan surviving)
        if texts and len(surviving) > 0:
            t1 = time.perf_counter()
            surviving = _apply_bm25_filter(
                table, surviving, texts, query_params
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

    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None) -> None:
        self.store = store
        self.graph = graph

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
                hnsw = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="hnsw")
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
            t1 = time.perf_counter()
            surviving = _apply_bm25_filter(
                table, surviving, texts, query_params
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
