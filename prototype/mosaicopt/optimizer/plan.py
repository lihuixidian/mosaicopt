"""Logical -> physical plan mapping.

This module is the "bridge" called out in algebra.md Section 4.2:
each logical operator is bound to one or more physical operators
and the executor walks the resulting tree.

We keep the physical plan tree as a small set of dataclasses that
mirror the logical plan, with two additions:
  * TopK      - a heap-based top-K, produced by the T1 rewrite
  * Sort      - a full sort, used when T1 is not applicable
  * BM25ScanP - a BM25 top-K / threshold scan

The RBO produces a tree containing a ``_TopKFlag`` for the
``Tau + Lambda`` pair; the ``to_physical`` function recognizes it
and emits a TopK.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from mosaicopt.algebra.ops import (
    Plan, Scan, Project, Select, Materialize, Order, Limit, Expand, Join,
)
from mosaicopt.algebra.predicates import (
    Predicate, ColumnPredicate, VectorRange, BM25Score, GraphReach, And,
)
from mosaicopt.algebra.materialization import MaterializeExpr, distance, const
from mosaicopt.optimizer.rbo import _TopKFlag


# ---------------------------------------------------------------------
# Physical plan nodes
# ---------------------------------------------------------------------

@dataclass
class PhysicalPlan:
    """Marker class for physical plan nodes."""
    op: str = "?"


class ScanP(PhysicalPlan):
    store: Any                       # TableStore
    name: str
    op: str = "ScanP"

    def __init__(self, store=None, name="", op="ScanP"):
        self.store = store
        self.name = name
        self.op = op


class SelectP(PhysicalPlan):
    child: PhysicalPlan
    predicate: Predicate
    vec_strategy: Optional[str] = None
    vec_threshold: Optional[float] = None
    op: str = "SelectP"

    def __init__(self, child=None, predicate=None, vec_strategy=None, vec_threshold=None, op="SelectP"):
        self.child = child
        self.predicate = predicate
        self.vec_strategy = vec_strategy
        self.vec_threshold = vec_threshold
        self.op = op


class MaterializeP(PhysicalPlan):
    child: PhysicalPlan
    src_col: str
    expr: MaterializeExpr
    alias: str
    op: str = "MaterializeP"

    def __init__(self, child=None, src_col="", expr=None, alias="", op="MaterializeP"):
        self.child = child
        self.src_col = src_col
        self.expr = expr
        self.alias = alias
        self.op = op


class ExpandP(PhysicalPlan):
    child: PhysicalPlan
    seed_col: str
    rule_name: str
    depth: int
    op: str = "ExpandP"

    def __init__(self, child=None, seed_col="", rule_name="", depth=1, op="ExpandP"):
        self.child = child
        self.seed_col = seed_col
        self.rule_name = rule_name
        self.depth = depth
        self.op = op


class JoinP(PhysicalPlan):
    left: PhysicalPlan
    right: PhysicalPlan
    condition: Predicate
    op: str = "JoinP"

    def __init__(self, left=None, right=None, condition=None, op="JoinP"):
        self.left = left
        self.right = right
        self.condition = condition
        self.op = op


class Sort(PhysicalPlan):
    child: PhysicalPlan
    key_col: str
    key_expr: str
    ascending: bool
    op: str = "Sort"

    def __init__(self, child=None, key_col="", key_expr="", ascending=True, op="Sort"):
        self.child = child
        self.key_col = key_col
        self.key_expr = key_expr
        self.ascending = ascending
        self.op = op


class TopK(PhysicalPlan):
    """Heap-based top-K. Produced by the T1 rewrite."""
    child: PhysicalPlan
    key_col: str
    key_expr: str
    ascending: bool
    k: int
    op: str = "TopK"

    def __init__(self, child=None, key_col="", key_expr="", ascending=True, k=10, op="TopK"):
        self.child = child
        self.key_col = key_col
        self.key_expr = key_expr
        self.ascending = ascending
        self.k = k
        self.op = op


class LimitP(PhysicalPlan):
    child: PhysicalPlan
    k: int
    op: str = "LimitP"

    def __init__(self, child=None, k=10, op="LimitP"):
        self.child = child
        self.k = k
        self.op = op


class BM25ScanP(PhysicalPlan):
    """A BM25 top-K/threshold scan on a text column."""
    child: PhysicalPlan
    text_col: str
    q_param: str
    threshold: Optional[float] = None
    k: Optional[int] = None
    op: str = "BM25ScanP"

    def __init__(self, child=None, text_col="", q_param="", threshold=None, k=None, op="BM25ScanP"):
        self.child = child
        self.text_col = text_col
        self.q_param = q_param
        self.threshold = threshold
        self.k = k
        self.op = op


# ---------------------------------------------------------------------
# Logical -> physical mapping
# ---------------------------------------------------------------------

def _vector_dim_fast(store, column: str) -> int:
    """Get the dimension of a vector column WITHOUT materializing the
    full (n, d) numpy array.

    The previous implementation called ``store.vector_column(...)``
    just to read ``.shape[1]``; at n=1M d=128 that's a 488 MB
    Arrow→numpy conversion paid per query. The dim is a property
    of the storage layout, so it should not require materializing
    the column.

    Falls back gracefully when the schema cannot answer directly.
    """
    if store is None or column not in store.arrow.column_names:
        return 768
    field = store.arrow.schema.field(column)
    t = field.type
    if pa.types.is_fixed_size_list(t):
        return int(t.list_size)
    arr = store.arrow.column(column)
    n = len(arr)
    if n <= 0:
        return 0
    # For pa.list_(pa.float32()) the inner ``values`` chunked array
    # stores all floats contiguously. ``len(values) / n`` gives d
    # without materializing the full (n, d) array.
    try:
        return int(len(arr.values) // n)
    except Exception:
        return 768  # safe default


def to_physical(
    plan: Plan,
    store: Any = None,
    selectivity_hint: Optional[Dict[str, float]] = None,
    rho_hint: Optional[Dict[str, float]] = None,
    classical_s_hint: Optional[Dict[str, float]] = None,
) -> PhysicalPlan:
    """Map a (rewritten) logical plan to a physical plan tree.

    Parameters
    ----------
    plan : Plan
        The logical plan (output of `rbo.rewrite`).
    store : TableStore, optional
        The substrate; required for Scan nodes. If None, a placeholder
        is used (useful for printing the plan).
    selectivity_hint : dict, optional
        Optional map of {vector_column: estimated_selectivity} used
        by the CBO when picking an ANN strategy.
    rho_hint : dict, optional
        Optional map of {vector_column: rho_f} for the HNSW
        connectivity ratio of the surviving subset (Section 6.2.2).
        Missing keys default to 1.0 (no fragmentation penalty).
    classical_s_hint : dict, optional
        Optional map of {classical_column: selectivity} for the
        upstream classical filters, used to compute the effective
        `n_input` for the vector scan. Missing keys default to 1.0
        (no classical filter).
    """
    selectivity_hint = selectivity_hint or {}
    rho_hint = rho_hint or {}
    classical_s_hint = classical_s_hint or {}

    if isinstance(plan, Scan):
        return ScanP(store=store, name=plan.name)

    if isinstance(plan, Project):
        return to_physical(plan.child, store, selectivity_hint, rho_hint, classical_s_hint)

    if isinstance(plan, Select):
        phys_child = to_physical(plan.child, store, selectivity_hint, rho_hint, classical_s_hint)
        # If the predicate is a vector range, the CBO picks a strategy.
        vec_strategy = None
        vec_threshold = None
        pred = plan.predicate
        if isinstance(pred, VectorRange):
            s = selectivity_hint.get(pred.column, 0.5)
            n = store.arrow.num_rows if store is not None else 10000
            # Read dim cheaply without materializing the full (n, d)
            # array — the previous version of this line did a
            # ``store.vector_column(...)`` call per VectorRange, which
            # is O(n) on each invocation. For n=1M d=128 that's a
            # 488 MB Arrow→numpy conversion paid per query (the
            # dominant cost measured on the n=1M ECS sweep).
            dim = _vector_dim_fast(store, pred.column)
            rho = rho_hint.get(pred.column, 1.0)
            # Effective n_input is the surviving rows after the
            # upstream classical filter; if the column-specific
            # classical_s is not given, default to 1.0 (no filter).
            classical_s = 1.0
            for ccol, cs in classical_s_hint.items():
                # We don't know which classical column corresponds
                # to this vector column's filter; the caller should
                # pass a single classical_s value or one per column.
                classical_s = min(classical_s, cs)
            from .cbo import cbo_pick
            # Planner side has no IndexManager; pass empty set so CBO
            # only considers ``"flat"``. The runtime side (MosaicOptSystem)
            # rebuilds the strategy selection with the IndexManager's
            # verdict after the plan is materialized.
            vec_strategy = cbo_pick(s, n, dim, indexes_available=(),
                                    rho=rho, classical_s=classical_s)
            vec_threshold = pred.threshold
        elif isinstance(pred, And):
            # If a conjunct is a VectorRange, pick for that.
            for sub in [pred.left, pred.right]:
                if isinstance(sub, VectorRange):
                    s = selectivity_hint.get(sub.column, 0.5)
                    n = store.arrow.num_rows if store is not None else 10000
                    # Same dim-fast path; see comment above.
                    dim = _vector_dim_fast(store, sub.column)
                    rho = rho_hint.get(sub.column, 1.0)
                    classical_s = 1.0
                    for cs in classical_s_hint.values():
                        classical_s = min(classical_s, cs)
                    from .cbo import cbo_pick
                    vec_strategy = cbo_pick(s, n, dim, indexes_available=(),
                                            rho=rho, classical_s=classical_s)
                    vec_threshold = sub.threshold
                    break
        return SelectP(
            child=phys_child,
            predicate=plan.predicate,
            vec_strategy=vec_strategy,
            vec_threshold=vec_threshold,
        )

    if isinstance(plan, Materialize):
        return MaterializeP(
            child=to_physical(plan.child, store, selectivity_hint, rho_hint, classical_s_hint),
            src_col=plan.src_col,
            expr=plan.expr,
            alias=plan.alias,
        )

    if isinstance(plan, Order):
        return Sort(
            child=to_physical(plan.child, store, selectivity_hint, rho_hint, classical_s_hint),
            key_col=plan.key_col,
            key_expr=plan.key_expr,
            ascending=plan.ascending,
        )

    if isinstance(plan, Limit):
        # If child is Order, the canonical (Tau, Lambda) pair -> Limit over Sort
        return LimitP(
            child=to_physical(plan.child, store, selectivity_hint, rho_hint, classical_s_hint),
            k=plan.k,
        )

    if isinstance(plan, _TopKFlag):
        return TopK(
            child=to_physical(plan.child, store, selectivity_hint, rho_hint, classical_s_hint),
            key_col=plan.key_col,
            key_expr=plan.key_expr,
            ascending=plan.ascending,
            k=plan.k,
        )

    if isinstance(plan, Expand):
        return ExpandP(
            child=to_physical(plan.child, store, selectivity_hint, rho_hint, classical_s_hint),
            seed_col=plan.seed_col,
            rule_name=plan.rule_name,
            depth=plan.depth,
        )

    if isinstance(plan, Join):
        return JoinP(
            left=to_physical(plan.left, store, selectivity_hint, rho_hint, classical_s_hint),
            right=to_physical(plan.right, store, selectivity_hint, rho_hint, classical_s_hint),
            condition=plan.condition,
        )

    raise ValueError(f"unknown plan node: {type(plan).__name__}")
