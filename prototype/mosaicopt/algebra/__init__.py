"""MosaicOpt logical algebra.

This package implements the plan tree and predicate language from
docs/algebra.md and docs/brief.md Section 4. The operators are
*logical*: their semantics are bag-interpretation; physical choices
(HNSW vs IVF vs scan) live in `mosaicopt.physical`.

Public surface
--------------
Plan node types:
    Scan, Project, Select, Materialize, Order, Limit, Expand, Join
Predicate types:
    ColumnPredicate, VectorRange, BM25Score, GraphReach
Materialization expression language:
    MaterializeExpr (constant / attribute / distance / score / arith)
"""

from .ops import (
    Plan,
    Scan,
    Project,
    Select,
    Materialize,
    Order,
    Limit,
    Expand,
    Join,
    plan_to_string,
)
from .predicates import (
    Predicate,
    ColumnPredicate,
    VectorRange,
    BM25Score,
    GraphReach,
    and_pred,
    or_pred,
    not_pred,
    free_attrs,
)
from .materialization import (
    MaterializeExpr,
    const,
    attr,
    distance,
    bm25_score,
    arith,
)

__all__ = [
    # ops
    "Plan", "Scan", "Project", "Select", "Materialize",
    "Order", "Limit", "Expand", "Join", "plan_to_string",
    # predicates
    "Predicate", "ColumnPredicate", "VectorRange", "BM25Score", "GraphReach",
    "and_pred", "or_pred", "not_pred", "free_attrs",
    # materialization expressions
    "MaterializeExpr", "const", "attr", "distance", "bm25_score", "arith",
]
