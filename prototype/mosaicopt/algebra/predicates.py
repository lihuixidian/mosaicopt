"""Predicate language for MosaicOpt.

The predicate forms are taken from algebra.md Section 3.3 / 3.4:
    - ColumnPredicate(a, op, c)             classical: a <op> c
    - VectorRange(V, q, tau, op='<')         vector: ||V - q|| < tau
    - BM25Score(text_col, q, theta, op='>')  text: BM25(col, q) > theta
    - GraphReach(src_q, a, depth, rule)     graph: reachable(q, a) in <d> hops
    - And(P, Q), Or(P, Q), Not(P)            boolean combinations

Each predicate carries:
    * a structural equality / hash so the RBO can match it;
    * a `free_attrs()` set listing the relation attributes it references
      (used by J1 to decide whether a selection can be pushed under a join);
    * a `references_param(name)` predicate so the RBO can tell when
      a materialized value (no params) vs a query-bound parameter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, FrozenSet, Optional, Union


# ---------------------------------------------------------------------
# Predicate base
# ---------------------------------------------------------------------

class Predicate:
    """Marker class for all predicate types."""
    def free_attrs(self) -> FrozenSet[str]:
        """Return the set of relation attributes referenced by this predicate.

        Query-bound parameters (those starting with `?`) are NOT returned.
        """
        return frozenset()

    def uses_param(self, param_name: str) -> bool:
        """True iff this predicate references a query-bound parameter."""
        return False


# ---------------------------------------------------------------------
# Atomic predicate types
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class ColumnPredicate(Predicate):
    """Classical predicate: a <op> c (or a in S).

    Attributes
    ----------
    column : str
        Relation attribute name.
    op : str
        One of ``'=','!=','<','<=','>','>='``.
    value : Any
        Either a literal value or a query-bound parameter name (string
        starting with ``?``). For set membership, `op == 'in'` and
        `value` is a frozenset of literals.
    """
    column: str
    op: str
    value: Any

    def free_attrs(self) -> FrozenSet[str]:
        if isinstance(self.value, str) and self.value.startswith("?"):
            return frozenset()
        return frozenset({self.column})

    def uses_param(self, param_name: str) -> bool:
        return self.value == f"?{param_name}"


@dataclass(frozen=True)
class VectorRange(Predicate):
    """Vector range predicate: ||V - q|| <op> tau.

    `V` is a vector-typed attribute of the relation; `q` is a
    query-bound vector (name `?q_param`); `tau` is a literal threshold
    or a query-bound parameter.
    """
    column: str                # relation attribute (e.g. "emb")
    q_param: str               # name of the query-bound vector (e.g. "q1")
    threshold: float
    op: str = "<"
    metric: str = "l2"         # "l2" or "cosine"

    def free_attrs(self) -> FrozenSet[str]:
        return frozenset({self.column})

    def uses_param(self, param_name: str) -> bool:
        return self.q_param == param_name


@dataclass(frozen=True)
class BM25Score(Predicate):
    """Text relevance predicate: BM25(text_col, q) >op theta."""
    column: str
    q_param: str
    threshold: float
    op: str = ">"             # typically ">" (we want high relevance)

    def free_attrs(self) -> FrozenSet[str]:
        return frozenset({self.column})

    def uses_param(self, param_name: str) -> bool:
        return self.q_param == param_name


@dataclass(frozen=True)
class GraphReach(Predicate):
    """Graph reachability: reachable(?n, a) within <depth> hops via <rule>.

    Used in the predicate context; for *output* expansion, see
    algebra.ops.Expand.
    """
    q_param: str               # start node id
    column: str                # column in current row
    depth: int
    rule_name: str

    def free_attrs(self) -> FrozenSet[str]:
        return frozenset({self.column})

    def uses_param(self, param_name: str) -> bool:
        return self.q_param == param_name


# ---------------------------------------------------------------------
# Boolean combinations
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class And(Predicate):
    left: Predicate
    right: Predicate

    def free_attrs(self) -> FrozenSet[str]:
        return self.left.free_attrs() | self.right.free_attrs()


@dataclass(frozen=True)
class Or(Predicate):
    left: Predicate
    right: Predicate

    def free_attrs(self) -> FrozenSet[str]:
        return self.left.free_attrs() | self.right.free_attrs()


@dataclass(frozen=True)
class Not(Predicate):
    child: Predicate

    def free_attrs(self) -> FrozenSet[str]:
        return self.child.free_attrs()


def and_pred(*ps: Predicate) -> Predicate:
    """Build an And-tree from a list of predicates (flattened)."""
    flat: list[Predicate] = []
    for p in ps:
        if isinstance(p, And):
            flat.extend([p.left, p.right])
        else:
            flat.append(p)
    if not flat:
        raise ValueError("and_pred requires at least one predicate")
    out = flat[0]
    for p in flat[1:]:
        out = And(out, p)
    return out


def or_pred(*ps: Predicate) -> Predicate:
    flat: list[Predicate] = []
    for p in ps:
        if isinstance(p, Or):
            flat.extend([p.left, p.right])
        else:
            flat.append(p)
    out = flat[0]
    for p in flat[1:]:
        out = Or(out, p)
    return out


def not_pred(p: Predicate) -> Predicate:
    if isinstance(p, Not):
        return p.child
    return Not(p)


def free_attrs(p: Predicate) -> FrozenSet[str]:
    return p.free_attrs()


# ---------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------

def pred_to_string(p: Predicate) -> str:
    if isinstance(p, ColumnPredicate):
        v = p.value
        if isinstance(v, frozenset):
            v = "{" + ", ".join(sorted(map(str, v))) + "}"
        return f"{p.column} {p.op} {v}"
    if isinstance(p, VectorRange):
        return f"||{p.column} - ?{p.q_param}|| {p.op} {p.threshold}"
    if isinstance(p, BM25Score):
        return f"BM25({p.column}, ?{p.q_param}) {p.op} {p.threshold}"
    if isinstance(p, GraphReach):
        return f"reachable(?{p.q_param}, {p.column}) depth<={p.depth} via '{p.rule_name}'"
    if isinstance(p, And):
        return f"({pred_to_string(p.left)} AND {pred_to_string(p.right)})"
    if isinstance(p, Or):
        return f"({pred_to_string(p.left)} OR {pred_to_string(p.right)})"
    if isinstance(p, Not):
        return f"NOT ({pred_to_string(p.child)})"
    return f"<pred:{type(p).__name__}>"
