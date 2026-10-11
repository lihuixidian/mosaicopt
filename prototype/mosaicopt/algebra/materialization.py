"""Materialization expression language.

A `MaterializeExpr` is what `Mu_{e->c}(R)` computes per row. The
expression language mirrors Section 3.3.1 of algebra.md:
    <mexpr> ::= <const> | <ident> | ||<ident> - <qparam>||
              |  BM25(<ident>, <qparam>) | (<mexpr>) | <mexpr> <arithop> <mexpr>
              |  min(<mexpr>, <mexpr>) | max(<mexpr>, <mexpr>)

The implementation is a small algebraic data type. Each node carries:
    * the textual form (for printing);
    * the set of relation attributes it references (for free_attrs
      computation in M3 and for the RBO);
    * a `compile` method returning a callable that takes a dict
      {col_name: value, qparam_name: value} and returns the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, Optional, Union

import numpy as np


# ---------------------------------------------------------------------
# MaterializeExpr base
# ---------------------------------------------------------------------

class MaterializeExpr:
    """A scalar expression evaluable per row of a relation.

    Used by the Materialize operator (algebra.ops.Materialize).
    """
    def free_attrs(self) -> FrozenSet[str]:
        return frozenset()

    def uses_param(self, name: str) -> bool:
        return False

    def compile(self, query_params: Dict[str, Any]) -> Callable[[Dict[str, Any]], Any]:
        """Return a function that maps a row-dict to a value.

        The function should be cheap to call per row; physical
        implementations may batch rows and call numpy operations
        on the columns directly (see physical/ann.py).
        """
        raise NotImplementedError


# ---------------------------------------------------------------------
# Atomic expression forms
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _Const(MaterializeExpr):
    value: Any

    def compile(self, qp):
        v = self.value
        def fn(row):
            return v
        return fn


@dataclass(frozen=True)
class _Attr(MaterializeExpr):
    column: str

    def free_attrs(self) -> FrozenSet[str]:
        return frozenset({self.column})

    def compile(self, qp):
        col = self.column
        def fn(row):
            return row[col]
        return fn


@dataclass(frozen=True)
class _Distance(MaterializeExpr):
    """||V - q|| where V is a vector column and q is a query-bound vector."""
    column: str
    q_param: str
    metric: str = "l2"   # "l2" | "cosine" | "ip"

    def free_attrs(self) -> FrozenSet[str]:
        return frozenset({self.column})

    def uses_param(self, name: str) -> bool:
        return self.q_param == name

    def compile(self, qp):
        col = self.column
        q = np.asarray(qp[self.q_param], dtype="float32")
        metric = self.metric
        if metric == "l2":
            def fn(row):
                v = np.asarray(row[col], dtype="float32")
                return float(np.linalg.norm(v - q))
        elif metric == "cosine":
            def fn(row):
                v = np.asarray(row[col], dtype="float32")
                nv = np.linalg.norm(v) + 1e-12
                nq = np.linalg.norm(q) + 1e-12
                return float(1.0 - float(np.dot(v, q)) / (nv * nq))
        elif metric == "ip":
            def fn(row):
                v = np.asarray(row[col], dtype="float32")
                return float(-float(np.dot(v, q)))   # negative inner product as "distance"
        else:
            raise ValueError(f"unknown metric: {metric}")
        return fn


@dataclass(frozen=True)
class _BM25(MaterializeExpr):
    column: str
    q_param: str

    def free_attrs(self) -> FrozenSet[str]:
        return frozenset({self.column})

    def uses_param(self, name: str) -> bool:
        return self.q_param == name

    def compile(self, qp):
        # The actual BM25 score is computed at the physical layer; the
        # algebra expression only carries the binding.
        col = self.column
        q = qp[self.q_param]
        score_attr = f"__bm25_{self.q_param}_{col}"
        def fn(row):
            return row[score_attr]
        return fn


@dataclass(frozen=True)
class _Arith(MaterializeExpr):
    op: str              # "+" | "-" | "*" | "/"
    left: MaterializeExpr
    right: MaterializeExpr

    def free_attrs(self) -> FrozenSet[str]:
        return self.left.free_attrs() | self.right.free_attrs()

    def uses_param(self, name: str) -> bool:
        return self.left.uses_param(name) or self.right.uses_param(name)

    def compile(self, qp):
        lf = self.left.compile(qp)
        rf = self.right.compile(qp)
        op = self.op
        if op == "+":
            def fn(row): return lf(row) + rf(row)
        elif op == "-":
            def fn(row): return lf(row) - rf(row)
        elif op == "*":
            def fn(row): return lf(row) * rf(row)
        elif op == "/":
            def fn(row): return lf(row) / rf(row)
        else:
            raise ValueError(f"unknown arith op: {op}")
        return fn


@dataclass(frozen=True)
class _MinMax(MaterializeExpr):
    op: str              # "min" | "max"
    left: MaterializeExpr
    right: MaterializeExpr

    def free_attrs(self) -> FrozenSet[str]:
        return self.left.free_attrs() | self.right.free_attrs()

    def compile(self, qp):
        lf = self.left.compile(qp)
        rf = self.right.compile(qp)
        op = self.op
        if op == "min":
            def fn(row): return min(lf(row), rf(row))
        elif op == "max":
            def fn(row): return max(lf(row), rf(row))
        else:
            raise ValueError(f"unknown minmax op: {op}")
        return fn


# ---------------------------------------------------------------------
# Public constructors (used by SQL frontend and tests)
# ---------------------------------------------------------------------

def const(v: Any) -> MaterializeExpr:
    return _Const(v)

def attr(c: str) -> MaterializeExpr:
    return _Attr(c)

def distance(c: str, q_param: str, metric: str = "l2") -> MaterializeExpr:
    return _Distance(c, q_param, metric)

def bm25_score(c: str, q_param: str) -> MaterializeExpr:
    return _BM25(c, q_param)

def arith(op: str, left: MaterializeExpr, right: MaterializeExpr) -> MaterializeExpr:
    if op in ("min", "max"):
        return _MinMax(op, left, right)
    return _Arith(op, left, right)


# ---------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------

def mexpr_to_string(e: MaterializeExpr) -> str:
    if isinstance(e, _Const):
        return str(e.value)
    if isinstance(e, _Attr):
        return e.column
    if isinstance(e, _Distance):
        return f"||{e.column} - ?{e.q_param}||"
    if isinstance(e, _BM25):
        return f"BM25({e.column}, ?{e.q_param})"
    if isinstance(e, _Arith):
        return f"({mexpr_to_string(e.left)} {e.op} {mexpr_to_string(e.right)})"
    if isinstance(e, _MinMax):
        return f"{e.op}({mexpr_to_string(e.left)}, {mexpr_to_string(e.right)})"
    return f"<expr:{type(e).__name__}>"
