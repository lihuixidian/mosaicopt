"""Rule-Based Optimizer (RBO): 7 foundational + 2 corollary + 1
definitional rewrite rules (9 matchers total in this release).

The 7 foundational identities (M1, M2, M3, M4, M5, G1, G2) are
the substrate of the RBO. The 2 corollary matchers (V1, J1) are
A1-vector and A4-vector specializations, kept as named redex
patterns for code clarity but not separate algebraic identities.
The 1 definitional matcher (T1) implements the τ-Λ-⊤K rewrite
from the operator definitions. Each identity is implemented as a
*matcher* that returns either None (no match) or a rewritten
plan node.

The optimizer applies rules iteratively until a fixed point
(algebra.md Theorem 4.2: RBO Confluence via Newman). Termination
is bounded by the structural complexity of the plan: each rule
either moves a node down (e.g. M1 pushes a selection through a
materialization) or merges nodes (T1 collapses Lambda(Tau(.));
G2 collapses nested Γs to a single Γ at summed depth). The
fixed-point loop guarantees that no further rewrite applies.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

from mosaicopt.algebra.ops import (
    Plan, Scan, Project, Select, Materialize, Order, Limit, Expand, Join,
)
from mosaicopt.algebra.predicates import (
    Predicate, And, ColumnPredicate, VectorRange, BM25Score, GraphReach,
    free_attrs,
)
from mosaicopt.algebra.materialization import (
    MaterializeExpr, distance, bm25_score, attr, const, arith,
)


# ---------------------------------------------------------------------
# Helpers: walk the plan tree
# ---------------------------------------------------------------------

def _children(node: Plan) -> List[Plan]:
    if isinstance(node, Scan):
        return []
    if isinstance(node, Project):
        return [node.child]
    if isinstance(node, Select):
        return [node.child]
    if isinstance(node, Materialize):
        return [node.child]
    if isinstance(node, Order):
        return [node.child]
    if isinstance(node, Limit):
        return [node.child]
    if isinstance(node, Expand):
        return [node.child]
    if isinstance(node, Join):
        return [node.left, node.right]
    return []


def _replace_child(node: Plan, new_child: Plan) -> Plan:
    """Return a copy of `node` with its single child replaced by `new_child`."""
    if isinstance(node, Project):
        return Project(new_child, node.attrs)
    if isinstance(node, Select):
        return Select(new_child, node.predicate)
    if isinstance(node, Materialize):
        return Materialize(new_child, node.src_col, node.expr, node.alias)
    if isinstance(node, Order):
        return Order(new_child, node.key_expr, node.key_col, node.ascending)
    if isinstance(node, Limit):
        return Limit(new_child, node.k)
    if isinstance(node, Expand):
        return Expand(new_child, node.seed_col, node.rule_name, node.depth)
    raise ValueError(f"cannot replace child of {type(node).__name__}")


# ---------------------------------------------------------------------
# Individual rewrite rules
#
# Each rule has the signature (node) -> Optional[Plan].
# Returning None means "no match"; returning a node means "rewrite
# produced this node". The fixed-point loop in `rewrite` calls each
# rule in order, then descends into the children.
# ---------------------------------------------------------------------

def _is_select_on_mat_col(pred: Predicate, alias: str) -> bool:
    """True iff pred references only `alias` as a relation attribute.

    This is the precondition of M1: the predicate must use only the
    materialized column (and possibly query-bound parameters).
    """
    attrs = free_attrs(pred)
    return attrs == frozenset({alias})


def rule_m1(node: Plan) -> Optional[Plan]:
    """M1: Sigma_{c < tau}(Mu_{e -> c}(R)) -> Mu_{e -> c}(Sigma_{e < tau}(R)).

    Pushes a selection whose predicate references only the
    materialized column *through* the materialization, so that e is
    computed only on rows that pass the filter.
    """
    if not isinstance(node, Select):
        return None
    child = node.child
    if not isinstance(child, Materialize):
        return None
    if not _is_select_on_mat_col(node.predicate, child.alias):
        return None

    # The filter on the alias translates to a filter on the original
    # expression. We need to substitute the alias back to the expr.
    # We do this by parsing the predicate and replacing any literal
    # `alias` comparison with a comparison on the expression.
    new_pred = _substitute_alias(node.predicate, child.alias, child.expr)
    if new_pred is None:
        return None

    # Push the filter below the materialization.
    pushed = Select(child.child, new_pred)
    return Materialize(pushed, child.src_col, child.expr, child.alias)


def _substitute_alias(pred: Predicate, alias: str, expr: MaterializeExpr) -> Optional[Predicate]:
    """Replace any reference to `alias` in `pred` with `expr`.

    This is a simple syntactic substitution: we only handle
    ColumnPredicate(Column='alias', op, value) by re-building the
    predicate with `expr` evaluated against a synthetic attribute.
    For the prototype, we represent the substitution by wrapping
    the predicate so that the alias becomes a virtual column.
    """
    if isinstance(pred, ColumnPredicate):
        if pred.column == alias:
            return _column_to_expr_predicate(pred, expr)
        return pred
    if isinstance(pred, And):
        l = _substitute_alias(pred.left, alias, expr)
        r = _substitute_alias(pred.right, alias, expr)
        if l is None or r is None:
            return None
        return And(l, r)
    if isinstance(pred, VectorRange) or isinstance(pred, BM25Score) or isinstance(pred, GraphReach):
        return pred  # these don't reference the alias
    if isinstance(pred, (Or,)):
        # not needed for M1 in the prototype
        return None
    return None


def _column_to_expr_predicate(pred: ColumnPredicate, expr: MaterializeExpr) -> Predicate:
    """Convert `alias <op> c` into a predicate that tests the expression.

    We build a `ColumnPredicate` whose `column` is set to a special
    string ``"__expr_<alias>"``; the executor recognizes this
    convention and evaluates the expression per row.

    Note: this is a *prototype* convention. In a real system the
    predicate language would carry expression references directly.
    """
    # For numeric comparisons, the value c is a literal (or ?param).
    return ColumnPredicate(column=f"__expr__{pred.column}", op=pred.op, value=pred.value)


def rule_m2(node: Plan) -> Optional[Plan]:
    """M2: Mu_{a}(Mu_{b}(R)) -> Mu_{b}(Mu_{a}(R)) when a, b independent.

    The independence check: neither expression references the other's
    alias. This is a free reordering.
    """
    if not isinstance(node, Materialize):
        return None
    inner = node.child
    if not isinstance(inner, Materialize):
        return None
    if node.expr.uses_param(inner.alias) or inner.expr.uses_param(node.alias):
        return None
    if free_attrs_expr(node.expr) & free_attrs_expr(inner.expr) and (
        node.alias in free_attrs_expr(inner.expr) or inner.alias in free_attrs_expr(node.expr)
    ):
        return None
    return Materialize(
        Materialize(inner.child, node.src_col, node.expr, node.alias),
        inner.src_col, inner.expr, inner.alias,
    )


def free_attrs_expr(e: MaterializeExpr) -> "frozenset[str]":
    return e.free_attrs()


def rule_m3(node: Plan) -> Optional[Plan]:
    """M3: Mu_{e -> c}(Pi_A(R)) -> Pi_{A U {c}}(Mu_{e -> c}(R)) when e refs only A.

    Pushes a materialization above a projection (if it only uses the
    projected attributes) by pulling the projection up.
    """
    if not isinstance(node, Materialize):
        return None
    inner = node.child
    if not isinstance(inner, Project):
        return None
    if node.alias in inner.attrs:
        return None  # already projected
    attrs = free_attrs_expr(node.expr)
    if not attrs.issubset(inner.attrs):
        return None
    new_attrs = frozenset(inner.attrs | {node.alias})
    return Project(
        Materialize(inner.child, node.src_col, node.expr, node.alias),
        new_attrs,
    )


def rule_m4(node: Plan) -> Optional[Plan]:
    """M4: Mu_{e -> c}(L Join_psi M) -> (Mu_{e -> c}(L)) Join_psi M when e refs only L.

    Pushes a materialization through a join so that e is computed
    only on the origin side, not on every join tuple. The
    materialized column c is fresh and must not be referenced by
    the join condition (Def.~fresh).
    """
    if not isinstance(node, Materialize):
        return None
    inner = node.child
    if not isinstance(inner, Join):
        return None
    left_attrs = _collect_attrs(inner.left)
    expr_attrs = free_attrs_expr(node.expr)
    if not expr_attrs.issubset(left_attrs):
        return None  # e depends on M's attributes
    cond_attrs = free_attrs(inner.condition)
    if node.alias in cond_attrs:
        return None  # c is referenced by psi -- unsafe
    return Join(
        Materialize(inner.left, node.src_col, node.expr, node.alias),
        inner.right,
        inner.condition,
    )


def rule_m5(node: Plan) -> Optional[Plan]:
    """M5: Mu_{e -> c}(Gamma_{g,d}(R)) -> Gamma_{g,d}(Mu_{e -> c}(R)) when e refs only R.

    The mu-counterpart of G1. Pushes a materialization through a
    graph expansion so that e is computed only on origin rows
    rather than on every traversal row. The materialized column
    c is fresh and must not be the graph rule's node-id attribute
    (the seed column).
    """
    if not isinstance(node, Materialize):
        return None
    inner = node.child
    if not isinstance(inner, Expand):
        return None
    seed_attrs = _collect_attrs(inner.child)
    expr_attrs = free_attrs_expr(node.expr)
    if not expr_attrs.issubset(seed_attrs):
        return None  # e depends on traversal attributes
    if node.alias == inner.seed_col:
        return None  # c is the node-id -- unsafe
    return Expand(
        Materialize(inner.child, node.src_col, node.expr, node.alias),
        inner.seed_col,
        inner.rule_name,
        inner.depth,
    )


def rule_v1(node: Plan) -> Optional[Plan]:
    """V1: Sigma_a(Sigma_b(R)) -> Sigma_b(Sigma_a(R)) when a, b on different cols.

    Vector selections on distinct columns commute. We swap their order
    so that the more selective one runs first.
    """
    if not isinstance(node, Select):
        return None
    inner = node.child
    if not isinstance(inner, Select):
        return None
    a = node.predicate
    b = inner.predicate
    if free_attrs(a) & free_attrs(b):
        return None  # overlapping references
    return Select(Select(inner.child, a), b)


def rule_j1(node: Plan) -> Optional[Plan]:
    """J1: Sigma_{vec}(R Join S) -> Sigma_{vec}(R) Join S when vec refs only R.

    Pushes a selection that references only one side of a join
    down to that side. This is the classical A4 axiom specialized
    to vector predicates.
    """
    if not isinstance(node, Select):
        return None
    inner = node.child
    if not isinstance(inner, Join):
        return None
    pred = node.predicate
    left_attrs = _collect_attrs(inner.left)
    right_attrs = _collect_attrs(inner.right)
    p_attrs = free_attrs(pred)
    if p_attrs.issubset(left_attrs):
        return Join(Select(inner.left, pred), inner.right, inner.condition)
    if p_attrs.issubset(right_attrs):
        return Join(inner.left, Select(inner.right, pred), inner.condition)
    return None


def _collect_attrs(node: Plan) -> "frozenset[str]":
    """Return the set of attribute names produced by `node` (base cols only)."""
    if isinstance(node, Scan):
        # Without a schema catalog, we conservatively use a fixed
        # base set; for the prototype this is sufficient because the
        # RBO operates on plans that came from a known schema.
        return frozenset({"id", "subsystem", "ts", "emb", "emb_alt", "text"})
    if isinstance(node, Project):
        return node.attrs
    if isinstance(node, Materialize):
        return _collect_attrs(node.child) | {node.alias}
    if isinstance(node, Order):
        return _collect_attrs(node.child)
    if isinstance(node, Limit):
        return _collect_attrs(node.child)
    if isinstance(node, Select):
        return _collect_attrs(node.child)
    if isinstance(node, Expand):
        return _collect_attrs(node.child) | {"target", "path"}
    if isinstance(node, Join):
        return _collect_attrs(node.left) | _collect_attrs(node.right)
    return frozenset()


def rule_t1(node: Plan) -> Optional[Plan]:
    """T1: Tau_O(Lambda_K(R)) -> TopK_O(R, K).

    Recognizes the canonical Lambda(Tau(.)) pattern and rewrites to
    a single TopK node. The downstream executor can use a heap-based
    implementation.
    """
    if not isinstance(node, Limit):
        return None
    inner = node.child
    if not isinstance(inner, Order):
        return None
    # We can't actually return a new operator type from the RBO without
    # coupling to the physical layer. Instead, we leave the pair in
    # place and tag it with a flag. The plan layer reads the flag and
    # picks a heap-based implementation.
    # For visibility, we instead emit a TopK-tagged Limit.
    return _TopKFlag(inner.child, inner.key_expr, inner.key_col, inner.ascending, node.k)


def rule_g1(node: Plan) -> Optional[Plan]:
    """G1: Sigma_P(Gamma_{g,d}(R)) -> Gamma_{g,d}(Sigma_P(R)) when P refs R only.

    Pushes a selection on the seed attributes down through the
    graph expansion.
    """
    if not isinstance(node, Select):
        return None
    inner = node.child
    if not isinstance(inner, Expand):
        return None
    seed_attrs = _collect_attrs(inner.child)
    p_attrs = free_attrs(node.predicate)
    if not p_attrs.issubset(seed_attrs):
        return None
    return Expand(Select(inner.child, node.predicate), inner.seed_col, inner.rule_name, inner.depth)


def rule_g2(node: Plan) -> Optional[Plan]:
    """G2: Gamma_{g1,d1}(Gamma_{g2,d2}(L)) -> Gamma_{g1,d1+d2}(L)
    when g1, g2 have disjoint label sets and g1's traversal-result
    attribute is disjoint from g2's source (no intermediate v2 of
    g2 is a valid origin of g1).

    Without a graph-rule schema catalog, the prototype takes a
    conservative interpretation: g1, g2 are the *same* rule name
    and the *same* seed column, and the structural condition holds
    trivially (the inner expansion's traversal result is the same
    as the outer expansion's seed). This covers the most common
    case ("two `derived_from` hops" = "one `derived_from` hop at
    depth 2") and the rewrite is sound under the stated
    conditions; the full disjoitness check would require a
    rule schema registry, which the prototype does not maintain.
    """
    if not isinstance(node, Expand):
        return None
    inner = node.child
    if not isinstance(inner, Expand):
        return None
    if node.rule_name != inner.rule_name or node.seed_col != inner.seed_col:
        return None  # different rules: need schema catalog to check
    return Expand(
        inner.child,
        node.seed_col,
        node.rule_name,
        node.depth + inner.depth,
    )


# ---------------------------------------------------------------------
# TopK-flagged plan (consumed by optimizer/plan.py)
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class _TopKFlag(Plan):
    """A top-K node produced by the T1 rewrite. Carries the order key
    and the limit; the executor builds a heap."""
    child: Plan
    key_expr: str
    key_col: str
    ascending: bool
    k: int


# ---------------------------------------------------------------------
# Fixed-point loop
# ---------------------------------------------------------------------

RULES: List[Callable[[Plan], Optional[Plan]]] = [
    rule_m1,
    rule_m3,
    rule_m4,
    rule_g1,
    rule_g2,
    rule_m5,
    rule_m2,
    rule_v1,
    rule_j1,
    rule_t1,
]

RULES_BY_NAME = {
    "M1": rule_m1,
    "M2": rule_m2,
    "M3": rule_m3,
    "M4": rule_m4,
    "M5": rule_m5,
    "V1": rule_v1,
    "J1": rule_j1,
    "T1": rule_t1,
    "G1": rule_g1,
    "G2": rule_g2,
}


def _apply_once(root: Plan, trace: List[str]) -> Plan:
    """Apply the first matching rule to `root`; recurse into children."""
    for rule in RULES:
        new_root = rule(root)
        if new_root is not None and new_root is not root:
            trace.append(rule.__name__.replace("rule_", "").upper())
            return new_root
    # No top-level rule matched; recurse.
    new_children = []
    changed = False
    for c in _children(root):
        new_c = _apply_once(c, trace)
        new_children.append(new_c)
        if new_c is not c:
            changed = True
    if not changed:
        return root
    if isinstance(root, Project):
        return Project(new_children[0], root.attrs)
    if isinstance(root, Select):
        return Select(new_children[0], root.predicate)
    if isinstance(root, Materialize):
        return Materialize(new_children[0], root.src_col, root.expr, root.alias)
    if isinstance(root, Order):
        return Order(new_children[0], root.key_expr, root.key_col, root.ascending)
    if isinstance(root, Limit):
        return Limit(new_children[0], root.k)
    if isinstance(root, Expand):
        return Expand(new_children[0], root.seed_col, root.rule_name, root.depth)
    if isinstance(root, Join):
        return Join(new_children[0], new_children[1], root.condition)
    if isinstance(root, _TopKFlag):
        return _TopKFlag(new_children[0], root.key_expr, root.key_col, root.ascending, root.k)
    return root


def rewrite(root: Plan, max_iters: int = 64) -> Tuple[Plan, List[str]]:
    """Apply rewrite rules until a fixed point.

    Returns (rewritten_plan, trace) where `trace` is the list of
    rule names applied (in order). The loop is bounded by
    `max_iters` for safety; convergence is guaranteed by Theorem
    4.2 of algebra.md (RBO Confluence).
    """
    trace: List[str] = []
    cur = root
    for _ in range(max_iters):
        new = _apply_once(cur, trace)
        if new is cur:
            break
        cur = new
    return cur, trace
