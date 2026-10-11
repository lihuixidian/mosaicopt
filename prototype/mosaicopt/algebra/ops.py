"""Logical plan tree for the MosaicOpt extended algebra.

The plan tree follows the BNF in docs/algebra.md Section 3.3:
    R    -> Scan(R) (base relation)
    R    -> Sigma_phi(R)
    R    -> Pi_A(R)
    R    -> Mu_{e->c}(R)
    R    -> Tau_O(R)
    R    -> Lambda_K(R)
    R    -> Gamma_{g,d}(R)
    R    -> R Join_psi R

Each node carries enough information to be:
  * printed as a tree (plan_to_string);
  * matched by the RBO (optimizer/rbo.py);
  * mapped to a physical plan (optimizer/plan.py).

The tree is immutable in the sense that the RBO builds *new* nodes
when it applies a rewrite (structural sharing with the original);
this matches the standard algebraic approach.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

# We keep these forward references as strings to avoid an import cycle.
PredicateRef = Any
MaterializeExprRef = Any


@dataclass(frozen=True)
class Plan:
    """Base class for all logical plan nodes.

    All concrete nodes inherit from this class. We use frozen dataclasses
    so that plan nodes can be hashed and used as dict keys in the
    memoized RBO. Mutability is unnecessary: rewrites produce *new* nodes.
    """
    op: str = field(init=False, default="?")

    def __post_init__(self) -> None:
        # frozen dataclasses need object.__setattr__ for op
        object.__setattr__(self, "op", type(self).__name__)


@dataclass(frozen=True)
class Scan(Plan):
    """Base relation: a stored relation identified by name.

    Attributes
    ----------
    name : str
        The relation name (e.g. ``"memory"``).
    alias : Optional[str]
        Optional alias (not used by the RBO but kept for SQL symmetry).
    """
    name: str
    alias: Optional[str] = None


@dataclass(frozen=True)
class Project(Plan):
    """Projection Pi_A(R).

    Keep only the attributes in `attrs` (a frozenset of column names).
    """
    child: Plan
    attrs: frozenset


@dataclass(frozen=True)
class Select(Plan):
    """Selection Sigma_phi(R).

    `predicate` is a Predicate object (see algebra.predicates).
    """
    child: Plan
    predicate: PredicateRef


@dataclass(frozen=True)
class Materialize(Plan):
    """Materialization Mu_{e->c}(R).

    Compute expression `expr` per row of `child` and bind the result
    to a fresh attribute `alias`. The alias is the new column name.

    Following Section 3.4.4 of algebra.md, the multiplicity of each
    input row is preserved exactly once in the output: one output tuple
    per input tuple, with the new column appended.
    """
    child: Plan
    src_col: str           # column whose value is used in the expression (e.g. "emb")
    expr: MaterializeExprRef  # the expression to compute
    alias: str             # the new column name (fresh)


@dataclass(frozen=True)
class Order(Plan):
    """Ordering Tau_O(R).

    `key` is a callable (Plan -> pyarrow.Array) used at execution time
    to extract the sort key. The Plan only carries the *logical*
    ordering intent; physical decisions (topK heap vs full sort) are
    made in optimizer/plan.py based on the presence of a downstream
    Limit node.
    """
    child: Plan
    # We store the textual form for printing; the runtime builds the
    # actual key function by looking at the column produced by a
    # prior Materialize (M1) or by the base relation.
    key_expr: str           # a human-readable key (e.g. "0.7*d1 + 0.3*d2")
    key_col: str            # the primary sort column name (for topK/quick)
    ascending: bool = True


@dataclass(frozen=True)
class Limit(Plan):
    """Limit Lambda_K(R) (top-K prefix).

    In MosaicOpt, the canonical form is Lambda_K(Tau_O(R)), which the
    RBO recognizes as a top-K and rewrites to a TopK node per T1.
    """
    child: Plan
    k: int


@dataclass(frozen=True)
class Expand(Plan):
    """Graph expansion Gamma_{g, d}(R).

    From each seed node (column `seed_col` of `child`), traverse graph
    `rule_name` (e.g. ``"derived_from"``) up to depth `depth`. Output
    has columns of `child` plus `target` (the reachable node) and
    `path` (the sequence of edge labels).
    """
    child: Plan
    seed_col: str           # column in child holding the seed node id
    rule_name: str          # the named graph rule
    depth: int              # max number of edges


@dataclass(frozen=True)
class Join(Plan):
    """Theta-join R Join_psi S.

    `condition` is a Predicate (typically ColumnPredicate comparing
    a column of `left` to a column of `right`).
    """
    left: Plan
    right: Plan
    condition: PredicateRef


# ---------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------

def plan_to_string(node: Plan, indent: int = 0) -> str:
    """Return a human-readable algebraic form of the plan tree.

    Uses the Greek / math notation from algebra.md:
        Sigma, Pi, Mu, Tau, Lambda, Gamma, Join.
    """
    pad = "  " * indent
    if isinstance(node, Scan):
        return f"{pad}{node.name}" + (f" AS {node.alias}" if node.alias else "")
    if isinstance(node, Project):
        attrs = ", ".join(sorted(node.attrs))
        return f"{pad}Pi_{{{attrs}}}(\n{plan_to_string(node.child, indent+1)}\n{pad})"
    if isinstance(node, Select):
        from .predicates import pred_to_string
        return f"{pad}Sigma_{{{pred_to_string(node.predicate)}}}(\n{plan_to_string(node.child, indent+1)}\n{pad})"
    if isinstance(node, Materialize):
        from .materialization import mexpr_to_string
        return f"{pad}Mu_{{{mexpr_to_string(node.expr)}->{node.alias}}}(\n{plan_to_string(node.child, indent+1)}\n{pad})"
    if isinstance(node, Order):
        arrow = "ASC" if node.ascending else "DESC"
        return f"{pad}Tau_{{{node.key_expr} {arrow}}}(\n{plan_to_string(node.child, indent+1)}\n{pad})"
    if isinstance(node, Limit):
        return f"{pad}Lambda_{{{node.k}}}(\n{plan_to_string(node.child, indent+1)}\n{pad})"
    if isinstance(node, Expand):
        return f"{pad}Gamma_{{{node.rule_name}, {node.depth}}}(\n{plan_to_string(node.child, indent+1)}\n{pad})"
    if isinstance(node, Join):
        from .predicates import pred_to_string
        return f"{pad}(\n{plan_to_string(node.left, indent+1)}\n{pad}) Join_{{{pred_to_string(node.condition)}}} (\n{plan_to_string(node.right, indent+1)}\n{pad})"
    if type(node).__name__ == "_TopKFlag":
        arrow = "ASC" if node.ascending else "DESC"
        return f"{pad}TopK_{{{node.key_expr} {arrow}, k={node.k}}}(\n{plan_to_string(node.child, indent+1)}\n{pad})"
    return f"{pad}<unknown:{type(node).__name__}>"
