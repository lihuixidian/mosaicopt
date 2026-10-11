"""SQL/MM-style parser for MosaicOpt.

This is a small, hand-rolled parser that handles the Q1->Q6 example
plus a few variants. It is intentionally minimal -- the goal is to
demonstrate the *desugaring* from SQL surface to algebra, not to be
a general SQL parser. The grammar we support is the dialect from
docs/brief.md Section 10.

The parser is regex-based and operates in three phases:
    1. Extract the CTE (WITH ... AS) clauses into named subqueries.
    2. Extract SELECT / FROM / WHERE / ORDER BY / LIMIT from the body.
    3. Desugar each clause into algebra operators.

The output is a logical plan tree (algebra.ops.Plan).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from mosaicopt.algebra.ops import (
    Plan, Scan, Project, Select, Materialize, Order, Limit, Expand, Join,
)
from mosaicopt.algebra.predicates import (
    Predicate, ColumnPredicate, VectorRange, BM25Score, GraphReach, And, Or, Not,
    and_pred, free_attrs,
)
from mosaicopt.algebra.materialization import (
    MaterializeExpr, const, attr, distance, bm25_score, arith,
)


# ---------------------------------------------------------------------
# Token-level helpers
# ---------------------------------------------------------------------

# A simple tokenizer: words, operators, numbers, strings, parentheses.
_TOKEN_RE = re.compile(
    r"""
    \s+                            |   # whitespace (skipped)
    (?P<number>\d+\.\d+|\d+)       |   # numeric literal
    '(?P<string>(?:[^']|'')*)'     |   # single-quoted string
    "(?P<dstring>(?:[^"]|"")*)"    |   # double-quoted string
    (?P<ident>[A-Za-z_][A-Za-z0-9_]*) | # identifier
    (?P<op>:=|=>|<>|<=|>=|<->|<=>|<=>|<=>|<=>|::|\*|\+|-|/|,|\(|\)|=|<|>)
    """,
    re.VERBOSE,
)


def _tokenize(sql: str) -> List[Tuple[str, str]]:
    """Return a list of (kind, text) tokens. Whitespace tokens are dropped."""
    out: List[Tuple[str, str]] = []
    pos = 0
    while pos < len(sql):
        m = _TOKEN_RE.match(sql, pos)
        if not m:
            raise ValueError(f"tokenizer stuck at: {sql[pos:pos+30]!r}")
        pos = m.end()
        if m.lastgroup is None:
            continue  # whitespace
        out.append((m.lastgroup, m.group(m.lastgroup)))
    return out


# ---------------------------------------------------------------------
# AST nodes (parse tree, *not* algebra)
# ---------------------------------------------------------------------

@dataclass
class _SqlSelect:
    select: List[Tuple[str, str]]            # (expr_text, alias) pairs
    from_: str
    where: Optional[str] = None
    order_by: Optional[str] = None
    order_asc: bool = True
    limit: Optional[int] = None


@dataclass
class _SqlCte:
    name: str
    select: _SqlSelect


# ---------------------------------------------------------------------
# Top-level entry points
# ---------------------------------------------------------------------

def parse_sql(sql: str) -> Tuple[List[_SqlCte], _SqlSelect]:
    """Parse a SQL string into a list of CTEs + the final SELECT.

    Returns ``(ctes, select)``. The select's `from_` may be a CTE
    name or a base relation.
    """
    s = _strip_comments(sql).strip()
    ctes: List[_SqlCte] = []
    body = s

    # WITH ... SELECT handling
    m = re.match(r"(?is)\s*with\s+(recursive\s+)?(.*)", s)
    if m:
        after_with = m.group(2)
        # Greedily parse CTEs of the form "name AS ( SELECT ... )"
        # until we hit a top-level SELECT. We use a balanced-paren
        # scan to find the matching `)` of the CTE's open paren.
        i = 0
        while i < len(after_with):
            mm = re.match(
                r"(?is)\s*(\w+)\s+as\s*\(",
                after_with[i:],
            )
            if not mm:
                break
            cte_name = mm.group(1).strip()
            open_paren_pos = i + mm.end() - 1   # index of '('
            # Find matching ')'
            depth = 0
            j = open_paren_pos
            while j < len(after_with):
                ch = after_with[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            if depth != 0:
                break
            cte_body = after_with[open_paren_pos + 1:j].strip()
            ctes.append(_SqlCte(name=cte_name, select=_parse_select(cte_body)))
            # Skip past `)` and any trailing comma/whitespace
            i = j + 1
            while i < len(after_with) and after_with[i] in " \t\n\r,":
                i += 1
        body = after_with[i:].strip()

    sel = _parse_select(body)
    return ctes, sel


def parse_to_plan(
    sql: str,
    cte_resolvers: Optional[Dict[str, Plan]] = None,
) -> Plan:
    """Parse SQL and desugar to a logical algebra plan.

    Parameters
    ----------
    sql : str
        The SQL text.
    cte_resolvers : dict, optional
        Map of CTE name -> plan to substitute when the FROM clause
        references a CTE. If None, the FROM is treated as a Scan
        (for the inner-CTE case the parser recursively builds plans).
    """
    ctes, sel = parse_sql(sql)
    cte_resolvers = dict(cte_resolvers or {})
    # Build CTE plans first (each CTE may reference prior CTEs)
    for c in ctes:
        cte_resolvers[c.name] = _select_to_plan(c.select, cte_resolvers)
    return _select_to_plan(sel, cte_resolvers)


# ---------------------------------------------------------------------
# SELECT parsing
# ---------------------------------------------------------------------

_SELECT_RE = re.compile(
    r"""
    \s* SELECT \s+
    (?P<select>[\s\S]*?)
    \s+ FROM \s+
    (?P<from>\w+)
    (?P<rest>[\s\S]*)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _parse_select(text: str) -> _SqlSelect:
    text = text.strip().rstrip(";").strip()
    m = _SELECT_RE.match(text)
    if not m:
        raise ValueError(f"unrecognized SELECT: {text[:80]!r}")
    select_text = m.group("select")
    from_ = m.group("from")
    rest = m.group("rest")

    # SELECT clause: split on top-level commas
    items = _split_top_commas(select_text)
    select_items: List[Tuple[str, str]] = []
    for it in items:
        # Match `expr AS alias` or `expr alias`
        am = re.match(r"(?is)\s*(.+?)\s+as\s+(\w+)\s*$", it)
        if am:
            select_items.append((am.group(1).strip(), am.group(2).strip()))
        else:
            # No AS - use the whole item as both expr and a synthesized alias
            select_items.append((it.strip(), ""))

    # WHERE / ORDER BY / LIMIT in rest
    where = None
    order_by = None
    order_asc = True
    limit = None
    s = rest.strip()
    wm = re.search(
        r"(?is)\bwhere\s+(.+?)(?=\s*(?:order\s+by|limit\b|group\s+by|having|$))",
        s,
    )
    if wm:
        where = wm.group(1).strip()
        s = (s[:wm.start()] + s[wm.end():]).strip()
    om = re.search(
        r"(?is)\border\s+by\s+(.+?)(?=\s*(?:limit\b|group\s+by|having|$))",
        s,
    )
    if om:
        order_by = om.group(1).strip()
        if re.search(r"(?i)\s+desc\s*$", order_by):
            order_asc = False
            order_by = re.sub(r"(?i)\s+(asc|desc)\s*$", "", order_by).strip()
        elif re.search(r"(?i)\s+asc\s*$", order_by):
            order_asc = True
            order_by = re.sub(r"(?i)\s+(asc|desc)\s*$", "", order_by).strip()
        s = (s[:om.start()] + s[om.end():]).strip()
    lm = re.search(r"(?is)\blimit\s+(\d+)", s)
    if lm:
        limit = int(lm.group(1))
        s = (s[:lm.start()] + s[lm.end():]).strip()

    return _SqlSelect(
        select=select_items,
        from_=from_,
        where=where,
        order_by=order_by,
        order_asc=order_asc,
        limit=limit,
    )


def _split_top_commas(s: str) -> List[str]:
    """Split a string on top-level commas (ignoring those in parens)."""
    out: List[str] = []
    depth = 0
    cur = []
    for ch in s:
        if ch == "," and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            cur.append(ch)
    if cur:
        rest = "".join(cur).strip()
        if rest:
            out.append(rest)
    return out


def _strip_comments(sql: str) -> str:
    """Remove -- and /* ... */ comments."""
    sql = re.sub(r"/\*.*?\*/", "", sql, flags=re.DOTALL)
    sql = re.sub(r"--[^\n]*", "", sql)
    return sql


# ---------------------------------------------------------------------
# SELECT -> algebra plan
# ---------------------------------------------------------------------

def _select_to_plan(sel: _SqlSelect, cte_resolvers: Dict[str, Plan]) -> Plan:
    """Convert a parsed SELECT into a logical plan."""
    # 1. FROM
    if sel.from_ in cte_resolvers:
        cur = cte_resolvers[sel.from_]
    else:
        cur = Scan(sel.from_)

    # 2. WHERE -> Select
    if sel.where:
        pred = _parse_where(sel.where)
        cur = Select(cur, pred)

    # 3. SELECT clause -> materializations + projection
    # We split SELECT items into:
    #   * "attribute" - column reference (passthrough, in projection)
    #   * "expr AS alias" - materialization
    proj_attrs: List[str] = []
    for expr_text, alias in sel.select:
        if alias == "" and expr_text == "*":
            continue
        if alias == "" and re.match(r"^\w+$", expr_text):
            proj_attrs.append(expr_text)
            continue
        # Treat as a materialization
        if alias == "":
            # synthesize an alias from the expression
            alias = _synthesize_alias(expr_text)
        proj_attrs.append(alias)
        # Parse the expression
        mexpr = _parse_materialize_expr(expr_text)
        # We need the source column - infer from a distance(...) call
        src_col = _infer_source_col(expr_text)
        cur = Materialize(cur, src_col=src_col, expr=mexpr, alias=alias)

    # 4. ORDER BY
    if sel.order_by:
        # We assume ORDER BY references an alias in the SELECT clause
        key_col = _order_by_primary_column(sel.order_by, sel.select)
        cur = Order(cur, key_expr=sel.order_by, key_col=key_col, ascending=sel.order_asc)

    # 5. LIMIT
    if sel.limit is not None:
        cur = Limit(cur, k=sel.limit)

    # 6. Projection (if we collected attributes)
    if proj_attrs:
        cur = Project(cur, attrs=frozenset(proj_attrs))

    return cur


def _parse_where(text: str) -> Predicate:
    """Parse a WHERE clause into a Predicate.

    Supports:
        col = literal
        col < literal | col > literal | col <= literal | col >= literal
        col < literal
        expr AND expr | expr OR expr
    Each operand can be a column ref, a number, a string, or a
    query-bound parameter (":name" or "?name").
    """
    # Normalize params: ":name" -> "?name"
    norm = re.sub(r":(\w+)", r"?\1", text)
    return _parse_predicate_expr(norm)


def _parse_predicate_expr(text: str) -> Predicate:
    """Recursive descent on a simple boolean expression.

    Operators (in precedence order): OR, AND, NOT, comparison.
    Comparisons: =, !=, <, <=, >, >=, <->, <=>
    """
    # Strip outer parens
    text = text.strip()
    if text.startswith("(") and text.endswith(")") and _balanced_parens(text):
        return _parse_predicate_expr(text[1:-1])
    # OR
    parts = _split_top_level(text, " OR ")
    if len(parts) > 1:
        return Or(*[_parse_predicate_expr(p) for p in parts])
    parts = _split_top_level(text, " AND ")
    if len(parts) > 1:
        return And(*[_parse_predicate_expr(p) for p in parts])
    if text.upper().startswith("NOT "):
        return Not(_parse_predicate_expr(text[4:].strip()))
    return _parse_predicate_atom(text)


def _balanced_parens(s: str) -> bool:
    depth = 0
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if depth == 0:
            return False
    return depth == 0


def _split_top_level(s: str, sep: str) -> List[str]:
    """Split `s` on `sep` only at top level (not inside parens)."""
    out = []
    depth = 0
    i = 0
    while i < len(s):
        if s[i] == "(":
            depth += 1
        elif s[i] == ")":
            depth -= 1
        if depth == 0 and s[i:i + len(sep)].lower() == sep.lower():
            out.append(s[:i].strip())
            s = s[i + len(sep):]
            i = 0
            continue
        i += 1
    out.append(s.strip())
    return out


def _parse_predicate_atom(text: str) -> Predicate:
    """Parse a single comparison atom.

    Examples:
        subsystem = 'fault-tolerance'
        ts > ?recent
        d1 < ?tau1
    """
    text = text.strip()
    # Find comparison operator (longest first)
    for op in ["<=", ">=", "<->", "<=>", "!=", "<>", "=", "<", ">"]:
        if op in text:
            left, right = text.split(op, 1)
            left = left.strip()
            right = right.strip()
            col = left if re.match(r"^\w+$", left) else _operand_to_str(left)
            val = _parse_value(right)
            return ColumnPredicate(column=col, op=op, value=val)
    raise ValueError(f"could not parse predicate: {text!r}")


def _operand_to_str(s: str) -> str:
    """Coerce an operand to a string identifier (used for the LHS of comparisons)."""
    return s.strip()


def _parse_value(text: str) -> Any:
    text = text.strip()
    if text.startswith("'") and text.endswith("'"):
        return text[1:-1]
    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]
    if text.startswith("?") or text.startswith(":"):
        return "?" + text.lstrip("?:")
    if re.match(r"^-?\d+\.\d+$", text):
        return float(text)
    if re.match(r"^-?\d+$", text):
        return int(text)
    return text  # bare identifier


# ---------------------------------------------------------------------
# Materialization expression parsing
# ---------------------------------------------------------------------

_DISTANCE_RE = re.compile(r"\(\s*(\w+)\s*<=?>\s*[:?]?(\w+)\s*\)")
_BM25_RE = re.compile(r"\bBM25\s*\(\s*(\w+)\s*,\s*[:?]?(\w+)\s*\)", re.IGNORECASE)


def _parse_materialize_expr(text: str) -> MaterializeExpr:
    """Parse a materialization expression from text.

    Supports:
        (col <=> ?q)                -> distance(col, q)
        BM25(col, ?q)               -> bm25_score(col, q)
        0.7 * d1 + 0.3 * d2         -> arith composition
    """
    text = text.strip()
    # Try distance operator
    m = _DISTANCE_RE.search(text)
    if m:
        col, q = m.group(1), m.group(2)
        return distance(col, q, metric="l2")
    # Try BM25
    m = _BM25_RE.search(text)
    if m:
        col, q = m.group(1), m.group(2)
        return bm25_score(col, q)
    # Bare column reference
    if re.match(r"^\w+$", text):
        return attr(text)
    # Constant
    try:
        return const(float(text))
    except ValueError:
        pass
    # Arithmetic expression: try to parse as a sum/product tree
    return _parse_arith(text)


def _parse_arith(text: str) -> MaterializeExpr:
    """Parse a simple arithmetic expression into MaterializeExpr."""
    # Pre-process: factor out "scalar * ident" and "ident + ident" forms
    # We support +, -, * and parens.
    return _parse_arith_expr(text)


def _parse_arith_expr(text: str) -> MaterializeExpr:
    text = text.strip()
    # Strip outer parens
    if text.startswith("(") and text.endswith(")") and _balanced_parens(text):
        return _parse_arith_expr(text[1:-1])
    # Addition / subtraction (lowest precedence)
    parts = _split_arith(text, "+")
    if len(parts) > 1:
        result = _parse_arith_term(parts[0])
        for p in parts[1:]:
            if p.startswith(" "):
                p = p[1:]
            if p.startswith("-"):
                result = arith("-", result, _parse_arith_term(p[1:]))
            else:
                result = arith("+", result, _parse_arith_term(p))
        return result
    return _parse_arith_term(text)


def _parse_arith_term(text: str) -> MaterializeExpr:
    text = text.strip()
    if text.startswith("(") and text.endswith(")") and _balanced_parens(text):
        return _parse_arith_expr(text[1:-1])
    parts = _split_arith(text, "*")
    if len(parts) > 1:
        result = _parse_arith_factor(parts[0])
        for p in parts[1:]:
            if p.startswith(" "):
                p = p[1:]
            result = arith("*", result, _parse_arith_factor(p))
        return result
    return _parse_arith_factor(text)


def _parse_arith_factor(text: str) -> MaterializeExpr:
    text = text.strip()
    if text.startswith("(") and text.endswith(")") and _balanced_parens(text):
        return _parse_arith_expr(text[1:-1])
    # Number?
    try:
        return const(float(text))
    except ValueError:
        pass
    # Identifier?
    if re.match(r"^\w+$", text):
        return attr(text)
    raise ValueError(f"cannot parse arith factor: {text!r}")


def _split_arith(text: str, op: str) -> List[str]:
    """Split on `op` at top level (not inside parens)."""
    out = []
    depth = 0
    cur = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
        elif depth == 0 and text[i:i + len(op)] == op:
            out.append("".join(cur))
            cur = []
            i += len(op)
            continue
        else:
            cur.append(ch)
        i += 1
    out.append("".join(cur))
    return out


# ---------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------

def _infer_source_col(expr_text: str) -> str:
    """Infer the source column of a materialization expression."""
    m = _DISTANCE_RE.search(expr_text)
    if m:
        return m.group(1)
    m = _BM25_RE.search(expr_text)
    if m:
        return m.group(1)
    # Fallback: look for a bare identifier
    m = re.search(r"\b(\w+)\b", expr_text)
    return m.group(1) if m else ""


def _synthesize_alias(expr_text: str) -> str:
    """Make a short alias from an expression."""
    s = re.sub(r"[^A-Za-z0-9_]", "_", expr_text)[:30]
    return f"col_{s}"


def _order_by_primary_column(text: str, select_items: List[Tuple[str, str]]) -> str:
    """Pick the primary sort column from an ORDER BY expression."""
    # Try each ident in ORDER BY; pick the first that is a SELECT alias
    for tok in re.findall(r"\b(\w+)\b", text):
        for expr, alias in select_items:
            if alias and alias == tok:
                return tok
    # Otherwise, the first ident
    m = re.search(r"\b(\w+)\b", text)
    return m.group(1) if m else ""
