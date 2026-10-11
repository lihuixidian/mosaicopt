"""DuckDB substrate for MosaicOpt.

The substrate holds the base relations as DuckDB tables and exposes
*pyarrow* views for the Python operators (faiss, rank_bm25). The
rationale: DuckDB is the natural SQL surface, and pyarrow is the
fastest interop path to the Python data stack (faiss, scipy).

The substrate is intentionally simple: it does not push down any
algebra work. All multi-modal predicates and materializations run
in Python; DuckDB is used for relational scans, projections, and
joins via its standard columnar engine.

`materialize_column(table, name, expr, alias)` evaluates a Python
expression over a pyarrow table and attaches the result as a new
column. This implements the runtime side of Mu_{e->c}.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import duckdb
import numpy as np
import pyarrow as pa


@dataclass
class TableStore:
    """A base relation backed by a DuckDB table.

    The relation is loaded as a pyarrow table on demand and cached
    for the duration of a query. Vector columns are exposed as
    ``np.ndarray`` of shape ``(n, d)``.
    """
    name: str
    arrow: pa.Table
    _con: Optional[duckdb.DuckDBPyConnection] = field(init=False, default=None, repr=False)

    @classmethod
    def from_arrow(cls, name: str, table: pa.Table) -> "TableStore":
        return cls(name=name, arrow=table)

    @property
    def con(self) -> duckdb.DuckDBPyConnection:
        if self._con is None:
            self._con = duckdb.connect(":memory:")
            self._con.register(self.name, self.arrow)
        return self._con

    def sql(self, query: str, params: Optional[Sequence[Any]] = None) -> pa.Table:
        """Run a SQL query and return a pyarrow table."""
        if params is not None:
            return self.con.execute(query, params).fetch_arrow_table()
        return self.con.execute(query).fetch_arrow_table()

    def filter(self, predicate_sql: str) -> pa.Table:
        """Apply a SQL WHERE clause to the base table."""
        q = f"SELECT * FROM {self.name} WHERE {predicate_sql}"
        return self.sql(q)

    def vector_column(self, col: str) -> np.ndarray:
        """Return a (n, d) float32 numpy array for a vector column.

        Vector columns are stored as ``pa.list_(pa.float32())`` in
        Arrow. This converts them to a contiguous 2-D array.
        """
        arr = self.arrow.column(col).combine_chunks()
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        if pa.types.is_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
            values = arr.values.to_numpy(zero_copy_only=False)
            n = len(arr)
            d = len(values) // n
            return values.reshape(n, d).astype("float32")
        # already a 2-D numpy array stored as a column of length 1?
        return np.asarray(arr.to_numpy(zero_copy_only=False))


def materialize_column(
    table: pa.Table,
    expr: Callable[[Dict[str, Any]], Any],
    alias: str,
    chunk_size: int = 4096,
) -> pa.Table:
    """Compute `expr(row_dict)` for each row of `table` and append as `alias`.

    The result column has the same type as the first value returned
    by `expr` (auto-cast to float64 for safety).
    """
    cols = [table[c] for c in table.column_names]
    names = table.column_names
    n = table.num_rows
    out = np.empty(n, dtype="float64")
    # Build a small row-dict for each row. For 10k rows, this is
    # ~10k dict allocations - acceptable for a prototype. A faster
    # path could pass columnar arrays directly to expr.
    for i in range(n):
        row = {names[j]: cols[j][i].as_py() for j in range(len(names))}
        out[i] = float(expr(row))
    new_col = pa.array(out)
    return table.append_column(alias, new_col)
