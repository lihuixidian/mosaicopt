"""SQLite substrate for MosaicOpt.

Mirrors the DuckDB / PG substrates (``store.py`` / ``store_pg.py``)
so the optimizer / experiments can run unchanged. Same pyarrow
view + vector-column interface.

SQLite is the third substrate in the prototype, alongside
DuckDB (in-process analytics, :memory:) and PostgreSQL
(server-side, full RDBMS). SQLite and DuckDB both run on a
local machine without a server daemon, and SQLite is a true
OLTP-flavored RDBMS (no columnar push-down, single-writer);
the structural claim — algebra + optimizer substrate-agnostic —
holds across all three backends.

The E6 cross-substrate experiment uses SQLite as the local
RDBMS counterpart to DuckDB, since PostgreSQL's EDB-based
installer on Windows is interactive and the runtime for the
paper is too short to wait for a daemon. The PG substrate in
``store_pg.py`` is complete and matches the SQLite / DuckDB
interface; switching to PG is a one-parameter change.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np
import pyarrow as pa


@dataclass
class SqliteTableStore:
    """A base relation backed by a SQLite table.

    Same interface as ``TableStore`` (DuckDB) and
    ``PgTableStore`` (PostgreSQL). Data is loaded via
    parameterized INSERT (vector columns serialized as JSON
    arrays in a TEXT column; the pyarrow view is the source
    of truth for the Python-side operators).
    """
    name: str
    arrow: pa.Table
    db_path: str = ":memory:"
    _con: Any = field(init=False, default=None, repr=False)

    @classmethod
    def from_arrow(cls, name: str, table: pa.Table,
                   db_path: str = ":memory:") -> "SqliteTableStore":
        return cls(name=name, arrow=table, db_path=db_path)

    @property
    def con(self) -> sqlite3.Connection:
        if self._con is None:
            self._con = sqlite3.connect(self.db_path)
            self._con.execute("PRAGMA journal_mode=WAL")
            self._con.execute("PRAGMA synchronous=NORMAL")
            self._load()
        return self._con

    def _load(self) -> None:
        """Bulk-load the pyarrow data into SQLite via INSERT."""
        cols = self.arrow.column_names
        n = self.arrow.num_rows
        type_sqls = []
        for c in cols:
            arr = self.arrow.column(c)
            t = arr.type
            if pa.types.is_list(t) or pa.types.is_fixed_size_list(t):
                type_sqls.append(f'"{c}" TEXT')
            elif pa.types.is_integer(t):
                type_sqls.append(f'"{c}" INTEGER')
            elif pa.types.is_floating(t):
                type_sqls.append(f'"{c}" REAL')
            elif pa.types.is_string(t) or pa.types.is_large_string(t):
                type_sqls.append(f'"{c}" TEXT')
            else:
                type_sqls.append(f'"{c}" TEXT')
        self._con.execute(f'DROP TABLE IF EXISTS "{self.name}"')
        self._con.execute(f'CREATE TABLE "{self.name}" ({", ".join(type_sqls)})')

        # Build insert statement + bulk-execute
        col_list = ", ".join(f'"{c}"' for c in cols)
        placeholders = ", ".join(["?"] * len(cols))
        insert_sql = f'INSERT INTO "{self.name}" ({col_list}) VALUES ({placeholders})'
        # Pre-compute rows in a list-of-tuples for executemany
        rows = []
        for i in range(n):
            row = []
            for c in cols:
                arr = self.arrow.column(c)
                val = arr[i].as_py()
                if isinstance(val, (list, tuple)):
                    # JSON-encode vector
                    import json
                    row.append(json.dumps([float(x) for x in val]))
                elif isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    row.append(None)
                else:
                    row.append(val)
            rows.append(tuple(row))
        self._con.executemany(insert_sql, rows)
        self._con.commit()

    def sql(self, query: str, params: Optional[Sequence[Any]] = None) -> pa.Table:
        """Run a SQL query and return a pyarrow table."""
        cur = self.con.execute(query, params or [])
        desc = [d[0] for d in cur.description]
        rows = cur.fetchall()
        arrays = {desc[i]: pa.array([row[i] for row in rows]) for i in range(len(desc))}
        return pa.table(arrays)

    def vector_column(self, col: str) -> np.ndarray:
        """Return a (n, d) float32 numpy array for a vector column."""
        arr = self.arrow.column(col).combine_chunks()
        if pa.types.is_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
            values = arr.values.to_numpy(zero_copy_only=False)
            n = len(arr)
            d = len(values) // n
            return values.reshape(n, d).astype("float32")
        return np.asarray(arr.to_numpy(zero_copy_only=False))


def make_substrate(kind: str, name: str, table: pa.Table,
                   **kwargs) -> Any:
    """Factory: DuckDB | PG | SQLite store by kind."""
    if kind == "postgres":
        from mosaicopt.physical.store_pg import PgTableStore
        return PgTableStore.from_arrow(name=name, table=table,
                                       dsn=kwargs.get("dsn"))
    if kind == "sqlite":
        return SqliteTableStore.from_arrow(name=name, table=table,
                                           db_path=kwargs.get("db_path", ":memory:"))
    from mosaicopt.physical.store import TableStore as DuckStore
    return DuckStore.from_arrow(name=name, table=table)
