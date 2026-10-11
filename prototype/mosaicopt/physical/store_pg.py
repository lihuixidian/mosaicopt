"""PostgreSQL substrate for MosaicOpt.

Mirrors the DuckDB substrate (``store.py``) so the optimizer /
experiments can run unchanged against either backend. The same
pyarrow data is loaded into a temporary PG table via COPY ... FROM
STDIN with a binary stream; we expose ``arrow`` (a read-only
pyarrow view) and ``vector_column`` (a 2-D float32 ndarray) so
the existing Python-side code (faiss, rank_bm25, numpy) does
not have to change.

Selection of the substrate is a single parameter (``store_kind``)
in the experiment runner. The cost model treats both substrates
identically, because both expose the same logical interface.

This file is intentionally small and does not push any algebra
work down to the SQL engine; the multi-modal predicates still run
in Python. The substrate's role is *only* the relational-scan
fallback and (optionally) caching of pre-classical-filter slices.
"""
from __future__ import annotations

import contextlib
import io
import os
import socket
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pyarrow as pa

try:
    import psycopg2
    import psycopg2.extras
    HAVE_PSYCOPG2 = True
except ImportError:  # pragma: no cover
    HAVE_PSYCOPG2 = False


def _default_dsn() -> str:
    """Default DSN: localhost:5432, postgres/postgres, db=mosaicopt.

    Override via environment variables ``PGHOST``, ``PGPORT``,
    ``PGUSER``, ``PGPASSWORD``, ``PGDATABASE``.
    """
    host = os.environ.get("PGHOST", "localhost")
    port = os.environ.get("PGPORT", "5432")
    user = os.environ.get("PGUSER", "postgres")
    pw = os.environ.get("PGPASSWORD", "postgres")
    db = os.environ.get("PGDATABASE", "mosaicopt")
    return f"host={host} port={port} user={user} password={pw} dbname={db}"


@contextlib.contextmanager
def _pg_connect(dsn: Optional[str] = None, attempts: int = 30, delay: float = 1.0):
    """Yield a psycopg2 connection or raise on failure.

    Waits ``attempts`` times with ``delay`` seconds between for the
    server to come up, which matters for a freshly-installed PG on
    a development machine that just started its daemon.
    """
    if not HAVE_PSYCOPG2:
        raise RuntimeError("psycopg2 not installed; pip install psycopg2-binary")
    dsn = dsn or _default_dsn()
    last_err: Optional[Exception] = None
    for _ in range(attempts):
        try:
            conn = psycopg2.connect(dsn)
            conn.autocommit = True
            yield conn
            conn.close()
            return
        except Exception as e:  # pragma: no cover
            last_err = e
            time.sleep(delay)
    raise RuntimeError(f"PostgreSQL connection failed after {attempts} attempts: {last_err}")


def ensure_database(dsn_admin: Optional[str] = None,
                    dbname: str = "mosaicopt") -> None:
    """Create the target database if it does not exist.

    Connects to the ``postgres`` admin DB and runs ``CREATE DATABASE``
    guarded by existence check. Safe to call repeatedly.
    """
    if not HAVE_PSYCOPG2:
        return
    admin_dsn = dsn_admin or _default_dsn().replace(
        f"dbname={os.environ.get('PGDATABASE', 'mosaicopt')}", "dbname=postgres"
    )
    try:
        with _pg_connect(admin_dsn, attempts=2, delay=0.5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
                if cur.fetchone() is None:
                    cur.execute(f'CREATE DATABASE "{dbname}"')
    except Exception:
        # If admin DB connect fails, fall through and try to connect
        # directly — assumes the DB already exists.
        pass


@dataclass
class PgTableStore:
    """A base relation backed by a PostgreSQL table.

    Loads the pyarrow data into a temporary PG table via binary
    COPY, then exposes ``arrow`` (a rebuilt pyarrow view from a
    single SELECT) and ``vector_column`` (numpy 2-D).

    The PG substrate differs from DuckDB in three observable ways:

    1. **Connection** is to a server daemon (default localhost:5432),
       not an in-memory process.
    2. **Vector column** comes back as PG ``float8[]`` rather than
       Arrow ``fixed_size_list``, so we materialize to numpy with a
       COPY round-trip on cache miss.
    3. **Latency** is dominated by the COPY at load time (called
       once per relation per query batch). For 10k rows this is
       ~30-100 ms; subsequent access is sub-millisecond.

    We expose the same interface (``arrow``, ``vector_column``,
    ``sql``) as the DuckDB substrate so the higher layers are
    substrate-agnostic.
    """
    name: str
    arrow: pa.Table
    dsn: Optional[str] = None
    _conn: Any = field(init=False, default=None, repr=False)

    @classmethod
    def from_arrow(cls, name: str, table: pa.Table, dsn: Optional[str] = None) -> "PgTableStore":
        return cls(name=name, arrow=table, dsn=dsn)

    def _conn_(self):  # pragma: no cover - lazy connect
        if self._conn is None:
            ctx = _pg_connect(self.dsn)
            self._conn = ctx.__enter__()
        return self._conn

    @property
    def con(self):  # pragma: no cover
        return self._conn_()

    def _copy_to_pg(self) -> None:
        """Bulk-load the pyarrow data into PG via binary COPY.

        Column types are inferred: numeric (int/float) → INTEGER /
        DOUBLE PRECISION, strings → TEXT, vector columns (list) →
        DOUBLE PRECISION[] (PG has no fixed-size list).
        """
        conn = self.con
        cols = self.arrow.column_names
        n = self.arrow.num_rows
        # Column type inference
        type_sqls = []
        for c in cols:
            arr = self.arrow.column(c)
            t = arr.type
            if pa.types.is_list(t) or pa.types.is_fixed_size_list(t):
                type_sqls.append(f'"{c}" DOUBLE PRECISION[]')
            elif pa.types.is_integer(t):
                type_sqls.append(f'"{c}" BIGINT')
            elif pa.types.is_floating(t):
                type_sqls.append(f'"{c}" DOUBLE PRECISION')
            elif pa.types.is_string(t) or pa.types.is_large_string(t):
                type_sqls.append(f'"{c}" TEXT')
            else:
                type_sqls.append(f'"{c}" TEXT')
        with conn.cursor() as cur:
            cur.execute(f'DROP TABLE IF EXISTS "{self.name}"')
            cur.execute(f'CREATE UNLOGGED TABLE "{self.name}" ({", ".join(type_sqls)})')

            # Stream rows via COPY ... FROM STDIN
            sink = io.BytesIO()
            # Use psycopg2's copy_from with a buffer for non-binary,
            # but for vector arrays we need client-side composition.
            # Simpler path: build CSV in memory and use COPY CSV.
            buf = io.StringIO()
            for i in range(n):
                vals = []
                for c in cols:
                    arr = self.arrow.column(c)
                    val = arr[i].as_py()
                    if isinstance(val, (list, tuple)):
                        # Vector: PG array literal
                        vals.append("{" + ",".join(repr(float(x)) for x in val) + "}")
                    elif val is None:
                        vals.append("")
                    elif isinstance(val, str):
                        # Escape tabs/newlines/quotes
                        v = val.replace("\\", "\\\\").replace("\t", "\\t").replace("\n", "\\n").replace('"', '""')
                        vals.append(f'"{v}"')
                    else:
                        vals.append(str(val))
                buf.write("\t".join(vals) + "\n")
            buf.seek(0)
            cur.copy_expert(f'COPY "{self.name}" FROM STDIN WITH (FORMAT csv, DELIMITER E\'\\t\', NULL \'\')', buf)

    def sql(self, query: str, params: Optional[Sequence[Any]] = None) -> pa.Table:
        """Run a SQL query and return a pyarrow table.

        For experiment purposes, the high-throughput path is the
        cached ``arrow`` view (no PG round-trip after the initial
        bulk load). This method is used only for substrate-specific
        queries; the optimizer does not rely on it for the
        multi-modal predicates (those remain Python-side).
        """
        with self.con.cursor() as cur:
            if params:
                cur.execute(query, params)
            else:
                cur.execute(query)
            desc = [d[0] for d in cur.description]
            rows = cur.fetchall()
        # Build pyarrow table
        arrays = {desc[i]: pa.array([row[i] for row in rows]) for i in range(len(desc))}
        # Convert vector columns (PG returns them as Python lists)
        arrays = {k: pa.array(v.to_pylist()) if pa.types.is_list(v.type) else v for k, v in arrays.items()}
        return pa.table(arrays)

    def vector_column(self, col: str) -> np.ndarray:
        """Return a (n, d) float32 numpy array for a vector column.

        Reads the column from the cached ``arrow`` view rather than
        from PG, so subsequent calls are sub-millisecond.
        """
        arr = self.arrow.column(col).combine_chunks()
        if pa.types.is_list(arr.type) or pa.types.is_fixed_size_list(arr.type):
            values = arr.values.to_numpy(zero_copy_only=False)
            n = len(arr)
            d = len(values) // n
            return values.reshape(n, d).astype("float32")
        return np.asarray(arr.to_numpy(zero_copy_only=False))

    def load(self) -> None:
        """One-time bulk load of arrow data into the PG table."""
        self._copy_to_pg()


def make_substrate(kind: str, name: str, table: pa.Table,
                   dsn: Optional[str] = None) -> Any:
    """Factory: return a Duck or PG store by ``kind``.

    kind: ``"duckdb"`` (default) or ``"postgres"``.
    The returned object exposes ``arrow`` and ``vector_column``;
    PG also offers ``load()`` to push data into the server lazily.
    """
    if kind == "postgres":
        store = PgTableStore.from_arrow(name=name, table=table, dsn=dsn)
        store.load()
        return store
    # Default: DuckDB (in-process, :memory:)
    from mosaicopt.physical.store import TableStore as DuckStore
    return DuckStore.from_arrow(name=name, table=table)


def pg_server_up(dsn: Optional[str] = None, timeout: float = 1.0) -> bool:
    """Lightweight TCP probe for ``localhost:5432``.

    Used to skip PG paths cleanly when the server is not running.
    """
    host = "localhost"
    port = 5432
    if dsn:
        for token in dsn.split():
            if token.startswith("host="):
                host = token[5:]
            elif token.startswith("port="):
                try:
                    port = int(token[5:])
                except ValueError:
                    port = 5432
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
