"""Physical layer: H_{V,q,tau}, B_{col,range}, F(R), I_{col,term}, G_{g,d,dir}.

This package implements the physical operators that realize the
logical algebra on top of Python libraries (no DuckDB extensions
required because the network blocks extension downloads).

Modules
-------
ann     : faiss HNSW / IVF / flat vector range scanner -> H_{V,q,tau}
bm25    : rank_bm25 wrapper -> I_{col,term}
graph   : scipy.sparse graph reachability (no NetworkX) -> G_{g,d,dir}
store   : DuckDB substrate (table -> pyarrow) + materialization columns
store_pg: PostgreSQL substrate (alternative RDBMS, same interface)
"""

from .ann import AnnIndex, ann_range_query
from .bm25 import BM25Index
from .graph import GraphStore
from .store import TableStore, materialize_column
from .store_pg import (
    PgTableStore, make_substrate, pg_server_up, ensure_database,
    HAVE_PSYCOPG2,
)

__all__ = [
    "AnnIndex", "ann_range_query",
    "BM25Index",
    "GraphStore",
    "TableStore", "materialize_column",
    "PgTableStore", "make_substrate", "pg_server_up", "ensure_database",
    "HAVE_PSYCOPG2",
]
