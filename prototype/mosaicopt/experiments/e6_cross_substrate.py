"""E6: cross-substrate portability (DuckDB vs SQLite / PostgreSQL).

Re-runs the E1 sweep on \emph{two} substrates and reports
per-cell QPS + recall on each, on the same glue data. The
optimizer / cost model / multi-modal predicates are unchanged;
only the substrate layer swaps.

Three substrates in the prototype:
* DuckDB (in-process, :memory:), the default.
* SQLite (file-based, in-process), the E6 second substrate.
* PostgreSQL 17 (server-side), the EDB-based install on
  this machine is interactive, so for the E6 sweep we
  use SQLite; PG is supported via ``store_pg.py`` and
  matches the same interface.

Methodology: substitute ``ds.store`` with a different substrate
implementation (same ``arrow``, ``vector_column`` interface),
rebuild the systems dict, run the same QSpec set on each.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)


@dataclass
class CrossSubstrateRow:
    substrate: str
    category: str
    selectivity: float
    system: str
    qps: float
    recall: float
    latency_ms: float
    bulk_load_ms: float


def _swap_store(ds, new_store):
    """Return a copy of ds with .store replaced."""
    from dataclasses import replace
    return replace(ds, store=new_store)


def _build_systems(ds_view, graph):
    from mosaicopt.experiments.e1_real import SingleWorkloadBaseline
    from mosaicopt.optimizer.baselines import all_systems
    systems = all_systems(ds_view.store, graph)
    systems["bvb_native"] = SingleWorkloadBaseline(ds_view.store, graph)
    return systems


def _graph_from(ds):
    from mosaicopt.physical.graph import GraphStore
    from scipy.sparse import csr_matrix
    rows = np.array([u for u, v in ds.graph_edges], dtype=np.int32)
    cols = np.array([v for u, v in ds.graph_edges], dtype=np.int32)
    data = np.ones(len(ds.graph_edges), dtype=np.float32)
    mat = csr_matrix((data, (rows, cols)), shape=(ds.n, ds.n))
    return GraphStore(mat)


def _run_for_substrate(
    substrate: str,
    ds: Any,
    categories: Tuple[str, ...] = ("Q1_VTC", "Q7_MV", "Q8_Join"),
    selectivities: Tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
    n_runs: int = 3,
) -> List[CrossSubstrateRow]:
    """Run headline cells on one substrate; uses the E1 sweep machinery."""
    from mosaicopt.experiments.e1_real import (
        _make_qspecs, _plan_for_spec, ground_truth,
    )

    # Build / load the substrate
    if substrate == "duckdb":
        from mosaicopt.physical.store import TableStore
        t0 = time.perf_counter()
        new_store = TableStore.from_arrow("glue", ds.store.arrow)
        _ = new_store.con
        bulk = (time.perf_counter() - t0) * 1000.0
    elif substrate == "sqlite":
        from mosaicopt.physical.store_sqlite import SqliteTableStore
        t0 = time.perf_counter()
        new_store = SqliteTableStore.from_arrow("glue", ds.store.arrow)
        _ = new_store.con
        bulk = (time.perf_counter() - t0) * 1000.0
    elif substrate == "postgres":
        from mosaicopt.physical.store_pg import PgTableStore, pg_server_up
        if not pg_server_up():
            return []
        new_store = PgTableStore.from_arrow("glue", ds.store.arrow)
        t0 = time.perf_counter()
        new_store.load()
        bulk = (time.perf_counter() - t0) * 1000.0
    else:
        raise ValueError(substrate)

    view = _swap_store(ds, new_store)
    graph = _graph_from(ds)
    systems = _build_systems(view, graph)

    # Filter qspecs
    qspecs = _make_qspecs(view, cats=list(categories))
    qspecs = [q for q in qspecs
              if any(abs(q.target_selectivity - s) < 1e-6
                     for s in selectivities)]

    SYSTEMS_OF_INTEREST = {"mosaicopt", "post_filter", "pre_filter",
                           "navix", "acorn", "bvb_native"}

    rows: List[CrossSubstrateRow] = []
    for spec in qspecs:
        plan, qparams = _plan_for_spec(view, spec)
        gt = ground_truth(view, spec)
        for sys_name, sys_obj in systems.items():
            if sys_name not in SYSTEMS_OF_INTEREST:
                continue
            latencies, recalls = [], []
            for _ in range(n_runs):
                if sys_name == "bvb_native":
                    res = sys_obj.run(
                        plan, query_params=qparams,
                        query_id=spec.qid, ground_truth_ids=gt,
                        query_category=spec.category,
                    )
                else:
                    res = sys_obj.run(
                        plan, query_params=qparams,
                        query_id=spec.qid, ground_truth_ids=gt,
                    )
                latencies.append(res.wall_clock_ms)
                recalls.append(res.recall)
            lat = float(np.median(latencies))
            rec = float(np.median(recalls))
            rows.append(CrossSubstrateRow(
                substrate=substrate,
                category=spec.category,
                selectivity=spec.target_selectivity,
                system=sys_name,
                qps=1000.0 / max(1e-9, lat),
                recall=rec,
                latency_ms=lat,
                bulk_load_ms=bulk,
            ))
    return rows


def run(
    n: int = 10_000,
    dim: int = 128,
    out_csv: str = "results/e6_cross_substrate.csv",
    n_runs: int = 3,
) -> Dict:
    """Run E6 on DuckDB + SQLite (+ PostgreSQL if available), write CSV."""
    from mosaicopt.experiments.data_gen_real import gen_glue_table

    print(f"[E6] generating glue table: n={n}, d={dim}", flush=True)
    ds = gen_glue_table(n=n, dim=dim)
    all_rows: List[CrossSubstrateRow] = []

    substrates = ["duckdb", "sqlite"]
    # Try PG too if reachable
    from mosaicopt.physical.store_pg import pg_server_up
    if pg_server_up():
        substrates.append("postgres")
        print("[E6] PostgreSQL server reachable; will include PG cells.")
    else:
        print("[E6] PostgreSQL not reachable; SQLite used as the second substrate.")

    for sub in substrates:
        print(f"[E6] running on {sub} ...", flush=True)
        t0 = time.perf_counter()
        try:
            rows = _run_for_substrate(sub, ds, n_runs=n_runs)
        except Exception as e:
            print(f"   {sub} run failed: {e!r}; skipping")
            continue
        for r in rows:
            all_rows.append(r)
        print(f"   done in {(time.perf_counter()-t0):.1f}s, {len(rows)} cells")

    # Write CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["substrate", "category", "selectivity", "system",
                    "qps", "recall", "latency_ms", "bulk_load_ms"])
        for r in all_rows:
            w.writerow([r.substrate, r.category, r.selectivity, r.system,
                        f"{r.qps:.2f}", f"{r.recall:.4f}",
                        f"{r.latency_ms:.2f}", f"{r.bulk_load_ms:.2f}"])
    print(f"[E6] wrote {out_csv} with {len(all_rows)} rows")
    return {}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--out", default="results/e6_cross_substrate.csv")
    a = p.parse_args()
    run(n=a.n, dim=a.dim, out_csv=a.out, n_runs=a.runs)
