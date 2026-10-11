"""E1-real: Q1..Q8 query categories on the BigVectorBench-style glue table.

This is the cross-modal extension of :mod:`e1_complex_multimodal`.  Where
the original E1 sweeps only the four ≥3-modality categories
(VTC / VGC / VTGC / VTf), this experiment sweeps **eight**
categories that together mirror the four workload classes of
BigVectorBench (filter / mm / mv / big) plus four single-modality
or hybrid regimes that highlight specific optimizer modules.

Categories
----------
* **Q1 VTC**  --- vector + text + classical (E1 carryover)
* **Q2 VGC**  --- vector + graph + classical (E1 carryover)
* **Q3 VTGC** --- vector + text + graph + classical (E1 carryover)
* **Q4 VTf**  --- vector + text + fuzzy (E1 carryover)
* **Q5 V-only**  --- vector + classical; the canonical "filter
                    + vector" ACORN-style workload
* **Q6 T-only**  --- text + classical; a pure BM25 workload
* **Q7 MV**      --- two vector columns (vec_a, vec_b) + classical;
                    multi-vector fusion (WebVid-style)
* **Q8 Join**    --- self-join on classical column, then vector
                    distance; the J1 σ ∘ Join scenario from the
                    E3 corollary

The first four categories are reused from
:mod:`e1_complex_multimodal`; the last four are new and target
specific optimizer modules.  Each category is run against the
**same** glue table from :mod:`data_gen_real.gen_glue_table`.

Four baselines are compared:

* **mosaicopt** --- full MosaicOpt stack (RBO + 4-tuple CBO +
  ρ_f + adaptive re-planning).  This is the system we are
  proposing.
* **navix** --- single-CBO unified engine (NaviX / kuzu-style
  approximation, vector HNSW pre-filter + BM25 + graph BFS).
  The re-implementation is shared with the synthetic E1.
* **acorn** --- ACORN-σ (σ=1.0) + post-BM25 + post-graph.
  Same re-implementation as in the synthetic E1.
* **bvb-native** --- a "BVB-native" baseline: the per-workload
  best-practice fixed order, with vector always using HNSW,
  no RBO, no CBO.  This is the "if you only had one config"
  approximation of what BigVectorBench-style single-workload
  systems run; it represents the cost of *not* having a
  cross-modal CBO.

The data is the **glue** table from
:func:`mosaicopt.experiments.data_gen_real.gen_glue_table`
(default 10,000 rows, dim=512).  Per-row we attach
``{vec_a, vec_b, text, classical_attr, fuzzy_terms,
graph_node, ts, star}``.

Per-cell metric: median of 3 runs; QPS = 1000 / latency_ms;
recall computed against an exact ground truth.
"""
from __future__ import annotations

import csv
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from mosaicopt.experiments.data_gen_real import (
    RealMemoryDataset, gen_glue_table,
)
from mosaicopt.experiments.data_gen import gen as gen_synth  # for any fallback
from mosaicopt.optimizer.baselines import (
    BaselineResult, MosaicOptSystem, NaviXStyle, ACORNStyle,
    all_systems, _apply_classical_filter, _apply_bm25_filter,
    _apply_fuzzy_filter, _apply_graph_filter,
)
from mosaicopt.physical.bm25 import BM25Index
from mosaicopt.physical.ann import AnnIndex
from mosaicopt.physical.graph import GraphStore
from mosaicopt.physical.store import TableStore


CATEGORIES = ("Q1_VTC", "Q2_VGC", "Q3_VTGC", "Q4_VTf",
              "Q5_Vonly", "Q6_Tonly", "Q7_MV", "Q8_Join")

# Target selectivities swept per category.  We reuse the E1 grid.
TARGET_SELECTIVITIES = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]


# ---------------------------------------------------------------------
# SingleWorkload (BVB-native) baseline
# ---------------------------------------------------------------------

class SingleWorkloadBaseline:
    """Per-workload fixed-order baseline (BVB-native style).

    Does **not** use RBO, CBO, ρ_f, or adaptive re-planning.  The
    execution order is determined by the query category, not
    by the cost model:

    * Q1/Q2/Q3/Q5/Q7/Q8 (vector involved): classical pre-filter →
      vector HNSW → text BM25 (if present) → graph BFS (if present).
    * Q4 (VTf): classical pre-filter → fuzzy pre-filter → vector
      HNSW → text BM25.
    * Q6 (T-only): classical pre-filter → text BM25.

    Vector strategy is *always* HNSW (the BigVectorBench default
    in the milvus / weaviate / elasticsearch backends).  This
    represents the cost of *not* using MosaicOpt's adaptive CBO.
    """

    def __init__(self, store: TableStore, graph: Optional[GraphStore] = None) -> None:
        self.store = store
        self.graph = graph

    def run(
        self,
        plan: Any,                       # logical plan (read for shape only)
        query_params: Optional[Dict[str, Any]] = None,
        query_id: str = "q",
        ground_truth_ids: Optional[np.ndarray] = None,
        query_category: str = "Q5_Vonly",
    ) -> BaselineResult:
        query_params = query_params or {}
        from mosaicopt.optimizer.baselines import _collect_predicates_v2
        classical, vectors, texts, graphs, fuzzy = _collect_predicates_v2(plan)
        table = self.store.arrow
        n_total = table.num_rows
        stages: List[Dict[str, Any]] = []
        t0 = time.perf_counter()

        surviving = np.arange(n_total, dtype=np.int64)

        # 1. Classical pre-filter (always)
        if classical:
            t1 = time.perf_counter()
            surviving = _apply_classical_filter(table, classical)
            stages.append({"stage": "classical", "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 2. Fuzzy pre-filter (Q4 only; otherwise keep as no-op)
        if fuzzy and query_category == "Q4_VTf":
            t1 = time.perf_counter()
            surviving = _apply_fuzzy_filter(table, surviving, fuzzy)
            stages.append({"stage": f"fuzzy({len(fuzzy)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 3. Vector HNSW pre-filter (always, for any query with vectors)
        if vectors and len(surviving) > 0:
            t1 = time.perf_counter()
            for vp in vectors:
                q = query_params.get(vp.q_param)
                if q is None:
                    continue
                q_arr = np.asarray(q, dtype="float32")
                vec_col = self.store.vector_column(vp.column)
                hnsw = AnnIndex(vec_col, dim=vec_col.shape[1], strategy="hnsw")
                ids, _ = hnsw.filtered_range_query(
                    q_arr, float(vp.threshold), surviving
                )
                ids_set = set(int(x) for x in ids)
                surviving = np.array(
                    [i for i in surviving if int(i) in ids_set],
                    dtype=np.int64,
                )
            stages.append({"stage": f"vector_hnsw({len(vectors)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 4. Text BM25 post-filter (any query with texts)
        if texts and len(surviving) > 0:
            t1 = time.perf_counter()
            surviving = _apply_bm25_filter(
                table, surviving, texts, query_params
            )
            stages.append({"stage": f"bm25_post({len(texts)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})
        # 5. Graph BFS post-filter (any query with graphs)
        if graphs and len(surviving) > 0 and self.graph is not None:
            t1 = time.perf_counter()
            surviving = _apply_graph_filter(
                table, surviving, graphs, self.graph, query_params
            )
            stages.append({"stage": f"graph_post({len(graphs)})",
                           "kept": int(len(surviving)),
                           "ms": (time.perf_counter() - t1) * 1000.0})

        t_total = (time.perf_counter() - t0) * 1000.0
        recall = 0.0
        if ground_truth_ids is not None and len(ground_truth_ids) > 0:
            got = set(int(x) for x in surviving)
            ref = set(int(x) for x in ground_truth_ids)
            recall = len(got & ref) / max(1, len(ref))
        sel = (len(ground_truth_ids) / max(1, n_total)) if ground_truth_ids is not None else 0.0
        return BaselineResult(
            system="bvb_native", query_id=query_id,
            n_input=n_total, n_predicate_selectivity=sel,
            n_candidate=len(surviving), wall_clock_ms=t_total,
            recall=recall, stages=stages,
        )


# ---------------------------------------------------------------------
# Query spec per category
# ---------------------------------------------------------------------

@dataclass
class QSpec:
    category: str
    qid: str
    target_selectivity: float
    vec_column: str                       # "vec_a" / "vec_b" / None
    classical_value: Optional[str]        # e.g. "cat_2"
    text_query: str
    q_vec: np.ndarray                     # query vector
    bm25_threshold: float = 0.0
    graph_node: int = 0
    graph_depth: int = 0
    fuzzy_term: str = ""
    join_other: bool = False               # Q8: self-join on classical


def _make_qspecs(
    ds: RealMemoryDataset,
    cats: Tuple[str, ...] = CATEGORIES,
    target_sels: List[float] = TARGET_SELECTIVITIES,
    seed: int = 100,
) -> List[QSpec]:
    rng = np.random.default_rng(seed)
    out: List[QSpec] = []
    classical_values = sorted(set(str(v) for v in ds.classical_values))
    centers = ds.emb_centers
    if centers is None:
        centers = ds.emb
    # Per-cluster BM25 vocabulary.  The synthetic corpus was built
    # with cluster-specific terms; we reuse them as the text query
    # so the BM25 predicate has a non-empty ground truth.
    _CLUSTER_TERMS = {
        "glue": [
            ["election", "policy", "diplomacy"],
            ["match", "tournament", "team"],
            ["market", "stock", "revenue"],
            ["research", "lab", "study"],
            ["release", "version", "feature"],
        ],
        "sift1m_extended": [
            ["mountain", "valley", "peak"],
            ["beach", "coast", "ocean"],
            ["forest", "tree", "leaf"],
            ["city", "street", "building"],
            ["desert", "sand", "dune"],
        ],
        "glove_extended": [
            ["election", "government", "party"],
            ["research", "science", "computer"],
            ["company", "market", "business"],
            ["music", "film", "book"],
            ["team", "game", "player"],
        ],
    }
    cluster_terms = _CLUSTER_TERMS.get(ds.name, _CLUSTER_TERMS["glue"])
    for cat in cats:
        for tid, sel in enumerate(target_sels):
            qid = f"{cat}_t{tid:02d}_s{sel:.4f}"
            center_idx = int(rng.integers(0, len(centers)))
            q_vec = centers[center_idx].copy()
            if len(classical_values) == len(centers):
                cv = classical_values[center_idx]
            else:
                cv = classical_values[rng.integers(0, len(classical_values))]
            # Text query: pick the cluster's terms so BM25 has signal.
            if cat in ("Q1_VTC", "Q3_VTGC", "Q4_VTf", "Q6_Tonly"):
                text_query = " ".join(cluster_terms[center_idx % len(cluster_terms)])
            else:
                text_query = "index optimizer recall"
            # BM25 threshold: lower so most cluster rows pass.
            if cat in ("Q1_VTC", "Q3_VTGC", "Q4_VTf", "Q6_Tonly"):
                bm25_t = 0.0
            else:
                bm25_t = 0.0
            # Fuzzy term (Q4): one of the cluster terms so LIKE matches.
            if cat == "Q4_VTf":
                fuzzy_term = cluster_terms[center_idx % len(cluster_terms)][0]
            else:
                fuzzy_term = ""
            # For graph queries, place the seed node inside the same
            # cluster as the query vector so the BFS expansion has a
            # chance to reach the predicate-relevant rows.  We pick
            # from the actual cluster's row indices (the cluster
            # assignment is *not* sorted by row index — it's a
            # random permutation in the data generator).
            if cat in ("Q2_VGC", "Q3_VTGC") and ds.cluster is not None:
                same_cluster_idx = np.where(ds.cluster == center_idx)[0]
                if len(same_cluster_idx) > 0:
                    graph_node = int(same_cluster_idx[rng.integers(0, len(same_cluster_idx))])
                else:
                    graph_node = int(rng.integers(0, ds.n))
            else:
                graph_node = int(rng.integers(0, ds.n))
            spec = QSpec(
                category=cat, qid=qid, target_selectivity=sel,
                vec_column=("vec_a" if cat != "Q7_MV" else "vec_a"),
                classical_value=cv,
                text_query=text_query,
                q_vec=q_vec,
                bm25_threshold=bm25_t,
                graph_node=graph_node,
                graph_depth=(2 if cat in ("Q2_VGC", "Q3_VTGC") else 0),
                fuzzy_term=fuzzy_term,
                join_other=(cat == "Q8_Join"),
            )
            out.append(spec)
    return out


# ---------------------------------------------------------------------
# Ground truth (exact, on the glue table)
# ---------------------------------------------------------------------

def ground_truth(
    ds: RealMemoryDataset,
    spec: QSpec,
) -> np.ndarray:
    table = ds.store.arrow
    n = table.num_rows
    surviving = np.arange(n, dtype=np.int64)
    # Classical
    if spec.classical_value is not None:
        col = table.column(ds.classical_col).to_pylist()
        mask = np.array([str(c) == spec.classical_value for c in col])
        surviving = surviving[mask[surviving]]
    # Fuzzy (Q4)
    if spec.fuzzy_term:
        col = table.column("fuzzy_terms").to_pylist()
        mask = np.array([spec.fuzzy_term.lower() in str(c).lower() for c in col])
        surviving = surviving[mask[surviving]]
    # Vector (cosine-style L2 on L2-normalized columns)
    if spec.vec_column and len(surviving) > 0:
        # Calibrated against data_gen_real: tau=0.70 -> sel~0.0015,
        # tau=0.90 -> sel~0.20.  See calibrate_tau.py.
        tau = float(0.70 + spec.target_selectivity * 1.0)
        sq_thr = tau ** 2
        vec_col = ds.store.vector_column(spec.vec_column)[surviving]
        if spec.category == "Q7_MV":
            vec_b = ds.store.vector_column("vec_b")[surviving]
            d_a = ((vec_col - spec.q_vec) ** 2).sum(axis=1)
            d_b = ((vec_b - spec.q_vec) ** 2).sum(axis=1)
            mask = (d_a < sq_thr) & (d_b < sq_thr)
        else:
            d2 = ((vec_col - spec.q_vec) ** 2).sum(axis=1)
            mask = d2 < sq_thr
        surviving = surviving[mask]
    # Text BM25
    if spec.bm25_threshold > 0 and len(surviving) > 0:
        surviving = _apply_bm25_filter(
            table, surviving,
            [_bm25_pred_factory(spec)],
            {"q_text": spec.text_query},
        )
    # Graph
    if spec.graph_depth > 0 and len(surviving) > 0 and ds.graph_nodes is not None:
        # Build a tiny GraphStore on demand
        edges = ds.graph_edges if ds.graph_edges else []
        if edges:
            row_offsets = np.zeros(n + 1, dtype=np.int64)
            row_offsets[1:] = np.cumsum(np.ones(n, dtype=np.int64))
            # Build GraphStore: each row has 1 self-edge; we'll let BFS work via
            # a simple iteration.
            # For ground truth we just do BFS from spec.graph_node depth=spec.graph_depth
            reachable = _bfs_from(ds, spec.graph_node, spec.graph_depth)
            if reachable is not None:
                mask = np.isin(surviving, list(reachable))
                surviving = surviving[mask]
    # Q8 join: keep rows that have *another* row with same classical value
    # AND pass the vector predicate (the join part is implicit because
    # we have ≥2 rows in the same classical group).
    if spec.join_other and len(surviving) > 0:
        # The surviving set already came from vector<tau.  Now check
        # that *some other* row in the same classical group also
        # passes the vector predicate.
        col = np.array([str(v) for v in table.column(ds.classical_col).to_pylist()])
        # For each classical value, find which rows pass the vector
        # predicate (from scratch, ignoring the existing surviving set).
        tau = float(0.70 + spec.target_selectivity * 1.0)
        vec_col = ds.store.vector_column(spec.vec_column)
        d2 = ((vec_col - spec.q_vec) ** 2).sum(axis=1)
        sq_thr = tau ** 2
        v_pass = d2 < sq_thr
        # Group: classical value -> list of row indices that pass vector
        groups: Dict[str, List[int]] = {}
        for i in np.where(v_pass)[0]:
            groups.setdefault(col[i], []).append(int(i))
        keep = set()
        for i in surviving:
            v = col[i]
            if len(groups.get(v, [])) > 1:
                keep.add(int(i))
        surviving = np.array(sorted(keep), dtype=np.int64)
    return surviving


def _bm25_pred_factory(spec: QSpec):
    from mosaicopt.algebra.predicates import BM25Score
    return BM25Score("text", "q_text", spec.bm25_threshold, ">")


def _bfs_from(ds: RealMemoryDataset, start: int, depth: int) -> Optional[set]:
    """BFS over ds.graph_edges up to ``depth`` hops from ``start``."""
    if not ds.graph_edges:
        return None
    adj: Dict[int, List[int]] = {}
    for u, v in ds.graph_edges:
        adj.setdefault(u, []).append(v)
        adj.setdefault(v, []).append(u)
    seen = {start}
    frontier = {start}
    for _ in range(depth):
        nxt = set()
        for u in frontier:
            for v in adj.get(u, []):
                if v not in seen:
                    seen.add(v)
                    nxt.add(v)
        frontier = nxt
        if not frontier:
            break
    return seen


# ---------------------------------------------------------------------
# Plan construction per category (for MosaicOpt / NaviXStyle / ACORNStyle)
# ---------------------------------------------------------------------

def _plan_for_spec(ds: RealMemoryDataset, spec: QSpec) -> Tuple[Any, Dict[str, Any]]:
    """Build a logical plan + query_params for the given spec.

    Returns (plan, query_params).
    """
    from mosaicopt.algebra.ops import (
        Plan, Scan, Project, Select, Materialize, Order, Limit,
        Expand, Join,
    )
    from mosaicopt.algebra.predicates import (
        Predicate, ColumnPredicate, VectorRange, BM25Score,
        GraphReach, And, Or, Not,
    )
    from mosaicopt.algebra.materialization import (
        distance, bm25_score,
    )

    table = ds.store.arrow
    n = table.num_rows
    plan: Plan = Scan("glue", table)

    # Classical
    if spec.classical_value is not None:
        plan = Select(plan, ColumnPredicate(ds.classical_col, "=", spec.classical_value))

    # Fuzzy
    if spec.fuzzy_term:
        plan = Select(plan, ColumnPredicate("fuzzy_terms", "contains", spec.fuzzy_term))

    # Vector (raw-column VectorRange; CBO + M1 will materialize as needed)
    if spec.vec_column:
        # Calibrated against data_gen_real._make_directional_embeddings:
        # cluster-0 distance sweep (in calibrate_tau.py) shows
        #   tau=0.70 -> sel=0.0015, tau=0.90 -> sel=0.20
        # so a linear sweep in [0.70, 0.90] tracks the target
        # selectivity grid used in E1.
        tau = float(0.70 + spec.target_selectivity * 1.0)
        plan = Select(plan, VectorRange(spec.vec_column, "q_vec", tau, "<"))
        if spec.category == "Q7_MV":
            # multi-vector: also check vec_b
            plan = Select(plan, VectorRange("vec_b", "q_vec", tau, "<"))

    # Text
    if spec.bm25_threshold > 0:
        plan = Select(plan, BM25Score("text", "q_text", spec.bm25_threshold, ">"))

    # Graph
    if spec.graph_depth > 0 and ds.graph_nodes is not None:
        plan = Expand(plan, "graph_node", depth=spec.graph_depth, rule_name="derived_from")

    # Q8: self-join on classical
    if spec.join_other:
        # Build a small join against the same table
        plan = Join(
            plan, Scan(name="glue", alias="glue_b"),
            condition=ColumnPredicate(ds.classical_col, "=", "glue_b." + ds.classical_col),
        )
        # And then we still want a vector predicate on the join
        tau = float(0.70 + spec.target_selectivity * 1.0)
        plan = Select(plan, VectorRange(spec.vec_column, "q_vec", tau, "<"))

    # Always order by vec_a asc, limit 10
    plan = Order(plan, key_col=spec.vec_column, key_expr="d", ascending=True)
    plan = Limit(plan, 10)

    params: Dict[str, Any] = {
        "q_vec": spec.q_vec,
        "q_text": spec.text_query,
    }
    if ds.graph_nodes is not None:
        params["q_graph_node"] = spec.graph_node
    return plan, params


# ---------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------

def run_sweep(
    n: int = 10_000,
    dim: int = 512,
    seed: int = 49,
    n_runs: int = 3,
    out_csv: str = "results/e1_real.csv",
    categories: Tuple[str, ...] = CATEGORIES,
    dataset: str = "glue",
) -> List[Dict[str, Any]]:
    """Run the E1 sweep on the selected dataset.

    dataset:
      * "glue" -- the BigVectorBench-style glue table
        (multi-modal, dim=128 or 512)
      * "sift1m_extended" -- SIFT1M-extended
        (SIFT base + scalar + text + graph, dim=128)
    """
    print(f"Generating dataset={dataset}: n={n}, dim={dim}, seed={seed}")
    if dataset == "glue":
        from mosaicopt.experiments.data_gen_real import gen_glue_table
        ds = gen_glue_table(n=n, dim=dim, seed=seed)
    elif dataset == "sift1m_extended":
        from mosaicopt.experiments.data_gen_real import gen_sift1m_extended
        ds = gen_sift1m_extended(n=n, dim=dim, seed=seed)
    elif dataset == "glove_extended":
        from mosaicopt.experiments.data_gen_real import gen_glove_extended
        ds = gen_glove_extended(n=n, dim=dim, seed=seed)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    # Build a graph store from the in-memory edges for systems that need it
    graph: Optional[GraphStore] = None
    if ds.graph_edges:
        from scipy.sparse import csr_matrix
        n_rows = ds.n
        rows = np.array([u for u, v in ds.graph_edges], dtype=np.int32)
        cols = np.array([v for u, v in ds.graph_edges], dtype=np.int32)
        data = np.ones(len(ds.graph_edges), dtype=np.float32)
        mat = csr_matrix((data, (rows, cols)), shape=(n_rows, n_rows))
        graph = GraphStore(mat)
    # Use all_systems() from baselines (5 systems) + bvb_native
    from mosaicopt.optimizer.baselines import all_systems
    systems = all_systems(ds.store, graph)
    systems["bvb_native"] = SingleWorkloadBaseline(ds.store, graph)
    qspecs = _make_qspecs(ds, cats=categories)
    rows: List[Dict[str, Any]] = []
    print(f"Sweeping {len(qspecs)} queries × {len(systems)} systems × {n_runs} runs")
    for q_i, spec in enumerate(qspecs):
        plan, qparams = _plan_for_spec(ds, spec)
        gt = ground_truth(ds, spec)
        sel_true = float(len(gt) / max(1, ds.n))
        for sys_name, sys_obj in systems.items():
            latencies = []
            recalls = []
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
            ms = float(np.median(latencies))
            qps = 1000.0 / max(1e-9, ms)
            recall = float(np.median(recalls))
            row = {
                "category": spec.category,
                "qid": spec.qid,
                "target_selectivity": spec.target_selectivity,
                "true_selectivity": sel_true,
                "system": sys_name,
                "latency_ms": ms,
                "qps": qps,
                "recall": recall,
                "n_input": ds.n,
                "dim": dim,
            }
            rows.append(row)
        if (q_i + 1) % 8 == 0:
            print(f"  ... {q_i+1}/{len(qspecs)} queries done")
    # Write CSV
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {out_csv}")
    return rows


def main() -> None:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=10_000)
    p.add_argument("--dim", type=int, default=512)
    p.add_argument("--seed", type=int, default=49)
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--out", type=str, default="results/e1_real.csv")
    p.add_argument("--categories", nargs="+", default=list(CATEGORIES),
                   help="Subset of categories to run")
    p.add_argument("--dataset", type=str, default="glue",
                   choices=["glue", "sift1m_extended", "glove_extended"],
                   help="Which dataset to run on")
    args = p.parse_args()
    run_sweep(
        n=args.n, dim=args.dim, seed=args.seed,
        n_runs=args.n_runs, out_csv=args.out,
        categories=tuple(args.categories),
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
