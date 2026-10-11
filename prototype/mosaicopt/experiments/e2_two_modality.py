"""E2: Two-modality sanity check.

Six 2-modality cells (V+C, V+text, V+graph, C+text, C+graph, text+graph)
compared across MosaicOpt / NaviX / ACORN.  The point of E2 is to
verify that at 2-modality queries, the CBO's contribution is
modest: all three systems should be within a small constant factor
of each other.  E2 acts as a baseline that E1 (≥3 modalities) is
contrasted against.

For each (modality-A, modality-B) cell we sweep selectivity from
0.001 to 0.2 and report QPS and recall.
"""
from __future__ import annotations

import csv
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow as pa
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROTO = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _PROTO not in sys.path:
    sys.path.insert(0, _PROTO)

from mosaicopt.experiments.data_gen import gen, MemoryDataset
from mosaicopt.algebra.ops import Plan, Scan, Select
from mosaicopt.algebra.predicates import (
    Predicate, ColumnPredicate, VectorRange, BM25Score, GraphReach,
)
from mosaicopt.optimizer.baselines import (
    MosaicOptSystem, NaviXStyle, ACORNStyle, all_systems,
)
from mosaicopt.physical.graph import GraphStore
from mosaicopt.experiments.e1_complex_multimodal import (
    plan_from_spec, ground_truth_from_spec, measure_cell_v2, QuerySpec,
)


# ---------------------------------------------------------------------
# 2-modality query generators
# ---------------------------------------------------------------------

def _build_classical(spec_qid: str, sub: str) -> ColumnPredicate:
    return ColumnPredicate("subsystem", "=", sub)


def _build_vector(spec_qid: str, q_vec: np.ndarray, tau: float) -> VectorRange:
    return VectorRange("emb", f"q_{spec_qid}", tau, "<")


def _build_text(spec_qid: str) -> BM25Score:
    return BM25Score("text", f"qt_{spec_qid}", 0.0, ">")


def _build_graph(spec_qid: str, depth: int, seed: int) -> GraphReach:
    return GraphReach(f"g_{spec_qid}", "graph_node", depth, "derived_from")


def make_2modality(
    ds: MemoryDataset,
    modalities: Tuple[str, str],
    target_selectivities: List[float],
) -> List[QuerySpec]:
    """Build 2-modality queries for a (mod_a, mod_b) pair.

    modalities in {V, T, G, C} where:
      V = vector, T = text BM25, G = graph reach, C = classical subsystem
    """
    rng = np.random.default_rng(hash(modalities) & 0xFFFF)
    subsystems = ["fault-tolerance", "perf", "frontend", "tooling", "docs"]
    queries = []
    d = ds.embs.shape[1]
    for i, target_s in enumerate(target_selectivities):
        sub = subsystems[i % len(subsystems)]
        sub_idx = subsystems.index(sub)
        center = ds.centers[sub_idx]
        q_vec = (center + 0.05 * rng.standard_normal(d).astype("float32"))
        q_vec = q_vec / (np.linalg.norm(q_vec) + 1e-9)
        tau = float(np.sqrt(2.0 * max(target_s, 1e-4) * d))
        depth = 1 if target_s > 0.05 else 2
        same_sub = np.flatnonzero(ds.subsystems == sub)
        seed = int(same_sub[rng.integers(0, len(same_sub))]) if len(same_sub) > 0 else 0
        qid = f"{''.join(modalities)}_{i:02d}_s{target_s:.3f}"
        preds: List[Predicate] = []
        params: Dict[str, Any] = {}
        for m in modalities:
            if m == "C":
                preds.append(_build_classical(qid, sub))
            elif m == "V":
                preds.append(_build_vector(qid, q_vec, tau))
                params[f"q_{qid}"] = q_vec
            elif m == "T":
                preds.append(_build_text(qid))
                params[f"qt_{qid}"] = "index optimizer recall"
            elif m == "G":
                preds.append(_build_graph(qid, depth, seed))
                params[f"g_{qid}"] = seed
        queries.append(QuerySpec(
            category="".join(modalities), qid=qid,
            target_selectivity=target_s,
            predicates=preds, query_params=params,
        ))
    return queries


# ---------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------

def run(
    n: int = 10_000,
    dim: int = 128,
    seed: int = 42,
    out_csv: str = None,
) -> List[Dict[str, Any]]:
    """Run the E2 sweep and write a CSV."""
    if out_csv is None:
        out_csv = os.path.join(
            os.path.abspath(os.path.join(_HERE, "..", "..", "results")),
            "e2_two_modality.csv",
        )
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    ds = gen(n=n, dim=dim, seed=seed)
    # Build chain graph
    edges: Dict[str, List[Tuple[int, int]]] = {"derived_from": []}
    for sub in np.unique(ds.subsystems):
        idx = np.flatnonzero(ds.subsystems == sub)
        for i in range(len(idx) - 1):
            edges["derived_from"].append((int(idx[i]), int(idx[i + 1])))
    graph = GraphStore(n_nodes=len(ds.ids), edges=edges)
    ds.graph_store = graph

    # 6 modality pairs × 8 selectivity points = 48 cells
    target_s = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.15, 0.2]
    pairs = [("V", "C"), ("V", "T"), ("V", "G"),
             ("T", "C"), ("G", "C"), ("T", "G")]
    all_specs: List[QuerySpec] = []
    for pair in pairs:
        all_specs += make_2modality(ds, pair, target_s)
    print(f"Built {len(all_specs)} query specs "
          f"({len(pairs)} pairs × {len(target_s)} selectivity points)")

    systems = all_systems(ds.store, ds.graph_store)

    rows: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    for i, spec in enumerate(all_specs):
        truth = ground_truth_from_spec(ds, spec)
        cell_results = measure_cell_v2(ds, spec, systems, truth, n_runs=1)
        for r in cell_results:
            rows.append({
                "modality_pair": spec.category,
                "qid": spec.qid,
                "target_selectivity": spec.target_selectivity,
                "actual_selectivity": r.actual_selectivity,
                "n_truth": r.n_truth,
                "n_candidate": r.n_candidate,
                "system": r.system,
                "latency_ms": r.latency_ms,
                "qps": r.qps,
                "recall": r.recall,
            })
        if (i + 1) % 16 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  [{i+1}/{len(all_specs)}] elapsed={elapsed:.1f}s")

    if rows:
        fieldnames = list(rows[0].keys())
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\nWrote {len(rows)} rows to {out_csv}")
    return rows


if __name__ == "__main__":
    rows = run()
    print(f"\nE2 done: {len(rows)} cells")
