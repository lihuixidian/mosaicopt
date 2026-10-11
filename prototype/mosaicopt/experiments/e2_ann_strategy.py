"""E2: ANN strategy selection (CBO).

Setup
-----
Sweep selectivity s from 0.01 to 0.9. For each s, run a vector
range query Sigma_{||V-q||<tau}(R) with each of {flat, ivf, hnsw}
and also the CBO-chosen strategy. Measure:
    * query latency (ms)
    * recall@10 relative to the flat ground truth

Plot
----
time vs selectivity (one line per strategy, markers for CBO choice)
recall vs selectivity
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa

from mosaicopt.experiments.data_gen import MemoryDataset
from mosaicopt.physical.ann import AnnIndex
from mosaicopt.optimizer.cbo import cbo_pick


def run(data: MemoryDataset, n_repeats: int = 3) -> "E2Result":
    """Run the ANN strategy selection experiment.

    Returns
    -------
    E2Result
        Per-(s, strategy) measurements + CBO choices.
    """
    embs = data.embs
    n, dim = embs.shape
    rng = np.random.default_rng(7)

    selectivities = np.array([
        0.01, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.40, 0.50, 0.60, 0.70, 0.80, 0.90
    ])

    strategies = ["flat", "ivf", "hnsw"]
    # Build each index once (amortized build)
    indices = {s: AnnIndex(embs, dim, strategy=s) for s in strategies}

    # Use one query vector; vary tau to sweep selectivity
    q = rng.standard_normal(dim).astype("float32")
    q /= np.linalg.norm(q) + 1e-9
    d_all = np.linalg.norm(embs - q.reshape(1, -1), axis=1)
    quantiles = {float(s): float(np.quantile(d_all, s)) for s in selectivities}

    rows: List[Dict] = []
    for s in selectivities:
        tau = quantiles[float(s)]
        cbo_choice = cbo_pick(float(s), n, dim, recall_floor=0.95)
        # Ground truth from flat
        truth = np.where(d_all < tau)[0]
        for strat in strategies:
            idx = indices[strat]
            times: List[float] = []
            recalls: List[float] = []
            for _ in range(n_repeats):
                t0 = time.perf_counter()
                ids, dists = idx.range_query(q, threshold=tau)
                dt = (time.perf_counter() - t0) * 1000.0
                times.append(dt)
                # Recall = fraction of ground truth that appears in ids
                if len(truth) > 0:
                    r = len(set(ids.tolist()) & set(truth.tolist())) / len(truth)
                else:
                    r = 1.0
                recalls.append(r)
            rows.append({
                "selectivity": float(s),
                "tau": tau,
                "strategy": strat,
                "latency_ms": float(np.median(times)),
                "recall": float(np.median(recalls)),
                "n_truth": int(len(truth)),
                "cbo_choice": (strat == cbo_choice),
                "cbo_picked": cbo_choice,
            })
    return E2Result(rows=rows, selectivities=selectivities.tolist())


@dataclass
class E2Result:
    rows: List[Dict]
    selectivities: List[float]

    def to_csv(self, path: str) -> None:
        import csv
        if not self.rows:
            return
        keys = list(self.rows[0].keys())
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)

    def plot(self, results_dir: str) -> None:
        """Generate the time / recall plots."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        s = np.array(self.selectivities)
        # Time plot
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        colors = {"flat": "#C1121F", "ivf": "#003049", "hnsw": "#669BBC"}
        for strat in ["flat", "ivf", "hnsw"]:
            ys = []
            for sel in s:
                matches = [r for r in self.rows if r["strategy"] == strat and abs(r["selectivity"] - sel) < 1e-9]
                ys.append(matches[0]["latency_ms"] if matches else np.nan)
            ax.plot(s, ys, "-o", label=strat, color=colors[strat], markersize=4)
        # Mark CBO choices
        cbo_s = []
        cbo_t = []
        cbo_c = []
        for sel in s:
            matches = [r for r in self.rows if abs(r["selectivity"] - sel) < 1e-9 and r["cbo_choice"]]
            if matches:
                cbo_s.append(sel)
                cbo_t.append(matches[0]["latency_ms"])
                cbo_c.append(colors[matches[0]["strategy"]])
        for x, y, c in zip(cbo_s, cbo_t, cbo_c):
            ax.scatter([x], [y], marker="*", s=120, edgecolor="black", color=c, zorder=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Selectivity (s)")
        ax.set_ylabel("Query latency (ms)")
        ax.set_title("E2: ANN strategy latency vs selectivity (stars = CBO choice)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{results_dir}/e2_time.png", dpi=150)
        plt.close(fig)

        # Recall plot
        fig, ax = plt.subplots(figsize=(6.5, 3.5))
        for strat in ["flat", "ivf", "hnsw"]:
            ys = []
            for sel in s:
                matches = [r for r in self.rows if r["strategy"] == strat and abs(r["selectivity"] - sel) < 1e-9]
                ys.append(matches[0]["recall"] if matches else np.nan)
            ax.plot(s, ys, "-o", label=strat, color=colors[strat], markersize=4)
        # Mark CBO choices
        cbo_s = []
        cbo_r = []
        cbo_c = []
        for sel in s:
            matches = [r for r in self.rows if abs(r["selectivity"] - sel) < 1e-9 and r["cbo_choice"]]
            if matches:
                cbo_s.append(sel)
                cbo_r.append(matches[0]["recall"])
                cbo_c.append(colors[matches[0]["strategy"]])
        for x, y, c in zip(cbo_s, cbo_r, cbo_c):
            ax.scatter([x], [y], marker="*", s=120, edgecolor="black", color=c, zorder=10)
        ax.set_xscale("log")
        ax.set_xlabel("Selectivity (s)")
        ax.set_ylabel("Recall@K")
        ax.set_title("E2: ANN strategy recall vs selectivity (stars = CBO choice)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(f"{results_dir}/e2_recall.png", dpi=150)
        plt.close(fig)

    def summary(self) -> Dict[str, float]:
        # Number of times CBO picked each strategy
        counts = {"flat": 0, "ivf": 0, "hnsw": 0}
        s_to_choice: Dict[float, str] = {}
        for r in self.rows:
            if r["cbo_choice"]:
                counts[r["strategy"]] += 1
                s_to_choice[r["selectivity"]] = r["strategy"]
        return {
            "n_selectivities": len(self.selectivities),
            "cbo_picked_flat": counts["flat"],
            "cbo_picked_ivf": counts["ivf"],
            "cbo_picked_hnsw": counts["hnsw"],
        }
