"""Runner: executes E1-E4 in sequence and writes results/ to disk.

Outputs
-------
results/e1_rewrite.csv          per-query latencies (naive vs M1)
results/e2_ann_strategy.csv     per-(s, strategy) latencies
results/e3_orchestrator.csv     one-row multi-modal vs multi-path
results/e4_endtoend.csv         per-phase latency for Q1->Q6
results/e4_samples.csv          top-K sample results for Q1->Q6
results/e2_time.png             E2 latency plot
results/e2_recall.png           E2 recall plot
results/e3_latency.png          E3 comparison plot
results/summary.txt             aggregated summary
results/summary.json            machine-readable summary
"""
from __future__ import annotations

import json
import os
import sys
import time
from typing import Dict

import numpy as np

from mosaicopt.experiments.data_gen import gen


def _here() -> str:
    """Return the absolute path of the results/ directory."""
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def run_all(n: int = 10_000, dim: int = 768, seed: int = 42) -> Dict:
    """Run all four experiments and save results to mosaicopt/results/.

    Parameters
    ----------
    n : int
        Number of rows in the synthetic memory table.
    dim : int
        Vector dimensionality.
    seed : int
        RNG seed for the data generator.

    Returns
    -------
    dict
        Per-experiment summary dict.
    """
    here = _here()
    os.makedirs(here, exist_ok=True)

    print(f"[runner] generating data: n={n}, dim={dim}, seed={seed}")
    t0 = time.perf_counter()
    data = gen(n=n, dim=dim, seed=seed)
    print(f"[runner] data ready in {(time.perf_counter()-t0)*1000:.0f}ms; "
          f"embs shape = {data.embs.shape}, table = {data.table.num_rows} rows")

    summary: Dict[str, Dict] = {}

    # ----- E1 --------------------------------------------------------
    print("[runner] E1: M1 rewrite effect (100 queries)")
    from mosaicopt.experiments.e1_rewrite import run as e1_run
    t0 = time.perf_counter()
    e1 = e1_run(data, n_queries=100)
    print(f"[runner] E1 done in {(time.perf_counter()-t0)*1000:.0f}ms")
    e1.to_csv(os.path.join(here, "e1_rewrite.csv"))
    summary["E1"] = e1.summary()
    print(f"         {summary['E1']}")

    # ----- E2 --------------------------------------------------------
    print("[runner] E2: ANN strategy sweep (15 selectivities x 3 strategies x 3 repeats)")
    from mosaicopt.experiments.e2_ann_strategy import run as e2_run
    t0 = time.perf_counter()
    e2 = e2_run(data, n_repeats=3)
    print(f"[runner] E2 done in {(time.perf_counter()-t0)*1000:.0f}ms")
    e2.to_csv(os.path.join(here, "e2_ann_strategy.csv"))
    e2.plot(here)
    summary["E2"] = e2.summary()
    print(f"         {summary['E2']}")

    # ----- E3 --------------------------------------------------------
    print("[runner] E3: multi-modal vs multi-path orchestrator")
    from mosaicopt.experiments.e3_orchestrator import run as e3_run
    t0 = time.perf_counter()
    e3 = e3_run(data, k=10)
    print(f"[runner] E3 done in {(time.perf_counter()-t0)*1000:.0f}ms")
    e3.to_csv(os.path.join(here, "e3_orchestrator.csv"))
    e3.plot(here)
    summary["E3"] = e3.summary()
    print(f"         {summary['E3']}")

    # ----- E4 --------------------------------------------------------
    print("[runner] E4: Q1->Q6 end-to-end")
    from mosaicopt.experiments.e4_endtoend import run as e4_run
    t0 = time.perf_counter()
    e4 = e4_run(data, k=10)
    print(f"[runner] E4 done in {(time.perf_counter()-t0)*1000:.0f}ms")
    e4.to_csv(os.path.join(here, "e4_endtoend.csv"))
    e4.samples_to_csv(os.path.join(here, "e4_samples.csv"))
    summary["E4"] = e4.summary()
    print(f"         {summary['E4']}")

    # Save summary
    with open(os.path.join(here, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(os.path.join(here, "summary.txt"), "w") as f:
        f.write("MosaicOpt Prototype - Experiment Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset: n={n}, dim={dim}, seed={seed}\n")
        f.write(f"Subsystems: fault-tolerance, perf, frontend, tooling, docs\n")
        f.write(f"Vector columns: emb (768-dim), emb_alt (768-dim)\n")
        f.write(f"BM25 corpus: 10 tokens/row, 30-word vocabulary\n\n")
        f.write("E1: M1 rewrite effect\n")
        for k, v in summary["E1"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nE2: ANN strategy sweep\n")
        for k, v in summary["E2"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nE3: Multi-modal vs multi-path\n")
        for k, v in summary["E3"].items():
            f.write(f"  {k}: {v}\n")
        f.write("\nE4: Q1->Q6 end-to-end\n")
        for k, v in summary["E4"].items():
            f.write(f"  {k}: {v}\n")

    print()
    print("=" * 60)
    print(f"[runner] All experiments complete. Results in {here}")
    print("=" * 60)
    return summary


def summarize() -> Dict:
    """Re-print the saved summary."""
    here = _here()
    with open(os.path.join(here, "summary.json")) as f:
        return json.load(f)


if __name__ == "__main__":
    # Quick run: smaller n to verify everything works
    n = int(os.environ.get("MOSAICOPT_N", "10000"))
    dim = int(os.environ.get("MOSAICOPT_DIM", "768"))
    seed = int(os.environ.get("MOSAICOPT_SEED", "42"))
    run_all(n=n, dim=dim, seed=seed)
