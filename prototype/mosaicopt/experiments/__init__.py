"""Experiments: E1 (M1 rewrite), E2 (CBO strategy), E3 (orchestrator), E4 (Q1-Q6).

The runner (`runner.py`) executes all four experiments in sequence
and writes CSVs + PNGs to `results/`.
"""

from .data_gen import gen, MemoryDataset

__all__ = ["gen", "MemoryDataset"]
