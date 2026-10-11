"""Optimizer: RBO + CBO + physical plan.

Modules
-------
rbo     : 7 rewrite rules (M1, M2, M3, V1, J1, T1, G1); runs to fixed point.
cbo     : Cost-based physical-operator selection (4-tuple cost).
plan    : Logical -> physical plan mapping; topK heap / full sort.
"""

from .rbo import rewrite, RULES, RULES_BY_NAME
from .cbo import cbo_pick, cost_vector, CBO_REC, CostVector, STRATEGY_TABLE
from .plan import to_physical, PhysicalPlan, TopK, Sort, ScanP, SelectP, MaterializeP, ExpandP, JoinP, BM25ScanP

__all__ = [
    "rewrite", "RULES", "RULES_BY_NAME",
    "cbo_pick", "cost_vector", "CBO_REC", "CostVector", "STRATEGY_TABLE",
    "to_physical", "PhysicalPlan", "TopK", "Sort", "ScanP", "SelectP",
    "MaterializeP", "ExpandP", "JoinP", "BM25ScanP",
]
