"""MosaicOpt prototype: an algebraic foundation for Agentic memory databases.

Modules
-------
algebra        : logical plan tree (sigma, pi, mu, tau, Lambda, Gamma) + predicates
physical       : faiss (ANN), rank_bm25, NetworkX-free graph, DuckDB substrate
optimizer      : RBO (7 rewrite rules) + CBO (4-tuple cost) + plan
frontend       : SQL/MM-style parser that desugars SQL -> algebra
experiments    : E1-E4 experiments and the runner
results        : CSVs and PNGs produced by the runner
"""

__version__ = "0.1.0"
