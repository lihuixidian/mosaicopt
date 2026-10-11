# MosaicOpt

An extended relational algebra + Cascades-style optimizer for
multi-modal (vector + text + graph + classical) memory queries,
with single-Λ discipline and substrate-agnostic design
(DuckDB / SQLite / PostgreSQL).

This repository ships the prototype code (under `prototype/`)
and the SIGMOD 2027 paper source (under `docs/SIGMOD/`,
filename `sigconf-attached.tex`). The accompanying
anonymous archive link is on the title page of the paper.

## Layout

```
prototype/                          # Python prototype (~3,200 lines)
    mosaicopt/                      # package
        algebra/                    # extended algebra: σ, μ, Γ, Λ, τ, π
        physical/                   # H_{V,q,τ}, B_{col,range},
                                     # F(R), I_{col,term}, G_{g,d,dir},
                                     # + DuckDB / SQLite / PG substrates
        optimizer/                  # RBO + CBO + statistics + plan
        frontend/                   # SQL/MM-style desugarer
        experiments/                # E1..E6 drivers
    plot_style.py                   # shared bold matplotlib style
    plot_all_v2.py                  # unified figure script
    results/                        # CSV outputs (gitignored)
docs/SIGMOD/                        # paper source
    sigconf-attached.tex            # SIGCONF 2-col paper body
    ref.bib                         # ACM reference format bibliography
    figures/                         # rendered PNG figures
```

## Running the prototype

```bash
cd prototype
pip install -r requirements.txt  # numpy scipy faiss-cpu rank-bm25
                                   # duckdb pyarrow psycopg2-binary
python -m mosaicopt.experiments.runner   # E1..E4 (DuckDB substrate)
python mosaicopt/experiments/e6_cross_substrate.py --n 10000 --runs 3
                                           # E6: DuckDB / SQLite / PG
python plot_all_v2.py                     # regenerate all figures
```

The PG substrate reads `PGHOST` / `PGPORT` / `PGUSER` /
`PGPASSWORD` / `PGDATABASE` from the environment; default is
`localhost:5432`, `postgres/postgres`, `db=mosaicopt`.

## Headline results (n = 10,000)

* **E1** (8 query categories Q1..Q8 × 8 selectivities × 3 datasets):
  MosaicOpt is **8–10× faster than BVB-native with perfect
  recall** (1.00 on every cell, 0 zero-recall cells).
* **E2** (RBO scenarios): M1 = 9.14× on Q7 MV; M6 = additional
  2–3× from classical-filter pushdown; T1 = 1.19×; J1 = 25×
  join-input reduction.
* **E3** (CBO ablation): `default_0.5` picks `flat` correctly
  on every cell at recall 1.0; `with_adaptive` is 2.76× faster
  but loses ~15 % on Q7/Q8.
* **E5** (joint index): SeRF-style index wins 3.30× at
  (s_c=0.05, s_v=0.20); CBO routes through it without new
  algebraic identities.
* **E6** (cross-substrate): DuckDB / SQLite / PostgreSQL give
  QPS within a few percent of each other on every
  (system, category) cell; recall matches across substrates.

## Build the paper

```bash
cd docs/SIGMOD
pdflatex sigconf-attached.tex
bibtex   sigconf-attached
pdflatex sigconf-attached.tex
pdflatex sigconf-attached.tex
```

ACM SIGCONF two-column format, ~19 pages.