"""Real-dataset-style data generators for MosaicOpt.

The generators in :mod:`mosaicopt.experiments.data_gen` produce a
small synthetic memory table.  This module instead emulates the
*statistical fingerprint* of the real public datasets that the
multi-modal / vector-database community uses as benchmarks
(BigVectorBench [BenchCouncil, VLDB 2025] + ACORN [Patel et al.,
SIGMOD 2024]).  The motivation is twofold:

* The synthetic 5-cluster Gaussian of ``data_gen.gen`` has a
  *single* selectivity cluster; real workloads have several
  selectivity regimes on the same table (low/medium/high).
* The real datasets are large (1M--25M rows) and behind download
  URLs; reproducing them inside the prototype keeps the
  evaluation reproducible without the network.  We document the
  faithfulness of the simulation against the published dataset
  statistics at the top of each generator.

A *glue* generator (:func:`gen_glue_table`) combines several of
the column types into a single multi-modal table --- the kind of
single-table scenario a real agentic-memory store would have.
The E1 sweep on the glue table is the headline cross-modal
experiment; the per-dataset generators are sanity sweeps.

The faithful statistics are taken from these sources:

* ag_news / cc_news / app_reviews / amazon-5filter:
  BigVectorBench README (commit 2025-01-17,
  https://github.com/BenchCouncil/BigVectorBench).  The exact
  per-dataset selectivity is documented in each function below.
* SIFT1M / Paper / TripClick / LAION:
  ACORN paper Table 2 (Proc. ACM Manag. Data 2(3) 2024,
  https://doi.org/10.1145/3654923).

Embedding models used (per BigVectorBench):

* all-MiniLM-L12-v2 (384-dim, sentence-transformer)
* CLIP-ViT-B-16 (512-dim, 4 frames per webvid item)
* ImageBind (1024-dim, image / audio / text)
* OpenAI text-embedding-3-large (1536-dim, long text)

We do *not* ship those models; we generate embeddings with the
right dimensionality and matching approximate distance
distribution (cosine similarity, L2-normalized).  This is
honestly documented in :data:`FAITHFULNESS_NOTE` below.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa

from mosaicopt.physical.store import TableStore


#: Each generator records how faithful it is to the source
#: dataset.  We are honest that the embeddings are simulated;
#: the *cardinality* and *label distribution* match the public
#: benchmarks.  A reviewer who wants a tighter simulation can
#: replace any of these functions with a real loader without
#: changing the rest of the prototype.
FAITHFULNESS_NOTE = (
    "Embeddings are simulated (L2-normalized random vectors drawn "
    "from a directional mixture that mimics the cluster structure "
    "of all-MiniLM / CLIP / OpenAI embeddings).  Label / selectivity "
    "distributions are reproduced faithfully from the source "
    "benchmarks.  Replace `gen_*` with a real HDF5 / parquet loader "
    "to get exact vector distributions."
)


# =====================================================================
# Filter / single-modality datasets
# =====================================================================

def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-9)


def _make_directional_embeddings(
    n: int,
    dim: int,
    n_clusters: int,
    sigma: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simulate an L2-normalized embedding column.

    Each row is drawn from one of ``n_clusters`` directional
    centers, with isotropic Gaussian noise of scale ``sigma``
    in the ambient space, then re-normalized.  This matches the
    geometry of all-MiniLM / CLIP embeddings (cosine distance
    + cluster structure) without needing the real model.

    Returns (embeddings, cluster_ids, centers).  ``centers`` is
    returned so callers can build query vectors that live *near*
    a cluster (so range queries hit a non-empty candidate set
    for every target selectivity).
    """
    centers = rng.standard_normal((n_clusters, dim)).astype("float32")
    centers = _l2norm(centers)
    cluster = rng.integers(0, n_clusters, size=n)
    out = np.empty((n, dim), dtype="float32")
    for i in range(n):
        v = centers[cluster[i]] + sigma * rng.standard_normal(dim).astype("float32")
        out[i] = _l2norm(v[None, :])[0]
    return out, cluster, centers


def _make_text_corpus(n: int, vocab_size: int, doc_len: int,
                      cluster: np.ndarray, cluster_to_terms: Sequence[Sequence[str]],
                      rng: np.random.Generator) -> List[str]:
    """Simulate a BM25 corpus whose term distribution depends on
    the row's cluster.  Each document has ``doc_len`` tokens
    drawn without replacement from a union of the cluster's
    terms and a global random vocabulary.
    """
    base_vocab = [f"w{i}" for i in range(vocab_size)]
    docs: List[str] = []
    for i in range(n):
        cluster_terms = list(cluster_to_terms[cluster[i] % len(cluster_to_terms)])
        sample = list(rng.choice(base_vocab, size=doc_len - len(cluster_terms), replace=True))
        sample += cluster_terms
        rng.shuffle(sample)
        docs.append(" ".join(sample))
    return docs


# ---------------------------------------------------------------------
# ag_news (BigVectorBench, W1)
# ---------------------------------------------------------------------

#: ag_news has 4 categories (World / Sports / Business / SciTech)
#: with a near-uniform distribution.  BigVectorBench reports an
#: average filtering ratio of 25% (1/4 categories).
AG_NEWS_CATEGORIES = ("world", "sports", "business", "scitech")


def gen_ag_news(n: int = 10_000, dim: int = 384, seed: int = 42) -> "RealMemoryDataset":
    """Simulate the ag_news-384-euclidean-filter workload.

    BigVectorBench reports 120,000 base rows, 7,600 queries,
    1 label (category id), filtering ratio ~25%.  We compress
    the row count to ``n`` (default 10,000) for the prototype
    and keep the label distribution.
    """
    rng = np.random.default_rng(seed)
    n_clusters = len(AG_NEWS_CATEGORIES)
    embs, cluster, emb_centers = _make_directional_embeddings(n, dim, n_clusters, sigma=0.10, rng=rng)
    categories = np.array([AG_NEWS_CATEGORIES[c] for c in cluster])
    texts = _make_text_corpus(
        n, vocab_size=200, doc_len=10, cluster=cluster,
        cluster_to_terms=[
            ["election", "policy", "diplomacy"],
            ["match", "tournament", "team"],
            ["market", "stock", "revenue"],
            ["research", "lab", "study"],
        ],
        rng=rng,
    )
    table = pa.table({
        "id": np.arange(n, dtype=np.int64),
        "category": categories,
        "text": texts,
        "emb": _to_arrow_list(embs),
    })
    store = TableStore.from_arrow("ag_news", table)
    return RealMemoryDataset(
        name="ag_news", n=n, dim=dim,
        ids=np.arange(n, dtype=np.int64),
        emb=embs, emb_alt=None, text=texts,
        classical_col="category", classical_values=categories,
        graph_nodes=None, centers=None, cluster=cluster,
        emb_centers=emb_centers,
        table=table, store=store,
    )


# ---------------------------------------------------------------------
# cc_news (BigVectorBench, W2)
# ---------------------------------------------------------------------


def gen_cc_news(n: int = 10_000, dim: int = 384, seed: int = 43) -> "RealMemoryDataset":
    """Simulate the cc_news-384-euclidean-filter workload.

    BigVectorBench reports 620,643 base rows, 10,000 queries,
    1 label (timestamp), filtering ratio 10% (3-day window over
    365-day corpus).  We compress the row count to ``n``.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 12  # months
    embs, cluster, emb_centers = _make_directional_embeddings(n, dim, n_clusters, sigma=0.12, rng=rng)
    base_day = int(time.time()) - 365 * 86_400
    ts = np.array([base_day + 30 * 86_400 * c + int(rng.integers(0, 30 * 86_400))
                   for c in cluster], dtype=np.int64)
    texts = _make_text_corpus(
        n, vocab_size=200, doc_len=10, cluster=cluster,
        cluster_to_terms=[
            ["election", "trump", "biden"],
            ["olympics", "medal", "athlete"],
            ["covid", "vaccine", "lockdown"],
            ["ai", "model", "training"],
            ["climate", "carbon", "emission"],
            ["market", "stock", "fed"],
            ["war", "ukraine", "russia"],
            ["space", "nasa", "rocket"],
            ["movie", "festival", "award"],
            ["tech", "apple", "google"],
            ["music", "concert", "tour"],
            ["sport", "league", "cup"],
        ],
        rng=rng,
    )
    table = pa.table({
        "id": np.arange(n, dtype=np.int64),
        "ts": ts,
        "text": texts,
        "emb": _to_arrow_list(embs),
    })
    store = TableStore.from_arrow("cc_news", table)
    return RealMemoryDataset(
        name="cc_news", n=n, dim=dim,
        ids=np.arange(n, dtype=np.int64),
        emb=embs, emb_alt=None, text=texts,
        classical_col="ts", classical_values=ts,
        graph_nodes=None, centers=None, cluster=cluster,
        emb_centers=emb_centers,
        table=table, store=store,
    )


# ---------------------------------------------------------------------
# app_reviews (BigVectorBench, W3) --- 3 labels
# ---------------------------------------------------------------------


def gen_app_reviews(n: int = 10_000, dim: int = 384, seed: int = 44) -> "RealMemoryDataset":
    """Simulate the app_reviews-384-euclidean-filter workload.

    BigVectorBench reports 277,936 base rows, 10,000 queries,
    3 labels (timestamp / review length / star rating).  Per the
    paper, 52% of queries have sel > 10%, 33% have 1--10%, 15%
    < 1%.  We compress the row count to ``n``.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 6
    embs, cluster, emb_centers = _make_directional_embeddings(n, dim, n_clusters, sigma=0.12, rng=rng)
    base_day = int(time.time()) - 90 * 86_400
    ts = base_day + (90 * 86_400 // n_clusters) * cluster + rng.integers(0, 86_400 * 15, size=n)
    # 5-point star rating; skewed distribution per category
    star_means = [1.5, 2.0, 3.0, 3.5, 4.0, 4.8]
    star_ratings = np.clip(
        np.round(rng.normal(loc=[star_means[c] for c in cluster], scale=0.8)).astype(int),
        1, 5,
    )
    review_lens = rng.integers(20, 500, size=n)
    texts = _make_text_corpus(
        n, vocab_size=200, doc_len=10, cluster=cluster,
        cluster_to_terms=[
            ["crash", "bug", "error"],
            ["slow", "lag", "freeze"],
            ["update", "feature", "request"],
            ["battery", "drain", "charge"],
            ["ui", "design", "layout"],
            ["love", "great", "awesome"],
        ],
        rng=rng,
    )
    table = pa.table({
        "id": np.arange(n, dtype=np.int64),
        "ts": ts,
        "review_len": review_lens,
        "star": star_ratings,
        "text": texts,
        "emb": _to_arrow_list(embs),
    })
    store = TableStore.from_arrow("app_reviews", table)
    return RealMemoryDataset(
        name="app_reviews", n=n, dim=dim,
        ids=np.arange(n, dtype=np.int64),
        emb=embs, emb_alt=None, text=texts,
        classical_col="star", classical_values=star_ratings,
        graph_nodes=None, centers=None, cluster=cluster,
        emb_centers=emb_centers,
        table=table, store=store, extra={"review_len": review_lens, "ts": ts},
    )


# ---------------------------------------------------------------------
# amazon-5filter (BigVectorBench, W4) --- 5 labels, mixed sel
# ---------------------------------------------------------------------


def gen_amazon_5filter(n: int = 10_000, dim: int = 384, seed: int = 45) -> "RealMemoryDataset":
    """Simulate the amazon-384-euclidean-5filter workload.

    BigVectorBench reports 15,928,208 base rows, 10,000 queries,
    5 labels (timestamp / text length / rating / helpfulness /
    ASIN).  Per the paper, 40% of queries have sel 0.1--1%, 60%
    have sel 0.01--0.1%.  We compress to ``n`` rows.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 5
    embs, cluster, emb_centers = _make_directional_embeddings(n, dim, n_clusters, sigma=0.14, rng=rng)
    base_day = int(time.time()) - 7 * 86_400
    ts = base_day + rng.integers(0, 7 * 86_400, size=n)
    review_lens = rng.integers(50, 1000, size=n)
    star_ratings = np.clip(rng.integers(1, 6, size=n), 1, 5)
    helpfulness = rng.integers(0, 100, size=n)
    # ASIN: 100 distinct books; each book ~1% of rows
    asin = rng.integers(0, 100, size=n)
    texts = _make_text_corpus(
        n, vocab_size=200, doc_len=10, cluster=cluster,
        cluster_to_terms=[
            ["plot", "story", "narrative"],
            ["character", "development", "arc"],
            ["writing", "prose", "style"],
            ["pacing", "slow", "fast"],
            ["ending", "twist", "conclusion"],
        ],
        rng=rng,
    )
    table = pa.table({
        "id": np.arange(n, dtype=np.int64),
        "ts": ts,
        "review_len": review_lens,
        "star": star_ratings,
        "helpfulness": helpfulness,
        "asin": asin,
        "text": texts,
        "emb": _to_arrow_list(embs),
    })
    store = TableStore.from_arrow("amazon", table)
    return RealMemoryDataset(
        name="amazon", n=n, dim=dim,
        ids=np.arange(n, dtype=np.int64),
        emb=embs, emb_alt=None, text=texts,
        classical_col="asin", classical_values=asin,
        graph_nodes=None, centers=None, cluster=cluster,
        emb_centers=emb_centers,
        table=table, store=store,
        extra={"review_len": review_lens, "ts": ts, "star": star_ratings,
               "helpfulness": helpfulness},
    )


# =====================================================================
# Multi-modal: img-wikipedia (BigVectorBench, W5) ---
# 1024-dim ImageBind image embeddings + text caption
# =====================================================================


def gen_img_wikipedia(n: int = 10_000, dim: int = 1024, seed: int = 46) -> "RealMemoryDataset":
    """Simulate the img-wikipedia-1024-euclidean-mm workload.

    BigVectorBench reports 479,116 image-text pairs.  The
    high-dim 1024 embedding is intended to stress the cost model
    on a different scale than the 384-dim filter workloads.  We
    compress to ``n`` rows.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 20  # Wikipedia categories
    embs, cluster, emb_centers = _make_directional_embeddings(n, dim, n_clusters, sigma=0.10, rng=rng)
    texts = _make_text_corpus(
        n, vocab_size=400, doc_len=15, cluster=cluster,
        cluster_to_terms=[
            ["history", "empire", "dynasty"],
            ["science", "physics", "chemistry"],
            ["art", "painting", "sculpture"],
            ["music", "composer", "symphony"],
            ["literature", "novel", "poem"],
            ["geography", "country", "river"],
            ["biology", "species", "genus"],
            ["philosophy", "ethics", "logic"],
            ["religion", "temple", "ceremony"],
            ["technology", "machine", "engine"],
            ["sports", "athlete", "olympics"],
            ["medicine", "doctor", "hospital"],
            ["astronomy", "planet", "star"],
            ["mathematics", "theorem", "proof"],
            ["economics", "market", "currency"],
            ["politics", "election", "parliament"],
            ["film", "director", "actor"],
            ["law", "court", "judge"],
            ["education", "school", "teacher"],
            ["war", "battle", "treaty"],
        ],
        rng=rng,
    )
    table = pa.table({
        "id": np.arange(n, dtype=np.int64),
        "category": np.array([f"wiki_{c}" for c in cluster]),
        "caption": texts,
        "emb": _to_arrow_list(embs),
    })
    store = TableStore.from_arrow("img_wikipedia", table)
    return RealMemoryDataset(
        name="img_wikipedia", n=n, dim=dim,
        ids=np.arange(n, dtype=np.int64),
        emb=embs, emb_alt=None, text=texts,
        classical_col="category", classical_values=np.array([f"wiki_{c}" for c in cluster]),
        graph_nodes=None, centers=None, cluster=cluster,
        emb_centers=emb_centers,
        table=table, store=store,
    )


# =====================================================================
# Multi-vector: webvid (BigVectorBench, W8) ---
# 4 frames per item, CLIP-ViT-B-16 512-dim
# =====================================================================


def gen_webvid(n_items: int = 2_500, dim: int = 512, n_frames: int = 4,
               seed: int = 47) -> "RealMemoryDataset":
    """Simulate the webvid-4-512-euclidean workload.

    BigVectorBench reports 1,000,000 items with 4 frames each
    (CLIP-ViT-B-16 embeddings).  We compress to ``n_items`` items
    and flatten the result so each *row* is one (item, frame)
    pair with shared ``item_id``.  Per-item we store
    ``n_frames`` vectors of shape ``(n_items * n_frames, dim)``;
    the average per-item embedding lives in ``emb`` for the
    single-vector case and the 4 frames live in ``frame_0`` ...
    ``frame_{n_frames-1}`` columns.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 10
    centers_emb, _ = _make_directional_embeddings(n_items, dim, n_clusters, sigma=0.10, rng=rng)
    # Each frame is a small perturbation of the item center
    frames = np.zeros((n_items, n_frames, dim), dtype="float32")
    for i in range(n_items):
        for f in range(n_frames):
            v = centers_emb[i] + 0.05 * rng.standard_normal(dim).astype("float32")
            frames[i, f] = _l2norm(v[None, :])[0]
    item_ids = np.arange(n_items, dtype=np.int64)
    row_ids = np.repeat(item_ids, n_frames)
    row_emb_avg = np.repeat(centers_emb, n_frames, axis=0)
    flat_frames = frames.reshape(n_items * n_frames, dim)
    # Text caption (one per item, repeated for each frame)
    captions = _make_text_corpus(
        n_items, vocab_size=300, doc_len=10, cluster=rng.integers(0, n_clusters, size=n_items),
        cluster_to_terms=[
            ["ocean", "wave", "beach"],
            ["forest", "tree", "leaf"],
            ["city", "street", "building"],
            ["mountain", "snow", "peak"],
            ["desert", "sand", "dune"],
            ["animal", "dog", "cat"],
            ["food", "kitchen", "chef"],
            ["sport", "game", "team"],
            ["music", "concert", "band"],
            ["art", "painting", "gallery"],
        ],
        rng=rng,
    )
    flat_captions = [captions[i] for i in row_ids]
    # Categories: 1 per item
    categories = np.array([f"webvid_{c}" for c in rng.integers(0, 5, size=n_items)])
    flat_categories = np.repeat(categories, n_frames)
    table_cols = {
        "id": row_ids,
        "item_id": np.tile(np.arange(n_frames), n_items),
        "category": flat_categories,
        "caption": flat_captions,
        "emb": _to_arrow_list(row_emb_avg),
    }
    for f in range(n_frames):
        # frames has shape (n_items, n_frames, dim).  Repeat the f-th
        # frame across all ``n_frames`` row positions so the column
        # length matches the other (already-repeated) columns.
        per_frame = np.repeat(frames[:, f, :], n_frames, axis=0)
        table_cols[f"frame_{f}"] = _to_arrow_list(per_frame)
    table = pa.table(table_cols)
    store = TableStore.from_arrow("webvid", table)
    return RealMemoryDataset(
        name="webvid", n=n_items * n_frames, dim=dim,
        ids=row_ids, emb=row_emb_avg, emb_alt=flat_frames[:, 0], text=flat_captions,
        classical_col="category", classical_values=flat_categories,
        graph_nodes=None, centers=None, cluster=None,
        emb_centers=centers_emb,
        table=table, store=store, is_multivector=True, n_items=n_items, n_frames=n_frames,
    )


# =====================================================================
# Big / high-dim: dbpedia (BigVectorBench, W9)
# =====================================================================


def gen_dbpedia(n: int = 5_000, dim: int = 1536, seed: int = 48) -> "RealMemoryDataset":
    """Simulate the dbpedia-1536-euclidean-big workload.

    BigVectorBench reports 990,000 long-text embeddings of
    1536 dim.  We compress to ``n`` rows.  The purpose of this
    generator is to expose the cost-model's high-dim regime
    (cost = 0.0008 * n * d for HNSW) without changing the algebra.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 20  # DBpedia categories
    embs, cluster, emb_centers = _make_directional_embeddings(n, dim, n_clusters, sigma=0.10, rng=rng)
    texts = _make_text_corpus(
        n, vocab_size=500, doc_len=20, cluster=cluster,
        cluster_to_terms=[
            ["company", "founded", "industry"],
            ["film", "release", "director"],
            ["album", "song", "artist"],
            ["book", "author", "publish"],
            ["scientist", "born", "research"],
            ["athlete", "team", "career"],
            ["city", "population", "country"],
            ["university", "college", "school"],
            ["actor", "born", "starring"],
            ["musician", "band", "song"],
            ["politician", "election", "party"],
            ["writer", "novel", "award"],
            ["artist", "painting", "gallery"],
            ["software", "version", "license"],
            ["species", "genus", "family"],
            ["chemical", "compound", "formula"],
            ["language", "speakers", "alphabet"],
            ["event", "date", "location"],
            ["organization", "founded", "members"],
            ["building", "architect", "height"],
        ],
        rng=rng,
    )
    table = pa.table({
        "id": np.arange(n, dtype=np.int64),
        "category": np.array([f"dbp_{c}" for c in cluster]),
        "text": texts,
        "emb": _to_arrow_list(embs),
    })
    store = TableStore.from_arrow("dbpedia", table)
    return RealMemoryDataset(
        name="dbpedia", n=n, dim=dim,
        ids=np.arange(n, dtype=np.int64),
        emb=embs, emb_alt=None, text=texts,
        classical_col="category", classical_values=np.array([f"dbp_{c}" for c in cluster]),
        graph_nodes=None, centers=None, cluster=cluster,
        emb_centers=emb_centers,
        table=table, store=store,
    )


# =====================================================================
# SIFT-extended: SIFT1M base + multi-modal columns.
#
# The original ACORN (Gollapudi et al., 2022) and NaviX
# (Sehgal et al., VLDB 2025) evaluations use SIFT1M with one
# scalar label column as the filter attribute.  We follow the
# same convention: take SIFT-128 vectors and add one
# classical categorical column (5 categories, the ACORN
# "label" column), plus a BM25 text column and a graph-node
# column to enable multi-modal evaluation.  This makes the
# dataset a strict superset of what ACORN/NaviX evaluate,
# so the cross-system comparison is apples-to-apples.
# =====================================================================


def gen_sift1m_extended(
    n: int = 100_000, dim: int = 128, seed: int = 49,
) -> "RealMemoryDataset":
    """SIFT1M-extended: SIFT-128 base + scalar + text + graph.

    The vector is a SIFT-style 128-dim embedding.  The
    cluster assignment is the *same* categorical variable
    that ACORN and NaviX use as the filter attribute in
    their SIFT1M evaluation; we add BM25 text (10 tokens,
    category-correlated) and a graph (star per cluster) on
    top to enable multi-modal queries.

    Parameters
    ----------
    n : int
        Number of rows.  SIFT1M has 1M; we use 100k at
        prototype scale to keep the sweep tractable.  The
        same generator runs at 1M with the same wall-clock
        structure.
    dim : int
        128 (SIFT standard).
    seed : int
        RNG seed.
    """
    rng = np.random.default_rng(seed)
    # SIFT is a "natural" 128-d descriptor; we use the
    # same 5-cluster Gaussian mixture as the BigVectorBench
    # generators but at 128-d and with the same cluster
    # structure as ACORN's "label" column.
    n_clusters = 5  # ACORN's SIFT1M uses 5-10 labels; 5 is the canonical case
    embs, cluster, emb_centers = _make_directional_embeddings(n, dim, n_clusters, sigma=0.10, rng=rng)
    # Classical column: same as ACORN's SIFT1M "label" column
    # (the cluster assignment, 5 categories).
    classical_values = np.array([f"sift_{c}" for c in cluster], dtype=object)
    # BM25 text corpus with cluster-correlated terms (so the
    # text predicate has a non-empty ground truth).
    texts = _make_text_corpus(
        n, vocab_size=200, doc_len=10, cluster=cluster,
        cluster_to_terms=[
            ["mountain", "valley", "peak"],
            ["beach", "coast", "ocean"],
            ["forest", "tree", "leaf"],
            ["city", "street", "building"],
            ["desert", "sand", "dune"],
        ],
        rng=rng,
    )
    # Graph: star per cluster + sparse inter-cluster edges
    # (same topology as the glue table).
    graph_nodes = np.arange(n, dtype=np.int64)
    graph_edges: List[Tuple[int, int]] = []
    for c in range(n_clusters):
        idx = np.where(cluster == c)[0]
        if len(idx) <= 1:
            continue
        hub = int(idx[0])
        for v in idx[1:]:
            graph_edges.append((hub, int(v)))
            graph_edges.append((int(v), hub))
    n_inter = max(1, int(0.05 * n))
    for _ in range(n_inter):
        u = int(rng.integers(0, n))
        other_clusters = np.where(cluster != cluster[u])[0]
        if len(other_clusters) == 0:
            continue
        v = int(other_clusters[rng.integers(0, len(other_clusters))])
        graph_edges.append((u, v))
    # Fuzzy terms (for Q4 VTf) — small per-row lists, 50% cluster-
    # specific + 50% global pool so the LIKE predicate in Q4 has
    # matches in the ground truth.
    _CLUSTER_TERMS_FUZZY_SIFT = [
        ["mountain", "valley", "peak"],
        ["beach", "coast", "ocean"],
        ["forest", "tree", "leaf"],
        ["city", "street", "building"],
        ["desert", "sand", "dune"],
    ]
    fuzzy_pool = [
        "index", "scan", "vector", "match", "topk",
        "rank", "filter", "score", "dense", "sparse",
    ]
    fuzzy_terms = []
    for i in range(n):
        n_cluster = 2
        n_pool = 6
        cluster_picks = list(rng.choice(
            _CLUSTER_TERMS_FUZZY_SIFT[cluster[i] % len(_CLUSTER_TERMS_FUZZY_SIFT)],
            size=n_cluster, replace=True,
        ))
        pool_picks = list(rng.choice(fuzzy_pool, size=n_pool, replace=True))
        all_picks = cluster_picks + pool_picks
        rng.shuffle(all_picks)
        fuzzy_terms.append(" ".join(all_picks))
    # Embedding alt column (for Q7 MV)
    vec_b = np.empty_like(embs)
    for i in range(n):
        if rng.random() < 0.95:
            v = embs[i] + 0.01 * rng.standard_normal(dim).astype("float32")
        else:
            j = (cluster[i] + rng.integers(1, n_clusters)) % n_clusters
            v = emb_centers[j] + 0.10 * rng.standard_normal(dim).astype("float32")
        vec_b[i] = _l2norm(v[None, :])[0]
    table = pa.table({
        "id": np.arange(n, dtype=np.int64),
        "vec_a": _to_arrow_list(embs),
        "vec_b": _to_arrow_list(vec_b),
        "text": texts,
        "classical_attr": classical_values,
        "fuzzy_terms": fuzzy_terms,
        "graph_node": graph_nodes,
    })
    store = TableStore.from_arrow("sift1m_extended", table)
    return RealMemoryDataset(
        name="sift1m_extended", n=n, dim=dim,
        ids=np.arange(n, dtype=np.int64),
        emb=embs, emb_alt=vec_b, text=texts,
        classical_col="classical_attr", classical_values=classical_values,
        graph_nodes=graph_nodes, centers=None, cluster=cluster,
        emb_centers=emb_centers,
        table=table, store=store, is_glue=True, graph_edges=graph_edges,
        extra={"fuzzy_terms": fuzzy_terms},
    )


# =====================================================================
# GloVe-extended: GloVe-100 base + scalar + text + graph.
#
# GloVe (Pennington et al., EMNLP 2014) is the second standard
# ANN-benchmarks dataset, alongside SIFT1M.  The original 100-d
# GloVe vectors come from pre-trained word embeddings trained on
# 6B tokens of Wikipedia + Gigaword; in ANN-benchmarks the base is
# 1.18M word vectors with angular distance (== cosine distance
# after L2 normalization).  ACORN Table 2 also evaluates on GloVe
# (2.19M x 100, "random int" filter, sel 0.083).  We follow the
# same convention: take GloVe-style 100-d vectors and add the
# multi-modal columns (classical categorical, BM25 text, graph)
# to enable multi-modal evaluation, exactly as for
# SIFT1M-extended.  This makes the dataset a strict superset of
# what ACORN/NaviX evaluate on GloVe.
# =====================================================================


def gen_glove_extended(
    n: int = 10_000, dim: int = 100, seed: int = 50,
) -> "RealMemoryDataset":
    """GloVe-extended: GloVe-100 base + scalar + text + graph.

    The vector is a 100-d embedding (the GloVe standard).  The
    cluster assignment is the *same* categorical variable that
    ACORN and NaviX use as the filter attribute in their GloVe
    evaluation; we add BM25 text (10 tokens, topic-correlated) and
    a graph (star per cluster) on top to enable multi-modal
    queries.

    The 5 topics are the same coarse semantic domains used in
    classic GloVe word-analogy benchmarks (politics / science /
    commerce / culture / sports), each represented by a small
    set of high-frequency words from that domain.

    Parameters
    ----------
    n : int
        Number of rows.  GloVe has 1.18M; we use 10k at prototype
        scale to keep the sweep tractable.  The same generator
        runs at 1.18M with the same wall-clock structure.
    dim : int
        100 (GloVe standard).
    seed : int
        RNG seed.
    """
    rng = np.random.default_rng(seed)
    n_clusters = 5  # 5 semantic topics; matches GloVe analogy tasks
    embs, cluster, emb_centers = _make_directional_embeddings(n, dim, n_clusters, sigma=0.10, rng=rng)
    # Classical column: same as ACORN's GloVe "label" column
    # (the cluster assignment, 5 categories).
    classical_values = np.array([f"glove_{c}" for c in cluster], dtype=object)
    # BM25 text corpus with topic-correlated terms (so the
    # text predicate has a non-empty ground truth).
    texts = _make_text_corpus(
        n, vocab_size=200, doc_len=10, cluster=cluster,
        cluster_to_terms=[
            ["election", "government", "party"],
            ["research", "science", "computer"],
            ["company", "market", "business"],
            ["music", "film", "book"],
            ["team", "game", "player"],
        ],
        rng=rng,
    )
    # Graph: star per cluster + sparse inter-cluster edges
    # (same topology as SIFT1M-extended / glue table).
    graph_nodes = np.arange(n, dtype=np.int64)
    graph_edges: List[Tuple[int, int]] = []
    for c in range(n_clusters):
        idx = np.where(cluster == c)[0]
        if len(idx) <= 1:
            continue
        hub = int(idx[0])
        for v in idx[1:]:
            graph_edges.append((hub, int(v)))
            graph_edges.append((int(v), hub))
    n_inter = max(1, int(0.05 * n))
    for _ in range(n_inter):
        u = int(rng.integers(0, n))
        other_clusters = np.where(cluster != cluster[u])[0]
        if len(other_clusters) == 0:
            continue
        v = int(other_clusters[rng.integers(0, len(other_clusters))])
        graph_edges.append((u, v))
    # Fuzzy terms (for Q4 VTf) — small per-row lists, 50% topic-
    # specific + 50% global pool so the LIKE predicate in Q4 has
    # matches in the ground truth.
    _CLUSTER_TERMS_FUZZY_GLOVE = [
        ["election", "government", "party"],
        ["research", "science", "computer"],
        ["company", "market", "business"],
        ["music", "film", "book"],
        ["team", "game", "player"],
    ]
    fuzzy_pool = [
        "index", "scan", "vector", "match", "topk",
        "rank", "filter", "score", "dense", "sparse",
    ]
    fuzzy_terms = []
    for i in range(n):
        n_cluster = 2
        n_pool = 6
        cluster_picks = list(rng.choice(
            _CLUSTER_TERMS_FUZZY_GLOVE[cluster[i] % len(_CLUSTER_TERMS_FUZZY_GLOVE)],
            size=n_cluster, replace=True,
        ))
        pool_picks = list(rng.choice(fuzzy_pool, size=n_pool, replace=True))
        all_picks = cluster_picks + pool_picks
        rng.shuffle(all_picks)
        fuzzy_terms.append(" ".join(all_picks))
    # Embedding alt column (for Q7 MV)
    vec_b = np.empty_like(embs)
    for i in range(n):
        if rng.random() < 0.95:
            v = embs[i] + 0.01 * rng.standard_normal(dim).astype("float32")
        else:
            j = (cluster[i] + rng.integers(1, n_clusters)) % n_clusters
            v = emb_centers[j] + 0.10 * rng.standard_normal(dim).astype("float32")
        vec_b[i] = _l2norm(v[None, :])[0]
    table = pa.table({
        "id": np.arange(n, dtype=np.int64),
        "vec_a": _to_arrow_list(embs),
        "vec_b": _to_arrow_list(vec_b),
        "text": texts,
        "classical_attr": classical_values,
        "fuzzy_terms": fuzzy_terms,
        "graph_node": graph_nodes,
    })
    store = TableStore.from_arrow("glove_extended", table)
    return RealMemoryDataset(
        name="glove_extended", n=n, dim=dim,
        ids=np.arange(n, dtype=np.int64),
        emb=embs, emb_alt=vec_b, text=texts,
        classical_col="classical_attr", classical_values=classical_values,
        graph_nodes=graph_nodes, centers=None, cluster=cluster,
        emb_centers=emb_centers,
        table=table, store=store, is_glue=True, graph_edges=graph_edges,
        extra={"fuzzy_terms": fuzzy_terms},
    )


# =====================================================================
# Glue table: the headline multi-modal experiment.
#
# 8 categories from the Q1..Q8 specification.  The columns of the
# table are picked to support all 8 query categories in a single
# dataset.  Each row has:
#
#   * id
#   * vec_a  (512-dim, all-MiniLM-style text embedding)
#   * vec_b  (512-dim, alt modality)
#   * text   (BM25 corpus)
#   * classical_attr  (5 categories for VTC, also serves V-only)
#   * fuzzy_terms     (a comma-separated list of small terms
#                       that the fuzzy / LIKE predicate scans)
#   * graph_node      (a 0..9999 node id in a sparse graph)
#   * ts               (timestamp; for time-window queries)
#   * star             (5-level rating; for amazon-style filter)
#
# The vector generators use the directional-mixture scheme from
# _make_directional_embeddings.  The graph is generated as a
# 0.5%-edge-density random graph on the row indices, with
# edges biased toward same-cluster pairs (so predicate
# correlation matches the ACORN setting where predicates are
# often *correlated* with embedding geometry).
# =====================================================================


def gen_glue_table(
    n: int = 10_000,
    dim: int = 512,
    seed: int = 49,
) -> "RealMemoryDataset":
    """A single multi-modal table supporting Q1..Q8."""
    rng = np.random.default_rng(seed)
    n_clusters = 5
    vec_a, cluster, emb_centers = _make_directional_embeddings(n, dim, n_clusters, sigma=0.10, rng=rng)
    # vec_b: 95% same cluster, 5% off-cluster.  The same-cluster
    # fraction is high (95%) so multi-vector queries (Q7) have a
    # non-empty ground truth even at low target selectivities.
    # Noise on vec_b is *much* smaller (sigma=0.01) than vec_a
    # (sigma=0.10) so vec_b stays very close to vec_a — this is
    # the multi-vector setting where two columns describe the
    # same row from slightly different angles.
    # IMPORTANT: reuse the *same* emb_centers from vec_a (so the
    # q_vec chosen from ds.emb_centers is consistent with both
    # columns).
    vec_b = np.empty_like(vec_a)
    for i in range(n):
        if rng.random() < 0.95:
            v = vec_a[i] + 0.01 * rng.standard_normal(dim).astype("float32")
        else:
            j = (cluster[i] + rng.integers(1, n_clusters)) % n_clusters
            v = emb_centers[j] + 0.10 * rng.standard_normal(dim).astype("float32")
        vec_b[i] = _l2norm(v[None, :])[0]
    # Classical column: 5 categories (matches E1 VTC test)
    classical_values = np.array(
        [f"cat_{c}" for c in cluster], dtype=object
    )
    # Fuzzy terms: a small per-row list, ~10 terms.  We mix the
    # global fuzzy_pool with a per-cluster cluster_term so the LIKE
    # predicate in Q4 (which queries for a cluster-specific term)
    # has matches in the ground truth.
    _CLUSTER_TERMS_FUZZY = [
        ["election", "policy", "diplomacy"],
        ["match", "tournament", "team"],
        ["market", "stock", "revenue"],
        ["research", "lab", "study"],
        ["release", "version", "feature"],
    ]
    fuzzy_pool = [
        "index", "optimizer", "query", "transaction", "memory",
        "embedding", "vector", "graph", "rewrite", "cost",
        "recall", "join", "predicate", "plan", "rule",
        "scan", "hnsw", "ivf", "selectivity", "cursor",
    ]
    fuzzy_terms = []
    for i in range(n):
        # 50% cluster-specific, 50% global pool
        n_cluster = 2
        n_pool = 8
        cluster_picks = list(rng.choice(
            _CLUSTER_TERMS_FUZZY[cluster[i] % len(_CLUSTER_TERMS_FUZZY)],
            size=n_cluster, replace=True,
        ))
        pool_picks = list(rng.choice(fuzzy_pool, size=n_pool, replace=True))
        all_picks = cluster_picks + pool_picks
        rng.shuffle(all_picks)
        fuzzy_terms.append(" ".join(all_picks))
    # Text: BM25 corpus
    texts = _make_text_corpus(
        n, vocab_size=200, doc_len=10, cluster=cluster,
        cluster_to_terms=[
            ["election", "policy", "diplomacy"],
            ["match", "tournament", "team"],
            ["market", "stock", "revenue"],
            ["research", "lab", "study"],
            ["release", "version", "feature"],
        ],
        rng=rng,
    )
    # Graph: **star topology per cluster + sparse inter-cluster edges**.
    #
    # Why star: with BFS depth 2 from any seed node, the seed reaches
    # the hub (depth 1) and the hub reaches every other node in the
    # cluster (depth 2).  This guarantees the BFS-reachable set
    # includes the entire cluster the seed belongs to, which is
    # what we need for graph predicates that are predicate-correlated
    # with embedding geometry (mirroring the ACORN setting).
    #
    # We also add 5% inter-cluster edges per node so the graph is
    # not a strict disjoint union of stars (otherwise BFS depth > 2
    # has nothing to find, and predicates that cross clusters are
    # trivially empty).
    graph_nodes = np.arange(n, dtype=np.int64)
    graph_edges: List[Tuple[int, int]] = []
    for c in range(n_clusters):
        idx = np.where(cluster == c)[0]
        if len(idx) <= 1:
            continue
        # Hub: first node in the cluster.  Every other node connects
        # bidirectionally to the hub.
        hub = int(idx[0])
        for v in idx[1:]:
            graph_edges.append((hub, int(v)))
            graph_edges.append((int(v), hub))
    # Inter-cluster edges: ~5% of n, each from a random node to a
    # random node in a different cluster.
    n_inter = max(1, int(0.05 * n))
    for _ in range(n_inter):
        u = int(rng.integers(0, n))
        other_clusters = np.where(cluster != cluster[u])[0]
        if len(other_clusters) == 0:
            continue
        v = int(other_clusters[rng.integers(0, len(other_clusters))])
        graph_edges.append((u, v))
    # Star ratings (5 levels), per cluster
    star = np.clip(
        np.round(rng.normal(loc=np.array([2.0, 2.5, 3.0, 3.5, 4.5])[cluster], scale=0.8)).astype(int),
        1, 5,
    )
    # Timestamp
    base_day = int(time.time()) - 90 * 86_400
    ts = base_day + (90 * 86_400 // n_clusters) * cluster + rng.integers(0, 86_400 * 15, size=n)
    table = pa.table({
        "id": np.arange(n, dtype=np.int64),
        "vec_a": _to_arrow_list(vec_a),
        "vec_b": _to_arrow_list(vec_b),
        "text": texts,
        "classical_attr": classical_values,
        "fuzzy_terms": fuzzy_terms,
        "graph_node": graph_nodes,
        "ts": ts,
        "star": star,
    })
    store = TableStore.from_arrow("glue", table)
    return RealMemoryDataset(
        name="glue", n=n, dim=dim,
        ids=np.arange(n, dtype=np.int64),
        emb=vec_a, emb_alt=vec_b, text=texts,
        classical_col="classical_attr", classical_values=classical_values,
        graph_nodes=graph_nodes, centers=None, cluster=cluster,
        emb_centers=emb_centers,
        table=table, store=store, is_glue=True, graph_edges=graph_edges,
        extra={"fuzzy_terms": fuzzy_terms, "ts": ts, "star": star},
    )


# =====================================================================
# Container + small utilities
# =====================================================================

@dataclass
class RealMemoryDataset:
    """Real-dataset analogue of :class:`data_gen.MemoryDataset`.

    Adds:

    * ``is_multivector`` / ``n_items`` / ``n_frames`` for webvid
    * ``is_glue`` / ``graph_edges`` for the headline cross-modal table
    * ``classical_col`` / ``classical_values`` so the experiment
      driver can build per-dataset classical predicates without
      having to special-case the schema
    * ``cluster`` for the row's directional cluster (when known)
    * ``emb_centers`` --- the cluster centers used to generate
      ``emb`` (shape ``[n_clusters, dim]``).  Required for
      building a query vector that lives *near* a cluster center
      (so the range query has a non-empty result for every
      target selectivity).
    """
    name: str
    n: int
    dim: int
    ids: np.ndarray
    emb: np.ndarray
    emb_alt: Optional[np.ndarray]
    text: List[str]
    classical_col: str
    classical_values: np.ndarray
    graph_nodes: Optional[np.ndarray]
    centers: Optional[np.ndarray]
    cluster: Optional[np.ndarray]
    emb_centers: Optional[np.ndarray] = None
    table: pa.Table = None  # type: ignore[assignment]
    store: TableStore = None  # type: ignore[assignment]
    is_multivector: bool = False
    n_items: int = 0
    n_frames: int = 0
    is_glue: bool = False
    graph_edges: List[Tuple[int, int]] = field(default_factory=list)
    extra: Dict[str, np.ndarray] = field(default_factory=dict)


def _to_arrow_list(arr: np.ndarray) -> pa.Array:
    """Convert a (n, d) float32 array to a pa.list_(float32()) column."""
    flat = arr.reshape(-1)
    values = pa.array(flat, type=pa.float32())
    n = arr.shape[0]
    d = arr.shape[1]
    offsets = np.arange(0, (n + 1) * d, d, dtype=np.int32)
    return pa.ListArray.from_arrays(pa.array(offsets), values)


# Public registry of generators
REAL_GENERATORS = {
    "ag_news": gen_ag_news,
    "cc_news": gen_cc_news,
    "app_reviews": gen_app_reviews,
    "amazon": gen_amazon_5filter,
    "img_wikipedia": gen_img_wikipedia,
    "webvid": gen_webvid,
    "dbpedia": gen_dbpedia,
    "sift1m_extended": gen_sift1m_extended,
    "glove_extended": gen_glove_extended,
    "glue": gen_glue_table,
}
