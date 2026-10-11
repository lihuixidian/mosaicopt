"""BM25 text index using rank_bm25.

Implements I_{col, term} from the brief: a BM25 query against a
text column. The implementation tokenizes on whitespace; for a
research prototype this is sufficient and matches what
Mem0/Letta's lexical channel do internally.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(s: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(s)]


@dataclass
class BM25Index:
    """BM25 index over a text column.

    The index is built once over a list of documents (one per row)
    and supports top-K queries.
    """
    docs: Sequence[str]
    k1: float = 1.5
    b: float = 0.75

    _bm25: Optional[BM25Okapi] = field(init=False, default=None)
    _doc_lens: Optional[np.ndarray] = field(init=False, default=None)

    def __post_init__(self) -> None:
        tokenized = [_tokenize(d) for d in self.docs]
        self._bm25 = BM25Okapi(tokenized, k1=self.k1, b=self.b)
        self._doc_lens = np.asarray([len(t) for t in tokenized], dtype=np.int32)
        self._tokenized = tokenized

    def top_k(self, query: str, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (ids, scores) for the top-k documents by BM25 score.

        Scores are clamped at 0; negative scores are replaced with 0
        to match the algebra's BM25 > theta semantics.
        """
        scores = np.asarray(self._bm25.get_scores(_tokenize(query)), dtype=np.float32)
        # Clamp negative scores to 0
        scores = np.maximum(scores, 0.0)
        if k >= len(scores):
            order = np.argsort(-scores)
        else:
            # argpartition for O(n) top-k, then sort the partition
            part = np.argpartition(-scores, k)[:k]
            order = part[np.argsort(-scores[part])]
        return order, scores[order]
