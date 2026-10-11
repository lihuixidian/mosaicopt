"""Graph store implementing G_{g, d, dir} (graph expansion).

We do not have NetworkX in this environment, so the implementation is
built directly on scipy.sparse.csr_matrix: an adjacency matrix per
edge label. Reachability is computed by repeated matrix powers
(truncated at depth d), which is exact for a directed graph.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy.sparse import csr_matrix, eye


@dataclass
class GraphStore:
    """A directed labeled graph stored as sparse adjacency matrices.

    For each edge label (e.g. ``"derived_from"``), a CSR matrix `A` is
    stored where ``A[i, j] = 1`` iff there is a directed edge from
    node id ``i`` to node id ``j`` under that label.
    """
    n_nodes: int
    edges: Dict[str, List[Tuple[int, int]]] = field(default_factory=dict)

    _matrices: Dict[str, csr_matrix] = field(init=False, default_factory=dict)
    _cached_powers: Dict[Tuple[str, int], csr_matrix] = field(init=False, default_factory=dict)

    def __post_init__(self) -> None:
        for label, pairs in self.edges.items():
            if not pairs:
                self._matrices[label] = csr_matrix((self.n_nodes, self.n_nodes), dtype=np.int8)
                continue
            src = np.asarray([p[0] for p in pairs], dtype=np.int32)
            dst = np.asarray([p[1] for p in pairs], dtype=np.int32)
            data = np.ones(len(pairs), dtype=np.int8)
            self._matrices[label] = csr_matrix(
                (data, (src, dst)), shape=(self.n_nodes, self.n_nodes)
            )

    # --- public API ----------------------------------------------------

    def reachable(self, label: str, start: int, depth: int) -> List[int]:
        """Return all node ids reachable from `start` in <= depth hops under `label`."""
        if label not in self._matrices:
            return []
        A = self._matrices[label]
        if depth <= 0:
            return [start]
        # Compute (I + A + A^2 + ... + A^depth)[start, :]
        result = eye(self.n_nodes, dtype=np.int8, format="csr")
        power = eye(self.n_nodes, dtype=np.int8, format="csr")
        for _ in range(depth):
            power = power @ A
            # clamp entries to 1 to avoid double-counting
            power.data = np.minimum(power.data, np.int8(1))
            result = result + power
            # Clip to {0, 1}
            result.data = np.minimum(result.data, np.int8(1))
        row = result.getrow(start)
        return [int(i) for i in row.indices.tolist()]

    def expansion(self, label: str, seeds: Sequence[int], depth: int) -> List[Tuple[int, int, List[Tuple[int, int]]]]:
        """Return (seed, target, path) for each reachable (target, path) pair.

        For the prototype, we return only the *target*; a path-preserving
        variant can be added if needed for downstream operators.
        """
        out: List[Tuple[int, int, List[Tuple[int, int]]]] = []
        for s in seeds:
            for t in self.reachable(label, s, depth):
                if t == s and depth > 0:
                    # include self via non-trivial path
                    pass
                out.append((int(s), int(t), [(int(s), int(t))]))
        return out
