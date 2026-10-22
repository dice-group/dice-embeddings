"""Model-agnostic Chebyshev polynomial propagation over the entity graph.

Implements the symmetric-normalized-adjacency Chebyshev recurrence used to
propagate "edit signals" (see :mod:`dicee.continual.influence`) across the
entity co-occurrence graph induced by a set of triples. This module has no
dependency on any KGE model; it only operates on integer entity ids and
``(head, relation, tail)`` index triples.

Relation labels are ignored (Section 7, scope restriction): for a triple
``(h, r, t)`` we add a single undirected propagation edge ``{h, t}``.

Sparse matrices are implemented with ``torch.sparse`` (COO) rather than
introducing a new dependency such as ``scipy``: ``torch`` is already a hard
requirement of ``dicee``.
"""

from __future__ import annotations

from typing import List

import numpy as np
import torch


def build_normalized_adjacency(
    triples: np.ndarray,
    num_entities: int,
) -> torch.Tensor:
    """Build the symmetrically normalized adjacency ``D^(-1/2) A D^(-1/2)``.

    Parameters
    ----------
    triples : np.ndarray
        ``(N, 3)`` array of ``(head_idx, relation_idx, tail_idx)``. Relation
        labels are ignored; one undirected edge ``{h, t}`` is added per
        triple (duplicate edges are merged; the graph is binarized).
    num_entities : int
        Size of the (global, stable) entity index space. The returned
        matrix has shape ``(num_entities, num_entities)`` so entity ids
        remain valid indices across snapshots even if some entities are
        not (yet) present in ``triples``.

    Returns
    -------
    torch.Tensor
        Sparse COO tensor ``L_tilde``, shape ``(num_entities, num_entities)``.
        Isolated nodes (degree 0) get a zero inverse-degree (zero rows/cols),
        as required.
    """
    if len(triples) == 0:
        return torch.sparse_coo_tensor(
            torch.zeros((2, 0), dtype=torch.long),
            torch.zeros((0,), dtype=torch.float64),
            size=(num_entities, num_entities),
        ).coalesce()

    h = torch.as_tensor(triples[:, 0], dtype=torch.long)
    t = torch.as_tensor(triples[:, 2], dtype=torch.long)
    rows = torch.cat([h, t])
    cols = torch.cat([t, h])

    # Binarize: dedupe (row, col) pairs via a coalesced sparse tensor.
    ones = torch.ones(rows.shape[0], dtype=torch.float64)
    a = torch.sparse_coo_tensor(torch.stack([rows, cols]), ones,
                                 size=(num_entities, num_entities),
                                 check_invariants=False).coalesce()
    a = torch.sparse_coo_tensor(a.indices(), torch.ones_like(a.values()),
                                 size=(num_entities, num_entities),
                                 check_invariants=False).coalesce()

    # Remove self loops.
    idx = a.indices()
    keep = idx[0] != idx[1]
    a = torch.sparse_coo_tensor(idx[:, keep], a.values()[keep],
                                 size=(num_entities, num_entities),
                                 check_invariants=False).coalesce()

    degree = torch.sparse.sum(a, dim=1).to_dense()
    inv_sqrt_degree = torch.where(degree > 0, degree.pow(-0.5), torch.zeros_like(degree))

    row_idx, col_idx = a.indices()
    norm_values = a.values() * inv_sqrt_degree[row_idx] * inv_sqrt_degree[col_idx]
    l_tilde = torch.sparse_coo_tensor(a.indices(), norm_values,
                                       size=(num_entities, num_entities),
                                       check_invariants=False).coalesce()
    return l_tilde


def chebyshev_propagate(
    l_tilde: torch.Tensor,
    signal: np.ndarray,
    K: int,
) -> List[torch.Tensor]:
    """Compute ``Z_0, Z_1, ..., Z_K`` via the Chebyshev recurrence.

        Z_0 = S_0
        Z_1 = L_tilde @ S_0
        Z_k = 2 * L_tilde @ Z_(k-1) - Z_(k-2),  k >= 2

    Parameters
    ----------
    l_tilde : torch.Tensor
        Sparse symmetrically normalized adjacency, shape ``(n, n)``.
    signal : np.ndarray
        Initial signal ``S_0``, shape ``(n, c)`` (``c`` channels) or
        ``(n,)`` (single channel, promoted to ``(n, 1)``).
    K : int
        Highest polynomial order to compute (``K >= 0``).

    Returns
    -------
    List[torch.Tensor]
        ``[Z_0, ..., Z_K]``, each of shape ``(n, c)`` (dense tensors).
    """
    if K < 0:
        raise ValueError("K must be >= 0")
    s0 = torch.as_tensor(np.asarray(signal, dtype=np.float64))
    if s0.ndim == 1:
        s0 = s0.unsqueeze(1)

    z: List[torch.Tensor] = [s0]
    if K == 0:
        return z

    z1 = torch.sparse.mm(l_tilde, s0)
    z.append(z1)
    for _k in range(2, K + 1):
        z_k = 2.0 * torch.sparse.mm(l_tilde, z[-1]) - z[-2]
        z.append(z_k)
    return z
