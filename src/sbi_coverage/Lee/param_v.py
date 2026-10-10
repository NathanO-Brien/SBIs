"""Assemble the SCLP constraint matrix (param_V) from seed access profiles
using the circulant APC structure (Lee et al. 2020).

Notation follows Lee: param_V[j*L + t, c] = 1 when candidate satellite c can reach target j at time t.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_param_V(seed_profiles: list[np.ndarray], cols_per_seed: list[int] | None = None) -> sp.csr_matrix:
    """Build the MILP constraint matrix param_V from seed access profiles.

    Each element of ``seed_profiles`` is the 0/1 access profile of one seed satellite simulated
    over one shared repeat period. The APC circulant property means column ``k`` of seed ``s`` is
    the k-step circular shift of ``seed_profiles[s]``, so the full matrix follows analytically
    from the seed simulations alone.

    Parameters
    ----------
    seed_profiles : list of np.ndarray, each shape (L, n_targets)
        ``seed_profiles[s][t, j] = 1`` if seed ``s``'s satellite can engage target ``j`` at step ``t``.
    cols_per_seed : list of int, optional
        Number of slot columns kept for each seed (default: L for every seed). A family whose
        ratio N_P:N_D is not in lowest terms (g = gcd > 1) repeats its ground track every T_r/g, so
        only the first L//g slots are distinct satellite positions; passing L//g for such a seed
        drops the repeats (each kept column is still an exact circular shift of its seed
        profile). Row structure is unchanged. Column order: seed 0 slots 0..m_0-1, then seed 1, ...

    Returns
    -------
    scipy.sparse.csr_matrix, shape (L * n_targets, sum(cols_per_seed)), dtype uint8.
    """
    if not seed_profiles:
        raise ValueError("build_param_V: seed_profiles must contain at least one array")

    L, n_targets = seed_profiles[0].shape
    for s, v0 in enumerate(seed_profiles):
        if v0.shape != (L, n_targets):
            raise ValueError(
                f"build_param_V: seed_profiles[{s}] has shape {v0.shape}, expected ({L}, {n_targets})"
            )

    n_seeds = len(seed_profiles)
    if cols_per_seed is None:
        cols_per_seed = [L] * n_seeds
    if len(cols_per_seed) != n_seeds or any(not (1 <= m <= L) for m in cols_per_seed):
        raise ValueError(f"build_param_V: cols_per_seed must have {n_seeds} entries in [1, {L}]")
    col_offsets = np.concatenate([[0], np.cumsum(cols_per_seed)]).astype(np.int64)
    n_rows = L * n_targets
    n_cols = int(col_offsets[-1])

    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []

    for s, v0 in enumerate(seed_profiles):
        # Hits of the seed (time step t, target j); column k sees each hit shifted by k steps.
        t_hit, j_hit = np.nonzero(np.asarray(v0))
        if t_hit.size == 0:
            continue
        k = np.arange(cols_per_seed[s], dtype=np.int64)
        rows = (j_hit[:, None].astype(np.int64) * L + (t_hit[:, None] + k[None, :]) % L).ravel()
        cols = np.broadcast_to(col_offsets[s] + k[None, :], (t_hit.size, k.size)).ravel()
        row_chunks.append(rows.astype(np.int32, copy=False))
        col_chunks.append(cols.astype(np.int32, copy=False))

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        data = np.ones(rows.size, dtype=np.uint8)
    else:
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)
        data = np.empty(0, dtype=np.uint8)

    return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.uint8)
