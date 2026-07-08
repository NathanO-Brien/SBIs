from __future__ import annotations

import numpy as np
import scipy.sparse as sp


def build_param_V(v0_list: list[np.ndarray]) -> sp.csr_matrix:
    """Build the MILP constraint matrix param_V from seed access profiles.

    Each element of ``v0_list`` is the binary access profile of one seed
    satellite (sub-constellation) simulated over one repeat period. The APC
    circulant property means that column ``k`` of sub-constellation ``s`` is a
    k-step circular shift of ``v0_list[s]``, so the full MILP matrix can be
    built analytically from seed simulations alone.

    Parameters
    ----------
    v0_list : list of np.ndarray, each shape (L, nTargets)
        Binary access profiles. ``v0_list[s][t, j] = 1`` if the seed satellite
        of sub-constellation ``s`` can engage target ``j`` at time step ``t``.

    Returns
    -------
    scipy.sparse.csr_matrix, shape (L * nTargets, n_seeds * L)
        Stacked circulant constraint matrix with one L-column block per seed
        family.
    """
    if not v0_list:
        raise ValueError("build_param_V: v0_list must contain at least one array")

    L, n_targets = v0_list[0].shape
    for s, v0 in enumerate(v0_list):
        if v0.shape != (L, n_targets):
            raise ValueError(
                f"build_param_V: v0_list[{s}] has shape {v0.shape}, "
                f"expected ({L}, {n_targets})"
            )

    n_seeds = len(v0_list)
    n_rows = L * n_targets
    n_cols = n_seeds * L
    k_idx = np.arange(L, dtype=np.int32)

    row_chunks: list[np.ndarray] = []
    col_chunks: list[np.ndarray] = []

    for s, v0 in enumerate(v0_list):
        col_offset = s * L
        v0 = np.asarray(v0, dtype=np.uint8)
        for j in range(n_targets):
            row_offset = j * L
            hit_times = np.flatnonzero(v0[:, j]).astype(np.int32, copy=False)
            if hit_times.size == 0:
                continue

            for hit_t in hit_times:
                row_chunks.append(row_offset + ((hit_t + k_idx) % L))
                col_chunks.append(col_offset + k_idx)

    if row_chunks:
        rows = np.concatenate(row_chunks)
        cols = np.concatenate(col_chunks)
        data = np.ones(rows.size, dtype=np.uint8)
    else:
        rows = np.empty(0, dtype=np.int32)
        cols = np.empty(0, dtype=np.int32)
        data = np.empty(0, dtype=np.uint8)

    return sp.csr_matrix((data, (rows, cols)), shape=(n_rows, n_cols), dtype=np.uint8)


def param_V_summary(param_V: sp.spmatrix | np.ndarray, n_seeds: int) -> None:
    """Print a diagnostic summary of param_V."""
    total_rows, total_cols = param_V.shape
    L = total_cols // n_seeds
    n_targets = total_rows // L

    print(f"param_V shape : {total_rows} rows x {total_cols} cols")
    print(f"  nTargets    : {n_targets}")
    print(f"  L (slots)   : {L}")
    print(f"  n_seeds     : {n_seeds}")
    print(f"  BILP vars   : {total_cols}  (= n_seeds x L)")

    nnz = int(param_V.nnz) if sp.issparse(param_V) else int(np.count_nonzero(param_V))
    density = float(nnz) / float(total_rows * total_cols)
    print(f"  Density     : {density:.4f}  ({density*100:.2f}% nonzero)")

    print("\n  Per-sub-constellation access density (fraction of V_s that is 1):")
    for s in range(n_seeds):
        block = param_V[:, s * L:(s + 1) * L]
        block_nnz = int(block.nnz) if sp.issparse(block) else int(np.count_nonzero(block))
        block_density = float(block_nnz) / float(block.shape[0] * block.shape[1])
        print(f"    seed {s}: {block_density:.4f}  ({block_density*100:.2f}%)")
