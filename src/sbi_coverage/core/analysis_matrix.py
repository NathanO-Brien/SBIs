"""In-memory analysis product (time-by-point coverage counts) and its provenance metadata."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np


def shell_id_from_latlon(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    *,
    shell_alt_km: float,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Compute a stable ID for a shell point ordering.

    Hashes lat/lon (float64 bytes) + shell_alt_km + optional extra fields. The ID is used to
    verify compatibility across stored analysis matrices. Assumes the lat/lon ordering uniquely
    identifies the point ordering used in the counts matrix (true for shells built by Shell).
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = np.asarray(lon_deg, dtype=np.float64)

    h = hashlib.sha256()
    h.update(lat.tobytes(order="C"))
    h.update(lon.tobytes(order="C"))
    h.update(np.float64(shell_alt_km).tobytes())

    if extra:
        extra_json = json.dumps(extra, sort_keys=True, separators=(",", ":")).encode("utf-8")
        h.update(extra_json)

    return h.hexdigest()[:16]


@dataclass
class Analysis:
    """Result of one coverage simulation.

    counts[t, p] = number of satellites that can reach shell point p at time step t
    (shape Nt x Np). ``meta`` is the run-provenance dict (empty when the run was made with
    ``build_meta=False``, as for the thousands of per-seed simulations).
    """
    counts: np.ndarray
    times_s: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    meta: Dict[str, Any]
    shell_id: Optional[str] = None
