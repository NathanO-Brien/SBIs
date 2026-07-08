from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
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

    We hash lat/lon (float64 bytes) + shell_alt_km + optional extra fields.
    This ID is used to ensure compatibility across stored analysis matrices.

    Notes:
    - This assumes that lat/lon ordering uniquely identifies the point ordering used
      in the counts matrix (which is true in your Shell object).
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

    # short but collision-resistant enough for filesystem naming
    return h.hexdigest()[:16]


@dataclass
class Analysis:
    """Core in-memory analysis product for a simulation run.

    - counts[t, p] = number of satellites in range of point p at timestep t
    - point_mean[p] = mean_t(counts[:, p])
    - time_mean[t] = mean_p(counts[t, :])
    """
    counts: np.ndarray
    times_s: np.ndarray
    lat_deg: np.ndarray
    lon_deg: np.ndarray
    meta: Dict[str, Any]
    point_mean: np.ndarray
    time_mean: np.ndarray

    # Optional bookkeeping
    shell_id: Optional[str] = None

    def flush(self) -> None:
        """Flush any memmap-backed arrays to disk."""
        try:
            # Memmap arrays have .flush(); normal ndarrays do not
            if hasattr(self.counts, "flush"):
                self.counts.flush()
        except Exception:
            pass


# Backward-compat alias for older imports
AnalysisMatrix = Analysis


def build_meta_bundle(
    *,
    sim: Any,
    cov: Any,
    shell: Any,
    earth: Any,
    layer_specs: Any,
    layer_name: str,
    dtype: str,
    counts_shape: tuple[int, int],
) -> Dict[str, Any]:
    """Build a provenance-rich metadata dict.

    This assumes sim/cov are dataclasses (asdict-able). If they are not, we fall back
    to repr().
    """
    def safe_asdict(obj: Any) -> Any:
        try:
            return asdict(obj)
        except Exception:
            return repr(obj)

    meta: Dict[str, Any] = {
        "layer_name": layer_name,
        "dtype": dtype,
        "shape": [int(counts_shape[0]), int(counts_shape[1])],
        "sim": safe_asdict(sim),
        "coverage": safe_asdict(cov),
        "earth": safe_asdict(earth),
        "shell_meta": dict(getattr(shell, "meta", {})),
        "layers": layer_specs,
    }

    # Helpful redundancy for quick filtering without opening configs
    # (only if those fields exist)
    for key in ["dt_s", "horizon_s", "n_points", "point_chunk"]:
        if hasattr(sim, key):
            meta[f"sim_{key}"] = getattr(sim, key)

    for key in ["min_elev_deg", "max_range_km"]:
        if hasattr(cov, key):
            meta[f"cov_{key}"] = getattr(cov, key)

    return meta
