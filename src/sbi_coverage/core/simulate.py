"""Coverage simulation driver.

Propagates a constellation over the simulation horizon, converts positions
to ECEF at each timestep, evaluates the coverage test against the target
shell, and returns an Analysis object (time-by-point counts plus provenance
metadata).
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from dataclasses import is_dataclass
from typing import Any

import numpy as np

from .analysis_matrix import (
    Analysis,
    shell_id_from_latlon,
)
from .config import SimConfig, CoverageConfig, EarthConstants
from .coverage_calc import coverage_counts_ecef
from .elements import OrbitalElements
from .frames import eci_to_ecef
from ..propagators import get_propagator
from .shell import Shell


def _assert_isinstance(obj: Any, cls: type, name: str) -> None:
    if not isinstance(obj, cls):
        raise TypeError(f"run_simulation(): {name} must be a {cls.__name__}, got {type(obj).__name__}")


def _assert_ndarray(x: Any, name: str) -> np.ndarray:
    if not isinstance(x, np.ndarray):
        raise TypeError(f"run_simulation(): {name} must be a numpy.ndarray, got {type(x).__name__}")
    return x


def _assert_shape(x: np.ndarray, name: str, shape_tail: tuple[int, ...]) -> None:
    if x.ndim != len(shape_tail) + 1:
        raise ValueError(f"run_simulation(): {name} must have ndim={len(shape_tail)+1}, got {x.ndim}")
    if tuple(x.shape[1:]) != shape_tail:
        raise ValueError(f"run_simulation(): {name} must have shape (N,{','.join(map(str, shape_tail))}), got {x.shape}")


def _assert_float64(x: np.ndarray, name: str) -> None:
    if x.dtype != np.float64:
        raise TypeError(f"run_simulation(): {name} must be float64, got {x.dtype}")


def _assert_finite(x: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(x)):
        raise ValueError(f"run_simulation(): {name} contains NaN or inf")


def _validate_sim_and_coverage(sim: SimConfig, cov: CoverageConfig) -> None:
    if not np.isfinite(sim.horizon_s) or sim.horizon_s < 0.0:
        raise ValueError("run_simulation(): sim.horizon_s must be finite and >= 0")
    if not np.isfinite(sim.dt_s) or sim.dt_s <= 0.0:
        raise ValueError("run_simulation(): sim.dt_s must be finite and > 0")
    if int(sim.point_chunk) <= 0:
        raise ValueError("run_simulation(): sim.point_chunk must be >= 1")
    if not np.isfinite(cov.max_range_km) or cov.max_range_km < 0.0:
        raise ValueError("run_simulation(): cov.max_range_km must be finite and >= 0")
    if not np.isfinite(cov.min_elev_deg) or cov.min_elev_deg < -90.0 or cov.min_elev_deg > 90.0:
        raise ValueError("run_simulation(): cov.min_elev_deg must be finite and in [-90, 90]")


def _validate_elements(elems0: OrbitalElements) -> int:
    """Check element arrays are finite and consistent; return the satellite count."""
    a = np.asarray(elems0.a_km, dtype=np.float64).ravel()
    e = np.asarray(elems0.e, dtype=np.float64).ravel()
    i = np.asarray(elems0.i_rad, dtype=np.float64).ravel()
    raan = np.asarray(elems0.raan_rad, dtype=np.float64).ravel()
    argp = np.asarray(elems0.argp_rad, dtype=np.float64).ravel()
    M0 = np.asarray(elems0.M0_rad, dtype=np.float64).ravel()

    ns = int(a.size)
    if ns <= 0:
        raise ValueError("run_simulation(): elems0 must contain at least one satellite")
    for name, arr in (("e", e), ("i_rad", i), ("raan_rad", raan), ("argp_rad", argp), ("M0_rad", M0)):
        if arr.size != ns:
            raise ValueError(f"run_simulation(): elems0.{name} length must match elems0.a_km ({ns}), got {arr.size}")

    _assert_finite(a, "elems0.a_km")
    _assert_finite(e, "elems0.e")
    _assert_finite(i, "elems0.i_rad")
    _assert_finite(raan, "elems0.raan_rad")
    _assert_finite(argp, "elems0.argp_rad")
    _assert_finite(M0, "elems0.M0_rad")

    if np.any(a <= 0.0):
        raise ValueError("run_simulation(): elems0.a_km must be > 0")
    if np.any(e < 0.0) or np.any(e >= 1.0):
        raise ValueError("run_simulation(): elems0.e must satisfy 0 <= e < 1")

    return ns


def _assert_sat_positions_shape(r_sat_eci: np.ndarray, ns: int) -> None:
    if not isinstance(r_sat_eci, np.ndarray):
        raise TypeError(f"run_simulation(): propagator must return numpy.ndarray, got {type(r_sat_eci).__name__}")
    if r_sat_eci.ndim != 2 or r_sat_eci.shape != (ns, 3):
        raise ValueError(
            "run_simulation(): propagator returned invalid position shape; "
            f"expected ({ns}, 3), got {tuple(r_sat_eci.shape)}"
        )
    _assert_finite(r_sat_eci, "propagator ECI positions")


def _resolve_analysis_dtype(sim: SimConfig) -> np.dtype:
    name = str(getattr(sim, "analysis_matrix_dtype", "uint16"))
    if name == "uint8":
        return np.dtype(np.uint8)
    if name == "uint16":
        return np.dtype(np.uint16)
    raise ValueError("run_simulation(): sim.analysis_matrix_dtype must be 'uint8' or 'uint16'")


def _to_jsonable(value: Any, *, max_array_inline: int = 64) -> Any:
    """Convert nested values into JSON-safe structures for metadata."""
    if is_dataclass(value):
        return _to_jsonable(asdict(value), max_array_inline=max_array_inline)
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v, max_array_inline=max_array_inline) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v, max_array_inline=max_array_inline) for v in value]
    if isinstance(value, np.ndarray):
        arr = np.asarray(value)
        if arr.size <= max_array_inline:
            return arr.tolist()
        arr64 = arr.astype(np.float64, copy=False) if np.issubdtype(arr.dtype, np.number) else None
        out = {
            "_kind": "ndarray",
            "dtype": str(arr.dtype),
            "shape": [int(x) for x in arr.shape],
            "size": int(arr.size),
            "sha256": hashlib.sha256(arr.tobytes(order="C")).hexdigest(),
        }
        if arr64 is not None:
            out["min"] = float(np.min(arr64))
            out["max"] = float(np.max(arr64))
            out["mean"] = float(np.mean(arr64))
        return out
    if isinstance(value, np.generic):
        return value.item()
    return value


def _validate_and_extract_shell_arrays(shell: Shell) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts_ecef_km = _assert_ndarray(shell.points_ecef_km, "shell.points_ecef_km")
    n_hat_ecef = _assert_ndarray(shell.normals_ecef, "shell.normals_ecef")
    lat_deg = _assert_ndarray(shell.lat_deg, "shell.lat_deg")
    lon_deg = _assert_ndarray(shell.lon_deg, "shell.lon_deg")

    _assert_shape(pts_ecef_km, "shell.points_ecef_km", (3,))
    _assert_shape(n_hat_ecef, "shell.normals_ecef", (3,))
    if lat_deg.ndim != 1 or lon_deg.ndim != 1:
        raise ValueError("run_simulation(): shell.lat_deg and shell.lon_deg must be 1D arrays")
    if (
        pts_ecef_km.shape[0] != n_hat_ecef.shape[0]
        or pts_ecef_km.shape[0] != lat_deg.shape[0]
        or pts_ecef_km.shape[0] != lon_deg.shape[0]
    ):
        raise ValueError("run_simulation(): shell arrays must all share the same length N")

    _assert_float64(pts_ecef_km, "shell.points_ecef_km")
    _assert_float64(n_hat_ecef, "shell.normals_ecef")
    _assert_float64(lat_deg, "shell.lat_deg")
    _assert_float64(lon_deg, "shell.lon_deg")
    return pts_ecef_km, n_hat_ecef, lat_deg, lon_deg


def _build_run_metadata(
    *,
    counts: np.ndarray,
    times_s: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    shell: Shell,
    shell_id: str,
    sim: SimConfig,
    cov: CoverageConfig,
    earth: EarthConstants,
    elems0: OrbitalElements,
    layer_specs: Any,
    propagator: str,
    prop_meta: Any,
) -> dict[str, Any]:
    """Assemble the provenance metadata dict stored on the Analysis result."""
    shell_meta = dict(getattr(shell, "meta", {}))
    n_points_used = int(lat_deg.size)
    n_points_requested = int(shell_meta.get("n_requested", getattr(shell, "n_points", n_points_used)))
    summary_scope = "regional" if ("roi_mode" in shell_meta or n_points_used < n_points_requested) else "global"

    counts_f = counts.astype(np.float64, copy=False)
    active_stats = {
        "min": int(np.min(counts)),
        "max": int(np.max(counts)),
        "mean": float(np.mean(counts_f)),
        "std": float(np.std(counts_f)),
        "nonzero_fraction": float(np.count_nonzero(counts) / counts.size),
    }
    has_true_global = summary_scope == "global"
    summary = {
        "summary_scope": summary_scope,
        "active": active_stats,
        "global": active_stats if has_true_global else None,
        "regional": active_stats if summary_scope == "regional" else None,
    }
    sim_meta = _sim_metadata_for_propagator(sim=sim, propagator=propagator)
    return {
        "schema_version": "run_meta.v2",
        "run": {
            "function": "run_simulation",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "propagator": propagator,
            "epoch0": getattr(sim, "epoch0", "J2000"),
        },
        "inputs": {
            "sim": _to_jsonable(sim_meta),
            "coverage": _to_jsonable(cov),
            "earth": _to_jsonable(earth),
            "shell": {
                "n_points_requested": n_points_requested,
                "n_points_used": n_points_used,
                "shell_alt_km": float(getattr(shell, "shell_alt_km", 0.0)),
                "analysis_shell_mode": str(shell_meta.get("analysis_shell_mode", "full")),
                "meta": _to_jsonable(shell_meta),
            },
            "elements0": _to_jsonable(elems0),
            "layer_specs": _to_jsonable(layer_specs),
        },
        "grid": {
            "Nt": int(times_s.size),
            "Np": int(lat_deg.size),
            "shape": [int(times_s.size), int(lat_deg.size)],
            "dtype": str(counts.dtype),
            "dt_s": float(sim.dt_s),
            "t_start_s": float(times_s[0]) if times_s.size else None,
            "t_end_s": float(times_s[-1]) if times_s.size else None,
            "lat_deg_range": [float(np.min(lat_deg)), float(np.max(lat_deg))],
            "lon_deg_range": [float(np.min(lon_deg)), float(np.max(lon_deg))],
            "analysis_shell_mode": str(shell_meta.get("analysis_shell_mode", "full")),
            "shell_id": shell_id,
        },
        "result": {
            "counts_summary": summary,
        },
        "propagator_meta": _to_jsonable(prop_meta),
    }


def _sim_metadata_for_propagator(*, sim: SimConfig, propagator: str) -> dict[str, Any]:
    """Simulation settings recorded in the run metadata."""
    return {
        "horizon_s": float(sim.horizon_s),
        "dt_s": float(sim.dt_s),
        "point_chunk": int(sim.point_chunk),
        "epoch0": str(sim.epoch0),
        "analysis_matrix_dtype": str(sim.analysis_matrix_dtype),
        "use_j2": bool(sim.use_j2),
    }


def run_simulation(
    elems0: OrbitalElements,
    sim: SimConfig,
    cov: CoverageConfig,
    earth: EarthConstants,
    shell: Shell,
    layer_specs: Any,
    *,
    propagator: str = "Nominal_Propagator",
    build_meta: bool = True,
) -> Analysis:
    """
    Run a coverage simulation for a constellation over the target shell.

    Parameters
    ----------
    elems0      : vectorized orbital elements at epoch (n_sats per field)
    sim         : simulation timeframe and propagation settings
    cov         : engagement parameters and derived interceptor reach
    earth       : Earth physical constants
    shell       : target shell (points, normals, lat/lon)
    layer_specs : constellation composition description, stored in metadata
    propagator  : registry name of the propagator to use
    build_meta  : build the provenance metadata dict (hashing/JSON conversion); pass False for the
                  thousands of per-seed simulations where only ``counts`` is used

    Returns
    -------
    Analysis with counts[t, p] = number of satellites covering shell point p
    at timestep t, plus provenance metadata.
    """
    _assert_isinstance(elems0, OrbitalElements, "elems0")
    _assert_isinstance(sim, SimConfig, "sim")
    _assert_isinstance(cov, CoverageConfig, "cov")
    _assert_isinstance(earth, EarthConstants, "earth")
    _assert_isinstance(shell, Shell, "shell")

    if not isinstance(propagator, str) or not propagator:
        raise TypeError("run_simulation(): propagator must be a non-empty string")
    _validate_sim_and_coverage(sim, cov)
    ns = _validate_elements(elems0)

    pts_ecef_km, n_hat_ecef, lat_deg, lon_deg = _validate_and_extract_shell_arrays(shell)

    Np = int(pts_ecef_km.shape[0])
    # Inclusive grid 0, dt, ..., floor(horizon/dt)*dt. Built from an integer count (not
    # np.arange with a float stop) so horizon=(L-1)*dt always yields exactly L samples.
    Nt = int(np.floor(sim.horizon_s / sim.dt_s + 1e-9)) + 1
    times_s = np.arange(Nt, dtype=np.float64) * sim.dt_s
    counts_acc = np.zeros((Nt, Np), dtype=np.uint16)
    if ns > np.iinfo(np.uint16).max:
        print(
            "[warning] run_simulation(): Ns exceeds uint16 max count; "
            f"point counts may clip/wrap during uint16 accumulation (Ns={ns})."
        )

    # --- Propagator selection + initialization ---
    prop = get_propagator(propagator)
    state = prop.init_state(elems0=elems0, sim=sim, earth=earth)

    # --- Main loop: propagate, rotate to ECEF, evaluate coverage in chunks ---
    for k, t in enumerate(times_s):
        dt = 0.0 if k == 0 else sim.dt_s

        r_sat_eci = prop.step_state_to_eci_positions(
            state=state,
            dt_s=dt,
            sim=sim,
            earth=earth,
        )

        r_sat_ecef = eci_to_ecef(r_sat_eci, t, earth)
        if k == 0:
            _assert_sat_positions_shape(r_sat_eci, ns)
            _assert_sat_positions_shape(r_sat_ecef, ns)

        for start in range(0, Np, sim.point_chunk):
            end = min(start + sim.point_chunk, Np)
            chunk_counts = coverage_counts_ecef(
                r_sat_ecef_km=r_sat_ecef,
                r_pts_ecef_km=pts_ecef_km[start:end],
                n_hat_ecef=n_hat_ecef[start:end],
                max_range_km=cov.max_range_km,
                min_elev_deg=cov.min_elev_deg,
            )
            if k == 0 and start == 0:
                if chunk_counts.ndim != 1 or chunk_counts.shape[0] != (end - start):
                    raise ValueError(
                        "run_simulation(): coverage_counts_ecef returned invalid shape; "
                        f"expected ({end - start},), got {tuple(chunk_counts.shape)}"
                    )
            counts_acc[k, start:end] = chunk_counts.astype(np.uint16, copy=False)

    out_dtype = _resolve_analysis_dtype(sim)
    if out_dtype == np.dtype(np.uint8):
        counts_max = int(np.max(counts_acc))
        if counts_max > 255:
            n_over = int(np.count_nonzero(counts_acc > 255))
            print(
                "[warning] run_simulation(): counts exceed uint8 range; "
                f"{n_over} entries >255 (max={counts_max})."
            )
    counts = counts_acc.astype(out_dtype, copy=False)

    shell_id = shell_id_from_latlon(
        lat_deg, lon_deg, shell_alt_km=float(getattr(shell, "shell_alt_km", 0.0)), extra=None
    )

    meta: dict[str, Any] = {}
    if build_meta:
        prop_meta = prop.get_metadata(sim=sim) if hasattr(prop, "get_metadata") else None
        meta = _build_run_metadata(
            counts=counts, times_s=times_s, lat_deg=lat_deg, lon_deg=lon_deg, shell=shell,
            shell_id=shell_id, sim=sim, cov=cov, earth=earth, elems0=elems0,
            layer_specs=layer_specs, propagator=propagator, prop_meta=prop_meta,
        )

    return Analysis(
        counts=counts,
        times_s=times_s,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        meta=meta,
        shell_id=shell_id,
    )
