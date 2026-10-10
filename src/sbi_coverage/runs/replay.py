"""Load a stored run and rebuild everything needed to re-simulate it with the current code.

Shared by the replay/analysis scripts (check_result_coverage, coverage_from_sclp, diagnose_coverage_gaps) so
they all read the CSV and sidecar the same way.
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np

from ..core.config import CoverageConfig, EarthConstants, SimConfig, coverage_config_from_sidecar
from ..core.elements import OrbitalElements
from ..core.shell import Shell
from .index import read_sidecar, resolve_run_files

PROPAGATOR = "Nominal_Propagator"


def load_orb_csv(csv_path: str | Path) -> np.ndarray:
    """(n_sats, 6) float64: a_km | e | i_deg | raan_deg | argp_deg | arg_lat_deg of the selected satellites."""
    rows = []
    with Path(csv_path).open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append([float(row[k]) for k in ("a_km", "e", "i_deg", "raan_deg", "argp_deg", "arg_lat_deg")])
    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")
    return np.array(rows, dtype=np.float64)


def load_run(target: str | Path | None, folder: str | Path = "results") -> dict[str, Any]:
    """Resolve a run (name, path or S3 URI; see resolve_run_files) and load it.

    Returns dict(csv_path, json_path, sidecar (normalized to the current vocabulary), orb (n_sats x 6)).
    """
    csv_path, json_path = resolve_run_files(target, folder)
    if not json_path.exists():
        raise FileNotFoundError(f"sidecar not found: {json_path}")
    return {"csv_path": csv_path, "json_path": json_path, "sidecar": read_sidecar(json_path), "orb": load_orb_csv(csv_path)}


def elements_from_orb(orb: np.ndarray) -> OrbitalElements:
    """OrbitalElements of the stored satellites (circular orbits: M0 = argument of latitude - argp)."""
    return OrbitalElements(
        a_km=orb[:, 0], e=orb[:, 1], i_rad=np.deg2rad(orb[:, 2]), raan_rad=np.deg2rad(orb[:, 3]),
        argp_rad=np.deg2rad(orb[:, 4]), M0_rad=np.deg2rad(orb[:, 5] - orb[:, 4]),
    )


def replay_configs(sidecar: dict[str, Any], *, analysis_dtype: str = "uint16", horizon_s: float | None = None,
                   dt_s: float | None = None) -> tuple[EarthConstants, CoverageConfig, SimConfig]:
    """EarthConstants, CoverageConfig and SimConfig exactly as stored (optionally with a new horizon/step).

    The count matrix defaults to uint16 so large constellations never wrap; ``use_j2`` is read from the
    sidecar (True for every run made by the current pipeline).
    """
    earth = EarthConstants()
    cov = coverage_config_from_sidecar(sidecar["cov"], earth)
    s = sidecar["sim"]
    sim = SimConfig(
        horizon_s=float(horizon_s if horizon_s is not None else s["horizon_s"]),
        dt_s=float(dt_s if dt_s is not None else s["dt_s"]),
        use_j2=bool(s.get("use_j2", False)),
        analysis_matrix_dtype=analysis_dtype,
    )
    return earth, cov, sim


def target_arrays(sidecar: dict[str, Any]) -> tuple[list[str], np.ndarray, np.ndarray]:
    """(labels, lat_deg, lon_deg) of the stored targets."""
    t = sidecar.get("targets", [])
    return ([x.get("name", f"Target {i}") for i, x in enumerate(t)],
            np.array([x["lat_deg"] for x in t], dtype=np.float64),
            np.array([x["lon_deg"] for x in t], dtype=np.float64))


def predefined_shell(sidecar: dict[str, Any], earth: EarthConstants, cov: CoverageConfig) -> Shell:
    """Shell containing exactly the stored target points (the optimizer's own analysis grid)."""
    labels, lats, lons = target_arrays(sidecar)
    return Shell.from_points(earth, lat_deg=lats, lon_deg=lons, shell_alt_km=cov.intercept_alt_km,
                             meta_extra={"labels": labels, "roi": "stored target points"})


def orb_from_csv_text(text: str) -> np.ndarray:
    """Same as load_orb_csv but from CSV text (e.g. downloaded from S3); no local file needed."""
    import io
    rows = [[float(row[k]) for k in ("a_km", "e", "i_deg", "raan_deg", "argp_deg", "arg_lat_deg")]
            for row in csv.DictReader(io.StringIO(text))]
    if not rows:
        raise ValueError("satellite CSV contains no data rows")
    return np.array(rows, dtype=np.float64)


def coverage_check(sidecar: dict[str, Any], orb: np.ndarray) -> dict[str, Any]:
    """Re-simulate a stored constellation with the current code and summarize per-target coverage.

    Returns dict(n_targets, n_covered, min_count, mean_count, required, per_target (name, min_count,
    fraction_under) list for under-covered targets).
    """
    from ..core.simulate import run_simulation

    earth, cov, sim = replay_configs(sidecar)
    labels, _, _ = target_arrays(sidecar)
    shell = predefined_shell(sidecar, earth, cov)
    ao = run_simulation(elements_from_orb(orb), sim, cov, earth, shell, [{"type": "stored-result", "n_total": int(orb.shape[0])}],
                        propagator=PROPAGATOR, build_meta=False)
    counts = ao.counts.astype(np.int32)
    need = int(sidecar.get("run_config", {}).get("r_required", 1))
    min_per = counts.min(axis=0)
    under = (counts < need).mean(axis=0)
    bad = [{"target": labels[i], "min_count": int(min_per[i]), "fraction_of_steps_under": float(under[i])}
           for i in np.argsort(min_per) if min_per[i] < need]
    return {"n_targets": int(counts.shape[1]), "n_covered": int((min_per >= need).sum()), "min_count": int(counts.min()),
            "mean_count": float(counts.mean()), "required": need, "under_covered": bad}
