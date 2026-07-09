"""
check_result_coverage.py

Re-simulate a stored SCLP result with the CURRENT code and report whether the
selected constellation actually covers its targets.

Given a results folder containing a run's output products (<name>.csv +
<name>.json), this loads the exact orbital elements and simulation settings
from those files, runs the coverage simulation from scratch, and prints
per-target coverage statistics against the run's r_required.

Usage:
    python scripts/check_result_coverage.py "results/china/tp-273/.../sal-1"
    python scripts/check_result_coverage.py "results/.../sal-1" --verbose
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np

from sbi_coverage.core.config import EarthConstants, SimConfig, CoverageConfig
from sbi_coverage.core.elements import OrbitalElements, SatellitePhysical
from sbi_coverage.core.shell import Shell
from sbi_coverage.core.simulate import run_simulation


def _find_result_pair(folder: Path) -> tuple[Path, Path]:
    """Locate the result CSV and its JSON sidecar in the folder."""
    csvs = [p for p in folder.glob("*.csv")]
    if not csvs:
        raise FileNotFoundError(f"No .csv found in {folder}")
    if len(csvs) > 1:
        # Prefer the newest if multiple runs share the folder
        csvs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        print(f"[note] multiple CSVs found; using newest: {csvs[0].name}")
    csv_path = csvs[0]
    json_path = csv_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON sidecar not found: {json_path}")
    return csv_path, json_path


def _load_orb_csv(csv_path: Path) -> np.ndarray:
    """Return (n_sats, 6): a_km | e | i_deg | raan_deg | argp_deg | arg_lat_deg."""
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append([
                float(row["a_km"]), float(row["e"]), float(row["i_deg"]),
                float(row["raan_deg"]), float(row["argp_deg"]), float(row["arg_lat_deg"]),
            ])
    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")
    return np.array(rows, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-simulate a stored SCLP result and check its coverage."
    )
    parser.add_argument("folder", help="Path to the run's results folder (contains .csv + .json)")
    parser.add_argument("--verbose", action="store_true",
                        help="List every under-covered target individually.")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        raise NotADirectoryError(f"Not a folder: {folder}")

    csv_path, json_path = _find_result_pair(folder)
    with json_path.open() as f:
        cfg = json.load(f)

    orb = _load_orb_csv(csv_path)
    n_sats = orb.shape[0]

    rc = cfg.get("run_config", {})
    r_required = int(rc.get("r_required", 1))

    # ------------------------------------------------------------------
    # Reconstruct configs exactly from the sidecar
    # ------------------------------------------------------------------
    earth = EarthConstants()
    c = cfg["cov"]
    cov = CoverageConfig(
        earth=earth,
        T_window_s=float(c["T_window_s"]),
        v_bo_km_s=float(c["v_bo_km_s"]),
        a_g=float(c["a_g"]),
        intercept_alt_km=float(c["intercept_alt_km"]),
        min_elev_deg=float(c["min_elev_deg"]),
    )
    s = cfg["sim"]
    sim = SimConfig(
        horizon_s=float(s["horizon_s"]),
        dt_s=float(s["dt_s"]),
        use_j2=bool(s.get("use_j2", False)),
        use_drag=False,
        analysis_matrix_dtype="uint16",   # avoid uint8 clipping for large constellations
    )

    elems0 = OrbitalElements(
        a_km=orb[:, 0], e=orb[:, 1], i_rad=np.deg2rad(orb[:, 2]),
        raan_rad=np.deg2rad(orb[:, 3]), argp_rad=np.deg2rad(orb[:, 4]),
        M0_rad=np.deg2rad(orb[:, 5] - orb[:, 4]),
    )
    phys = SatellitePhysical(bc_kg_m2=np.full(n_sats, float(cfg.get("bc_kg_m2", 80.0))))

    targets = cfg.get("targets", [])
    lats = np.array([t["lat_deg"] for t in targets], dtype=np.float64)
    lons = np.array([t["lon_deg"] for t in targets], dtype=np.float64)
    labels = [t.get("name", f"Target {i}") for i, t in enumerate(targets)]
    shell = Shell.from_points(earth, lat_deg=lats, lon_deg=lons, shell_alt_km=cov.intercept_alt_km)

    print(f"Run        : {csv_path.name}")
    print(f"Satellites : {n_sats}")
    print(f"Targets    : {len(lats)}   r_required: {r_required}")
    print(f"Horizon    : {sim.horizon_s/3600} h   dt: {sim.dt_s} s   use_j2: {sim.use_j2}")
    print(f"Stored     : status={cfg.get('status_str')}  mip_gap={cfg.get('mip_gap')}")

    # ------------------------------------------------------------------
    # Re-simulate with current code
    # ------------------------------------------------------------------
    t0 = time.perf_counter()
    ao = run_simulation(
        elems0, phys, sim, cov, earth, shell,
        [{"type": "stored-result", "n_total": n_sats}],
        propagator="Nominal_Propagator",
    )
    print(f"\nRe-simulated in {time.perf_counter() - t0:.2f} s with current code.\n")

    counts = ao.counts.astype(np.int32)             # (Nt, Np)
    min_per_target = counts.min(axis=0)
    always_covered = min_per_target >= r_required
    n_ok = int(always_covered.sum())
    n_bad = len(lats) - n_ok

    print("=" * 60)
    print(f"  Targets covered at every timestep : {n_ok} / {len(lats)}  "
          f"({100.0 * n_ok / len(lats):.2f}%)")
    print(f"  Overall min count                 : {int(counts.min())}")
    print(f"  Overall mean count                : {float(counts.mean()):.2f}")
    print("=" * 60)

    if n_bad > 0:
        # Fraction of timesteps each failing target is under-covered
        frac_under = (counts < r_required).mean(axis=0)
        print(f"\n{n_bad} under-covered target(s)"
              + ("" if args.verbose else "  (pass --verbose to list all)") + ":")
        order = np.argsort(min_per_target)
        shown = order[:len(order)] if args.verbose else order[:10]
        for idx in shown:
            if always_covered[idx]:
                continue
            print(f"  {labels[idx][:40]:<40}  min_count={int(min_per_target[idx])}  "
                  f"under-covered {100.0 * float(frac_under[idx]):.1f}% of timesteps")
        if not args.verbose and n_bad > 10:
            print(f"  ... and {n_bad - 10} more")
    else:
        print("\nConstellation fully covers all targets at the required level.")


if __name__ == "__main__":
    main()
