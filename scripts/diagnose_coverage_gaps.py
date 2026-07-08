"""
diagnose_coverage_gaps.py

Loads a saved SCLP result CSV, re-runs the coverage simulation, and reports
the time-step distribution of (j, target) pairs that have zero coverage.

This tells us whether coverage gaps are:
  - Clustered near j=0 or j=L-1  →  V matrix wrap-around error (APC circulant mismatch)
  - Uniform across time           →  propagator drift or insufficient seed set

Usage:
    python scripts/diagnose_coverage_gaps.py
    python scripts/diagnose_coverage_gaps.py path/to/result.csv
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

from sbi_coverage.core.config import EarthConstants, SimConfig, CoverageConfig
from sbi_coverage.core.elements import OrbitalElements, SatellitePhysical
from sbi_coverage.core.shell import Shell
from sbi_coverage.core.simulate import run_simulation

# ---------------------------------------------------------------------------
# Config — point to the result you want to diagnose
# ---------------------------------------------------------------------------
RESULT_CSV: Path | None = Path(r"results\china\burnout-vel-4.0km-s\intercept-window-170s\intercept-alt-200km\max-accel-10.0g\orbit-alt-262km\interceptors-per-sat-1\salvo-size-1\20260612_145418_result.csv"
)

N_ZERO_PAIRS_TO_PRINT = 20   # how many specific uncovered (j, target_idx) to show


# ---------------------------------------------------------------------------
def _resolve_csv(arg: Path | None) -> Path:
    if arg and arg.exists():
        return arg
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            return p
    # auto-find newest
    results_dir = Path(__file__).resolve().parents[1] / "results"
    csvs = sorted(results_dir.rglob("*_result.csv"), key=lambda p: p.stat().st_mtime)
    if not csvs:
        raise FileNotFoundError("No *_result.csv found under results/")
    return csvs[-1]


def _load_orb_csv(csv_path: Path) -> np.ndarray:
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append([
                float(row["a_km"]),
                float(row["e"]),
                float(row["i_deg"]),
                float(row["raan_deg"]),
                float(row["argp_deg"]),
                float(row["arg_lat_deg"]),
            ])
    return np.array(rows, dtype=np.float64)


def main() -> None:
    csv_path  = _resolve_csv(RESULT_CSV)
    json_path = csv_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON sidecar not found: {json_path}")

    with json_path.open() as f:
        cfg = json.load(f)

    orb_arr = _load_orb_csv(csv_path)
    n_sats  = orb_arr.shape[0]
    print(f"Loaded {n_sats} satellites from {csv_path.name}")
    print(f"  MIP gap at save time: {cfg.get('mip_gap', 0.0)*100:.4f}%")

    earth = EarthConstants()

    c = cfg["cov"]
    cov = CoverageConfig(
        earth=earth,
        T_window_s=       float(c["T_window_s"]),
        v_bo_km_s=        float(c["v_bo_km_s"]),
        a_g=              float(c["a_g"]),
        intercept_alt_km= float(c["intercept_alt_km"]),
        min_elev_deg=     float(c["min_elev_deg"]),
    )

    s = cfg["sim"]
    sim = SimConfig(
        horizon_s=                float(s["horizon_s"]),
        dt_s=                     float(s["dt_s"]),
        analysis_matrix_dtype=    str(s.get("analysis_matrix_dtype", "uint8")),
        use_j2=                   bool(s.get("use_j2", True)),
        use_drag=                 bool(s.get("use_drag", False)),
    )

    # Reconstruct target shell from JSON sidecar
    targets = cfg.get("targets", [])
    lats   = np.array([t["lat_deg"] for t in targets], dtype=np.float64)
    lons   = np.array([t["lon_deg"] for t in targets], dtype=np.float64)
    labels = [t.get("name", f"Target {i}") for i, t in enumerate(targets)]
    shell  = Shell.from_points(
        earth,
        lat_deg=lats,
        lon_deg=lons,
        shell_alt_km=cov.intercept_alt_km,
        meta_extra={"labels": labels},
    )

    # Reconstruct orbital elements (arg_lat = argp + M => M = arg_lat - argp)
    elems = OrbitalElements(
        a_km     = orb_arr[:, 0],
        e        = orb_arr[:, 1],
        i_rad    = np.deg2rad(orb_arr[:, 2]),
        raan_rad = np.deg2rad(orb_arr[:, 3]),
        argp_rad = np.deg2rad(orb_arr[:, 4]),
        M0_rad   = np.deg2rad(orb_arr[:, 5] - orb_arr[:, 4]),
    )
    phys = SatellitePhysical(bc_kg_m2=np.full(n_sats, float(cfg.get("bc_kg_m2", 80.0))))

    layer_specs = {"type": "SCLP-selected (RGT)", "n_satellites": n_sats}

    print(f"\nRunning coverage simulation ({n_sats} sats, horizon={sim.horizon_s/3600:.2f} h, dt={sim.dt_s:.6f} s) ...")
    analysis = run_simulation(
        elems0=elems, phys=phys, sim=sim, cov=cov, earth=earth,
        shell=shell, layer_specs=layer_specs,
        propagator="Nominal_Propagator",
    )

    # analysis.counts shape: (Nt, n_targets)
    counts = np.asarray(analysis.counts)
    n_tgt  = shell.points_ecef_km.shape[0]
    L      = counts.shape[0] if counts.ndim == 2 else int(round(sim.horizon_s / sim.dt_s)) + 1

    if counts.ndim == 1:
        counts = counts.reshape(L, n_tgt)

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    nonzero = float(np.count_nonzero(counts)) / counts.size
    n_zeros = int(np.sum(counts == 0))

    print(f"\n{'='*60}")
    print(f"COVERAGE SUMMARY")
    print(f"{'='*60}")
    print(f"  L (time steps)     : {L}")
    print(f"  n_targets          : {n_tgt}")
    print(f"  Total (j,p) pairs  : {counts.size}")
    print(f"  Zero-coverage pairs: {n_zeros}  ({100*n_zeros/counts.size:.4f}%)")
    print(f"  nonzero_fraction   : {nonzero:.6f}  ({nonzero*100:.4f}%)")

    if n_zeros == 0:
        print("\n  100% coverage confirmed — no gaps.")
        return

    # Time-step distribution of zero-coverage pairs
    zeros_per_timestep = (counts == 0).sum(axis=1)   # shape (L,)
    nonzero_ts         = np.flatnonzero(zeros_per_timestep)

    print(f"\n{'='*60}")
    print(f"TIME-STEP DISTRIBUTION OF ZERO-COVERAGE PAIRS")
    print(f"{'='*60}")
    print(f"  Time steps with any zero-coverage pair: {len(nonzero_ts)} / {L}")
    print(f"  j=0   (start of repeat period): {int(zeros_per_timestep[0])} zero pairs")
    print(f"  j=L-1 (end   of repeat period): {int(zeros_per_timestep[-1])} zero pairs")
    print(f"  j=1  : {int(zeros_per_timestep[1])}")
    print(f"  j=L-2: {int(zeros_per_timestep[-2])}")

    # Histogram in 8 equal-width bins across [0, L)
    bin_edges  = np.linspace(0, L, 9, dtype=int)
    print(f"\n  Histogram of zero-pair count by time-step bin (8 bins across [0, L={L})):")
    for i in range(8):
        lo, hi = int(bin_edges[i]), int(bin_edges[i+1])
        cnt = int(zeros_per_timestep[lo:hi].sum())
        bar = "#" * min(cnt // max(1, n_zeros // 60), 60)
        print(f"    j=[{lo:4d},{hi:4d})  {cnt:6d} zeros  {bar}")

    # First and last 10 time steps
    print(f"\n  zeros_per_timestep at j=0..9:   {zeros_per_timestep[:10].tolist()}")
    print(f"  zeros_per_timestep at j=L-10..L-1: {zeros_per_timestep[-10:].tolist()}")

    # Print specific zero-coverage (j, target_idx) pairs
    jj, pp = np.where(counts == 0)
    print(f"\n  First {min(N_ZERO_PAIRS_TO_PRINT, len(jj))} uncovered (j, target_idx) pairs:")
    for idx in range(min(N_ZERO_PAIRS_TO_PRINT, len(jj))):
        print(f"    j={int(jj[idx]):4d}  target={int(pp[idx]):3d}  "
              f"(j/L={jj[idx]/L:.3f}  t={jj[idx]*sim.dt_s:.1f}s)")

    # Are gaps concentrated at first/last 5% of the repeat period?
    boundary_frac = 0.05
    in_boundary = (jj < boundary_frac * L) | (jj > (1 - boundary_frac) * L)
    pct_boundary = 100.0 * in_boundary.sum() / len(jj)
    print(f"\n  {'='*50}")
    print(f"  {pct_boundary:.1f}% of zero-coverage pairs fall in the first/last {boundary_frac*100:.0f}% of the repeat period")
    if pct_boundary > 60:
        print("  >>> DIAGNOSIS: Gaps clustered at wrap-around boundary → V matrix APC error")
    elif pct_boundary < 20:
        print("  >>> DIAGNOSIS: Gaps distributed uniformly → propagator drift or insufficient seeds")
    else:
        print("  >>> DIAGNOSIS: Mixed — partial wrap-around error + other cause")


if __name__ == "__main__":
    main()
