"""
diagnose_coverage_gaps.py

Loads a stored run, re-runs the coverage simulation, and reports the time-step distribution of the
(time step, target) pairs that have zero coverage.

This tells us whether coverage gaps are:
  - clustered near j=0 or j=L-1  ->  circulant (APC) wrap-around mismatch in the constraint matrix
  - uniform across time          ->  propagator drift or an insufficient seed set

Usage:
    python scripts/diagnose_coverage_gaps.py                       # newest result in results/
    python scripts/diagnose_coverage_gaps.py <run name | path to _result.csv | s3://.../runs/<name>_result.csv>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sbi_coverage.core.simulate import run_simulation
from sbi_coverage.runs.replay import PROPAGATOR, elements_from_orb, load_run, predefined_shell, replay_configs

# Run name, local _result.csv path, or S3 URI/key; None = newest result. A command-line argument overrides it.
RESULT: str | None = None
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

N_ZERO_PAIRS_TO_PRINT = 20   # how many specific uncovered (j, target_idx) to show


def main() -> None:
    run = load_run(sys.argv[1] if len(sys.argv) > 1 else RESULT, RESULTS_DIR)
    sc, orb = run["sidecar"], run["orb"]
    n_sats = orb.shape[0]
    print(f"Loaded {n_sats} satellites from {run['csv_path'].name}")
    print(f"  MIP gap at save time: {(sc['result'].get('mip_gap_achieved') or 0.0) * 100:.4f}%")

    earth, cov, sim = replay_configs(sc)
    shell = predefined_shell(sc, earth, cov)
    elems = elements_from_orb(orb)
    layer_specs = {"type": "SCLP-selected (RGT)", "n_satellites": n_sats}

    print(f"\nRunning coverage simulation ({n_sats} sats, horizon={sim.horizon_s/3600:.2f} h, dt={sim.dt_s:.6f} s) ...")
    analysis = run_simulation(elems, sim, cov, earth, shell, layer_specs, propagator=PROPAGATOR)

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
