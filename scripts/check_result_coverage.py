"""
check_result_coverage.py

Re-simulate a stored result with the CURRENT code and report whether the selected constellation
actually covers its targets.

Given a run (its short name, a path to its _result.csv/_result.json, an S3 URI/key, or a folder holding
one), this loads the exact orbital elements and simulation settings from the run's files, runs the
coverage simulation from scratch, and prints per-target coverage statistics against the run's r_required.

Usage:
    python scripts/check_result_coverage.py CNp-s046-d15-v6-g10-b170-k0-i1-261009a-6dff
    python scripts/check_result_coverage.py results/CNp-...-6dff_result.json --verbose
    python scripts/check_result_coverage.py <name> --folder some/other/dir

A run name that is not in --folder (default: results) is downloaded from the flat
s3://sbi-optimization-runs/runs/ prefix into that folder first, so this also checks cloud results.
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from sbi_coverage.core.simulate import run_simulation
from sbi_coverage.runs.replay import (PROPAGATOR, elements_from_orb, load_run, predefined_shell, replay_configs,
                                      target_arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-simulate a stored result and check its coverage.")
    parser.add_argument("run", help="Run name, path to its _result.csv/.json, S3 URI/key, or a folder containing one")
    parser.add_argument("--folder", default="results", help="Local flat results folder (default: results)")
    parser.add_argument("--verbose", action="store_true", help="List every under-covered target individually.")
    args = parser.parse_args()

    run = load_run(args.run, args.folder)
    sc, orb = run["sidecar"], run["orb"]
    n_sats = orb.shape[0]
    r_required = int(sc["run_config"].get("r_required", 1))

    earth, cov, sim = replay_configs(sc)
    labels, lats, _ = target_arrays(sc)
    shell = predefined_shell(sc, earth, cov)

    print(f"Run        : {run['csv_path'].name}")
    print(f"Satellites : {n_sats}")
    print(f"Targets    : {len(lats)}   r_required: {r_required}")
    print(f"Horizon    : {sim.horizon_s / 3600} h   dt: {sim.dt_s} s   use_j2: {sim.use_j2}")
    res = sc["result"]
    print(f"Stored     : status={res.get('status')}  mip_gap={res.get('mip_gap_achieved')}")

    t0 = time.perf_counter()
    ao = run_simulation(elements_from_orb(orb), sim, cov, earth, shell, [{"type": "stored-result", "n_total": n_sats}],
                        propagator=PROPAGATOR)
    print(f"\nRe-simulated in {time.perf_counter() - t0:.2f} s with current code.\n")

    counts = ao.counts.astype(np.int32)             # (Nt, Np)
    min_per_target = counts.min(axis=0)
    always_covered = min_per_target >= r_required
    n_ok = int(always_covered.sum())
    n_bad = len(lats) - n_ok

    print("=" * 60)
    print(f"  Targets covered at every timestep : {n_ok} / {len(lats)}  ({100.0 * n_ok / len(lats):.2f}%)")
    print(f"  Overall min count                 : {int(counts.min())}")
    print(f"  Overall mean count                : {float(counts.mean()):.2f}")
    print("=" * 60)

    if n_bad > 0:
        frac_under = (counts < r_required).mean(axis=0)   # fraction of steps each target is under-covered
        print(f"\n{n_bad} under-covered target(s)" + ("" if args.verbose else "  (pass --verbose to list all)") + ":")
        order = np.argsort(min_per_target)
        shown = order if args.verbose else order[:10]
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
