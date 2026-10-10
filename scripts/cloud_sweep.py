"""
cloud_sweep.py -- run a parameter grid of optimizations on AWS Batch (or locally with --local).

Expands the grid below (salvo size x interceptors/sat x burnout velocity x detection/decision time x target
burnout time x max acceleration) for ONE prespecified target set into individual run configurations
(keys and defaults: sbi_coverage.runs.config). Each combination becomes one AWS Batch job (batch_worker.py in
the Docker image); with --local the same combinations run one after another in this process through the
same pipeline (sbi_coverage.Lee.run_pipeline.run_sclp), which is how to test a sweep before spending cloud time.

Results are flat files named by sbi_coverage.runs.names, in S3 runs/ (cloud) or --results-dir (local).
By default EVERY combination is submitted, layering new runs on top of whatever already exists (same-day
reruns get the next sequence letter). Pass --skip-existing (or set SKIP_EXISTING = True) to skip combinations
whose run identity -- readable settings plus the result-affecting hash, any date -- already has a finished
result in S3 (or in --results-dir with --local).

Examples:
    python scripts/cloud_sweep.py --target-points-csv "<...>/CN/Prespecified_100_ICBMS_CN_High_Priority.csv" \\
        --scenario CN --principal-set --dry-run
    python scripts/cloud_sweep.py --target-points-csv targets.csv --scenario NK --grid grid.json --skip-existing
    python scripts/cloud_sweep.py --target-points-csv targets.csv --local --max-jobs 2 --time-limit 60 --test-run

--grid FILE is a JSON object overriding any of the SWEEP_* lists below, e.g.
    {"salvo_sizes": [5, 10], "detection_times_s": [0, 15], "burnout_velocities_km_s": [6.0]}
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import sys
from pathlib import Path

from sbi_coverage.core.target_points import load_target_points_csv
from sbi_coverage.runs.cloud import S3_CONFIG_PREFIX, clients, s3_run_exists, sanitize_job_name, submit_job, upload_config
from sbi_coverage.runs.config import CONFIG_DEFAULTS, S3_BUCKET, normalize_config, settings_from_config
from sbi_coverage.runs.names import decode_run_name, parse_filename, scenario_from_path, settings_key

# ---------------------------------------------------------------------------
# Sweep grid (override per launch with --grid FILE)
# ---------------------------------------------------------------------------
SWEEP_SALVO_SIZES:             list[int]   = [100]
SWEEP_INTERCEPTORS_PER_SAT:    list[int]   = [1, 2, 3, 4, 5, 6]
SWEEP_BURNOUT_VELOCITIES_KM_S: list[float] = [4.0, 6.0]
SWEEP_DETECTION_TIMES_S:       list[float] = [0.0, 15.0, 30.0, 45.0]
SWEEP_DECISION_TIMES_S:        list[float] = [0.0]
SWEEP_BURNOUT_TIMES_S:         list[float] = [170.0]
SWEEP_MAX_ACCELS_G:            list[float] = [10.0]

# Fixed settings for every combination (everything else: CONFIG_DEFAULTS)
TIME_LIMIT_S: float = 14400.0            # per run; not part of run identity
MIP_GAP:      float = CONFIG_DEFAULTS["mip_gap"]
SKIP_EXISTING: bool = False              # --skip-existing sets this for one launch

_GRID_KEYS = {
    "salvo_sizes": "SWEEP_SALVO_SIZES", "interceptors_per_sat": "SWEEP_INTERCEPTORS_PER_SAT",
    "burnout_velocities_km_s": "SWEEP_BURNOUT_VELOCITIES_KM_S", "detection_times_s": "SWEEP_DETECTION_TIMES_S",
    "decision_times_s": "SWEEP_DECISION_TIMES_S", "burnout_times_s": "SWEEP_BURNOUT_TIMES_S",
    "max_accels_g": "SWEEP_MAX_ACCELS_G",
}


def build_grid(overrides: dict | None = None) -> list[dict]:
    """All combinations as dicts of run-configuration keys (row-major over the lists above)."""
    g = {"salvo_sizes": SWEEP_SALVO_SIZES, "interceptors_per_sat": SWEEP_INTERCEPTORS_PER_SAT,
         "burnout_velocities_km_s": SWEEP_BURNOUT_VELOCITIES_KM_S, "detection_times_s": SWEEP_DETECTION_TIMES_S,
         "decision_times_s": SWEEP_DECISION_TIMES_S, "burnout_times_s": SWEEP_BURNOUT_TIMES_S,
         "max_accels_g": SWEEP_MAX_ACCELS_G}
    for k, v in (overrides or {}).items():
        if k not in _GRID_KEYS:
            raise SystemExit(f"unknown grid key {k!r}; valid keys: {sorted(_GRID_KEYS)}")
        g[k] = list(v)
    return [
        dict(salvo_size=s, n_interceptors_per_sat=i, burnout_velocity_km_s=v, detection_time_s=dt,
             decision_time_s=dc, target_missile_burnout_time_s=b, max_accel_g=a)
        for s, i, v, dt, dc, b, a in itertools.product(
            g["salvo_sizes"], g["interceptors_per_sat"], g["burnout_velocities_km_s"], g["detection_times_s"],
            g["decision_times_s"], g["burnout_times_s"], g["max_accels_g"])
    ]


def _local_exists(results_dir: Path, cfg: dict) -> list[str]:
    key = settings_key(settings_from_config(normalize_config(cfg)))
    out = []
    for p in results_dir.glob("*_result.json") if results_dir.exists() else []:
        parsed = parse_filename(p.name)
        if parsed and decode_run_name(parsed[0])["key"] == key:
            out.append(parsed[0])
    return sorted(out)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run a parameter sweep of SBI optimizations (AWS Batch or --local).",
                                formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    p.add_argument("--target-points-csv", required=True, help="CSV of target points (Location/Lat/Lon)")
    p.add_argument("--scenario", help="run-name scenario code, e.g. CN (inferred from the CSV path if omitted)")
    p.add_argument("--label", help="free-text label stored in every run's JSON (the grid index is appended)")
    p.add_argument("--principal-set", action="store_true", help="mark runs as a principal research set")
    p.add_argument("--test-run", action="store_true", help="TEST- name prefix on every run")
    p.add_argument("--grid", help="JSON file overriding the SWEEP_* lists")
    p.add_argument("--time-limit", type=float, default=TIME_LIMIT_S, help="per-run solver time limit, s")
    p.add_argument("--skip-existing", action="store_true", help="skip combinations that already have a finished result")
    p.add_argument("--dry-run", action="store_true", help="list the combinations and their run identities; submit nothing")
    p.add_argument("--stagger-seconds", type=int, default=0, help="delay between the first N submissions")
    p.add_argument("--stagger-count", type=int, default=10, help="number of initial submissions to stagger")
    p.add_argument("--local", action="store_true", help="run the combinations locally, one after another")
    p.add_argument("--results-dir", default="results", help="--local: output folder (also used for --skip-existing)")
    p.add_argument("--max-jobs", type=int, default=None, help="stop after this many submitted/run combinations")
    return p


def main() -> None:
    a = _parser().parse_args()
    overrides = json.loads(Path(a.grid).read_text()) if a.grid else None
    grid = build_grid(overrides)
    points = load_target_points_csv(a.target_points_csv)
    scenario = a.scenario or scenario_from_path(a.target_points_csv)
    skip = a.skip_existing or SKIP_EXISTING
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_tag = f"sweep_{ts}"

    print("=" * 70)
    print(f"SBI parameter sweep  ({'LOCAL' if a.local else 'AWS Batch'}{', dry run' if a.dry_run else ''})")
    print(f"  Targets       : {len(points)} points from {a.target_points_csv}   scenario={scenario}")
    print(f"  Combinations  : {len(grid)}   skip existing: {skip}   time limit/run: {a.time_limit:.0f} s")
    print("=" * 70)

    s3 = batch = None
    if not a.dry_run and not a.local:
        s3, batch = clients()
    elif skip and not a.local:
        s3, _ = clients()

    submitted = skipped = failed = 0
    for idx, combo in enumerate(grid, 1):
        cfg = normalize_config({
            **combo, "target_mode": "prespecified", "target_points": points, "scenario": scenario,
            "label": f"{a.label}_{idx}" if a.label else None, "is_principal_set": a.principal_set,
            "is_test_run": a.test_run, "time_limit_s": a.time_limit, "mip_gap": MIP_GAP,
        })
        key = settings_key(settings_from_config(cfg))
        label = (f"[{idx:3d}/{len(grid)}] salvo={cfg['salvo_size']:<4} ips={cfg['n_interceptors_per_sat']} "
                 f"v={cfg['burnout_velocity_km_s']:g} detect={cfg['detection_time_s']:g} decide={cfg['decision_time_s']:g} "
                 f"burnout={cfg['target_missile_burnout_time_s']:g} g={cfg['max_accel_g']:g}")
        if a.max_jobs is not None and (submitted >= a.max_jobs):
            print(f"  STOP (--max-jobs {a.max_jobs} reached)")
            break
        if skip and (a.local or s3 is not None):
            existing = _local_exists(Path(a.results_dir), cfg) if a.local else s3_run_exists(s3, cfg)
            if existing:
                print(f"  SKIP {label}  (finished run exists: {existing[0]})")
                skipped += 1
                continue
        if a.dry_run:
            print(f"  DRY  {label}\n         identity: {key}")
            submitted += 1
            continue
        try:
            if a.local:
                from sbi_coverage.Lee.run_pipeline import run_sclp
                print(f"  RUN  {label}")
                summary = run_sclp(cfg, out_dir=a.results_dir, log=lambda m: None, solver_verbose=False)
                print(f"       -> {summary['run_name']}: {summary['n_satellites']} satellites ({summary['status']})")
            else:
                cfg["job_name"] = sanitize_job_name(f"{sweep_tag}_{(scenario or 'ps').lower()}_s{cfg['salvo_size']}"
                                                    f"_i{cfg['n_interceptors_per_sat']}_v{cfg['burnout_velocity_km_s']:g}"
                                                    f"_det{cfg['detection_time_s']:g}_dec{cfg['decision_time_s']:g}"
                                                    f"_b{cfg['target_missile_burnout_time_s']:g}_g{cfg['max_accel_g']:g}")
                uri = upload_config(s3, cfg, f"{S3_CONFIG_PREFIX}/{sweep_tag}/{cfg['job_name']}.json")
                job_id = submit_job(batch, job_name=cfg["job_name"], s3_config_uri=uri)
                print(f"  OK   {label}\n         job_id={job_id}  identity={key}")
                if a.stagger_seconds > 0 and submitted < a.stagger_count:
                    import time
                    time.sleep(a.stagger_seconds)
            submitted += 1
        except Exception as exc:  # keep sweeping; report at the end
            print(f"  FAIL {label}\n         {type(exc).__name__}: {exc}")
            failed += 1

    print("\n" + "=" * 70)
    verb = "would run" if a.dry_run else ("ran" if a.local else "submitted")
    print(f"{verb}: {submitted}   skipped: {skipped}   failed: {failed}")
    if not a.local and not a.dry_run:
        print(f"Output: s3://{S3_BUCKET}/runs/   (browse: streamlit run scripts/run_browser.py)")
    print("=" * 70)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
