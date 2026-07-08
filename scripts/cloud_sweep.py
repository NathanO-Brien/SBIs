"""
cloud_sweep.py

Cloud parameter sweep launcher for the SBI SCLP constellation optimizer.

Expands a parameter grid into individual job configurations, uploads each
config JSON to S3, and submits a separate AWS Batch job for every combination.
All jobs run in parallel on the cloud (subject to the vCPU limit on the
compute environment).  This script does no computation locally — it should
finish in well under a minute regardless of sweep size.

Edit the SWEEP_* constants below to define the parameter grid, then run:

    python scripts/cloud_sweep.py

Results land in S3 under automatically-generated hierarchical prefixes, one
per combination, matching the structure used by batch_worker.py.

To dry-run without submitting jobs (prints what would be submitted):
    python scripts/cloud_sweep.py --dry-run
"""
from __future__ import annotations

import argparse
import datetime
import itertools
import json
import sys

import boto3

# ---------------------------------------------------------------------------
# AWS / S3 configuration — edit to match your deployment
# ---------------------------------------------------------------------------
S3_BUCKET:          str = "sbi-optimization-runs"
S3_CONFIG_PREFIX:   str = "configs/sweeps"          # where per-job JSON configs are uploaded
JOB_QUEUE:          str = "sbi-queue"
JOB_DEFINITION:     str = "sbi-optimizer-job"
AWS_REGION:         str = "us-east-1"

# ---------------------------------------------------------------------------
# Fixed parameters — must match sweep_optimize.py / batch_worker.py exactly.
# ---------------------------------------------------------------------------
TARGET_SHELL_N_POINTS:      int   = 15000
USE_J2:                     bool  = False
MIP_GAP:                    float = 0.001
TIME_LIMIT_S:               float = 14400.0
ALTITUDE_MAX_REPEAT_DAYS:   int   = 2
DT_S:                       float = 120.0
T_WINDOW_S:                 float = 170.0
A_G:                        float = 10.0
INTERCEPT_ALT_KM:           float = 200.0
MIN_ELEV_DEG:               float = 0.0
INC_SCREEN_STEP_DEG:        float = 1.0
N_RAAN_OFFSETS:             int   = 10
SEED_WORKERS:               int   = 8
RESERVE_POLAR_ORBITS:       bool  = False
MIP_FOCUS:                  int   = 1
HEURISTICS_FRAC:            float = 0.2
SOLVER_PARAMS:              dict  = {
    # Method 1 = dual simplex. Previously Method 2 (Barrier) — barrier took 300s+ on root LP
    # and crossover added another 70s. For pure 0/1 set-cover with an all-ones matrix,
    # dual simplex is faster because the matrix is sparse and simplex exploits that.
    "Method":                1,
    # Write B&B nodes to disk at 4 GB — prevents OOM during branching.
    "NodefileStart":         4,
    # RINS heuristic every 50 nodes — this is what actually finds good solutions.
    # B&B alone cannot explore enough nodes on problems of this size.
    "RINS":                  50,
    # Presolve removed 0 rows and 0 columns in testing (pure 0/1 set-cover gives it
    # nothing to exploit) while consuming 268s. Disabled.
    # Previously: "Presolve": 2
    # No-improvement termination: stop if no incumbent improvement for 1 hour,
    # but don't start that clock until 3 hours in — gives RINS time to find its
    # first solution on hard instances before the termination check activates.
    # Previously used a fixed TimeLimit only, which wasted hours on a stalled solver.
}

# Per-country inclination sweep bounds — lower bound ≈ minimum latitude of
# interest for each country, matching sweep_optimize.py.
INCLINATION_SWEEP_BOUNDS_DEG: dict[str, tuple[float, float]] = {
    "China":       (20.0, 90.0),
    "North Korea": (35.0, 90.0),
    "Russia":      (40.0, 90.0),
    "Iran":        (20.0, 90.0)
}
DEFAULT_INCLINATION_BOUNDS: tuple[float, float] = (20.0, 90.0)

# ---------------------------------------------------------------------------
# Sweep grid — mirrors sweep_optimize.py defaults.
# Change these lists to define the combinations you want to run.
# ---------------------------------------------------------------------------
SWEEP_COUNTRIES:                list[str]   = ["North Korea"]
SWEEP_ALTITUDES_KM:             list[float] = [260.0, 400.0, 550.0]
SWEEP_SALVO_SIZES:              list[int]   = [1, 10, 20, 35]
SWEEP_INTERCEPTORS_PER_SAT:     list[int]   = [1, 2, 4, 6]
SWEEP_BURNOUT_VELOCITIES_KM_S:  list[float] = [4.0, 5.0, 6.0, 10.0]
SEEDS_FOR_MILP_BY_SALVO:        dict  = {1: 5, # 10
                                        5: 10,  # 15
                                        20: 15, # 20
                                        35: 20, # 30
                                        50: 25, # 40
                                        75: 30, # 45
                                        100: 40} # 50

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sweep_tag(ts: str) -> str:
    """Short tag used to namespace this sweep's config files in S3."""
    return f"sweep_{ts}"


def _config_for(
    *,
    country: str,
    altitude_km: float,
    salvo_size: int,
    n_interceptors_per_sat: int,
    v_bo_km_s: float,
    job_name: str,
) -> dict:
    """Build the JSON config dict for one batch_worker.py job."""
    inc_bounds = list(INCLINATION_SWEEP_BOUNDS_DEG.get(country, DEFAULT_INCLINATION_BOUNDS))
    return {
        "job_name":                     job_name,
        "target_country":               country,
        "target_shell_n_points":        TARGET_SHELL_N_POINTS,
        "altitude_override_km":         altitude_km,
        "inclination_sweep_bounds_deg": inc_bounds,
        "use_j2":                       USE_J2,
        "v_bo_km_s":                    v_bo_km_s,
        "salvo_size":                   salvo_size,
        "n_interceptors_per_sat":       n_interceptors_per_sat,
        "mip_gap":                      MIP_GAP,
        "time_limit_s":                 TIME_LIMIT_S,
        "altitude_max_repeat_days":     ALTITUDE_MAX_REPEAT_DAYS,
        "seeds_for_milp_by_salvo":      SEEDS_FOR_MILP_BY_SALVO,
        "dt_s":                         DT_S,
        "t_window_s":                   T_WINDOW_S,
        "a_g":                          A_G,
        "intercept_alt_km":             INTERCEPT_ALT_KM,
        "min_elev_deg":                 MIN_ELEV_DEG,
        "inc_screen_step_deg":          INC_SCREEN_STEP_DEG,
        "n_raan_offsets":               N_RAAN_OFFSETS,
        "seed_workers":                 SEED_WORKERS,
        "reserve_polar_orbits":         RESERVE_POLAR_ORBITS,
        "mip_focus":                    MIP_FOCUS,
        "heuristics_frac":              HEURISTICS_FRAC,
        "solver_params":                SOLVER_PARAMS,
    }


def _results_exist(s3: "boto3.client", bucket: str, country: str, altitude_km: float,
                   salvo: int, n_int: int, v_bo: float) -> bool:
    """Return True if a .csv result file already exists in S3 for this combination.

    Checks burnout vel, interceptors, salvo, intercept alt, AND orbit altitude.
    The snapped orbit altitude may differ slightly from the requested one, so we
    match on a rounded integer (e.g. 'orbit-alt-400km') — sufficient because
    sweep altitudes are spaced further apart than snapping precision.
    """
    country_slug  = country.lower().replace(" ", "-")
    intercept_alt = 200   # _INTERCEPT_ALT_KM
    broad_prefix  = f"country/{country_slug}/"
    paginator = s3.get_paginator("list_objects_v2")
    alt_candidates = {f"orbit-alt-{int(round(altitude_km + d))}km" for d in range(-15, 16)}
    for page in paginator.paginate(Bucket=bucket, Prefix=broad_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if (
                key.endswith(".csv")
                and any(cand + "/" in key for cand in alt_candidates)
                and f"burnout-vel-{v_bo:.1f}km-s/" in key
                and f"interceptors-per-sat-{n_int}/" in key
                and f"salvo-size-{salvo}/" in key
                and f"intercept-alt-{intercept_alt}km/" in key
            ):
                return True
    return False


def _upload_config(s3: "boto3.client", cfg: dict, bucket: str, key: str) -> str:
    """Serialize cfg to JSON and upload to S3. Returns the full s3:// URI."""
    body = json.dumps(cfg, indent=2).encode("utf-8")
    s3.put_object(Body=body, Bucket=bucket, Key=key)
    return f"s3://{bucket}/{key}"


def _submit_job(
    batch: "boto3.client",
    *,
    job_name: str,
    s3_config_uri: str,
) -> str:
    """Submit one AWS Batch job. Returns the job ID."""
    response = batch.submit_job(
        jobName=job_name,
        jobQueue=JOB_QUEUE,
        jobDefinition=JOB_DEFINITION,
        containerOverrides={
            "environment": [
                {"name": "S3_CONFIG", "value": s3_config_uri},
                {"name": "S3_BUCKET", "value": S3_BUCKET},
            ]
        },
    )
    return response["jobId"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(dry_run: bool = False, stagger_seconds: int = 0, stagger_count: int = 10) -> None:
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_tag = _sweep_tag(ts)

    grid = list(itertools.product(
        SWEEP_COUNTRIES,
        SWEEP_ALTITUDES_KM,
        SWEEP_SALVO_SIZES,
        SWEEP_INTERCEPTORS_PER_SAT,
        SWEEP_BURNOUT_VELOCITIES_KM_S,
    ))
    n_total = len(grid)

    dim_str = " × ".join(str(len(x)) for x in [
        SWEEP_COUNTRIES,
        SWEEP_ALTITUDES_KM,
        SWEEP_SALVO_SIZES,
        SWEEP_INTERCEPTORS_PER_SAT,
        SWEEP_BURNOUT_VELOCITIES_KM_S,
    ])

    print("=" * 70)
    print("cloud_sweep.py  —  SBI SCLP Cloud Parameter Sweep Launcher")
    print("=" * 70)
    print(f"  Sweep tag          : {sweep_tag}")
    print(f"  Combinations       : {n_total}  ({dim_str})")
    print(f"  S3 bucket          : {S3_BUCKET}")
    print(f"  Config prefix      : {S3_CONFIG_PREFIX}/{sweep_tag}/")
    print(f"  Job queue          : {JOB_QUEUE}")
    print(f"  Job definition     : {JOB_DEFINITION}")
    print(f"  Time limit / job   : {TIME_LIMIT_S / 3600:.1f} h")
    print(f"  MIP gap target     : {MIP_GAP * 100:.1f}%")
    if dry_run:
        print("\n  *** DRY RUN — no configs will be uploaded, no jobs submitted ***")
    if stagger_seconds > 0:
        print(f"  Stagger            : {stagger_seconds}s delay between first {stagger_count} submissions")
    print("=" * 70)

    if not dry_run:
        s3    = boto3.client("s3",    region_name=AWS_REGION)
        batch = boto3.client("batch", region_name=AWS_REGION)

    submitted: list[dict] = []
    skipped:   list[dict] = []
    failed:    list[dict] = []

    for idx, (country, alt_km, salvo, n_int, v_bo) in enumerate(grid, 1):
        country_slug = country.lower().replace(" ", "-")
        job_name     = (
            f"{sweep_tag}"
            f"_{country_slug}"
            f"_alt{int(alt_km)}km"
            f"_s{salvo}"
            f"_i{n_int}"
            f"_v{v_bo:.0f}"
        )
        # AWS Batch job names must match [a-zA-Z0-9_-] and be ≤128 chars
        job_name = job_name[:128]

        config_key = f"{S3_CONFIG_PREFIX}/{sweep_tag}/{job_name}.json"
        cfg        = _config_for(
            country=country,
            altitude_km=alt_km,
            salvo_size=salvo,
            n_interceptors_per_sat=n_int,
            v_bo_km_s=v_bo,
            job_name=job_name,
        )

        label = (
            f"[{idx:3d}/{n_total}]  {country:<12}  "
            f"alt={alt_km:.0f}km  salvo={salvo:<4}  "
            f"int/sat={n_int}  v_bo={v_bo:.0f}km/s"
        )

        if dry_run:
            print(f"  DRY  {label}")
            print(f"         config  → s3://{S3_BUCKET}/{config_key}")
            print(f"         job     → {job_name}")
            continue

        # Check if results already exist in S3 — skip if so
        if _results_exist(s3, S3_BUCKET, country, alt_km, salvo, n_int, v_bo):
            print(f"  SKIP {label}  (results already in S3)")
            skipped.append({"job_name": job_name, "reason": "results already in S3"})
            continue

        try:
            s3_uri = _upload_config(s3, cfg, S3_BUCKET, config_key)
            job_id = _submit_job(batch, job_name=job_name, s3_config_uri=s3_uri)
            print(f"  OK   {label}")
            print(f"         job_id  → {job_id}")
            submitted.append({"job_name": job_name, "job_id": job_id, "s3_config": s3_uri})
            if stagger_seconds > 0 and len(submitted) < stagger_count:
                import time
                print(f"         (staggering {stagger_seconds}s before next submission...)")
                time.sleep(stagger_seconds)
        except Exception as exc:
            print(f"  FAIL {label}")
            print(f"         error   → {exc}")
            failed.append({"job_name": job_name, "error": str(exc)})

    print("\n" + "=" * 70)
    if dry_run:
        print(f"Dry run complete — {n_total} job(s) would have been submitted.")
    else:
        print(f"Sweep launch complete.")
        print(f"  Submitted : {len(submitted)}")
        print(f"  Skipped   : {len(skipped)}  (results already in S3)")
        print(f"  Failed    : {len(failed)}")
        if failed:
            print("\nFailed submissions:")
            for f in failed:
                print(f"  {f['job_name']}  —  {f['error']}")
        # Always write manifest so skips are logged even if nothing was submitted
        manifest_path = f"sweep_manifest_{ts}.json"
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump({
                "sweep_tag": sweep_tag,
                "submitted": submitted,
                "skipped":   skipped,
                "failed":    failed,
            }, fh, indent=2)
        print(f"\n  Manifest written to: {manifest_path}")
    print("=" * 70)

    if failed and not dry_run:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch SBI SCLP cloud parameter sweep.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be submitted without uploading configs or submitting jobs.",
    )
    parser.add_argument(
        "--stagger-seconds",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Delay in seconds between the first N submissions (default: 0, no stagger).",
    )
    parser.add_argument(
        "--stagger-count",
        type=int,
        default=10,
        metavar="N",
        help="Number of initial submissions to stagger (default: 10).",
    )
    args = parser.parse_args()
    main(dry_run=args.dry_run, stagger_seconds=args.stagger_seconds, stagger_count=args.stagger_count)
