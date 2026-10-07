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
import re
import sys
from pathlib import Path

import boto3

from sbi_coverage.core.target_points import load_target_points_csv, smart_inclination_bounds

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
MIP_GAP:                    float = 0
TIME_LIMIT_S:               float = 14400.0
ALTITUDE_MAX_REPEAT_DAYS:   int   = 2
DT_S:                       float = 120.0
INTERCEPT_ALT_KM:           float = 200.0
MIN_ELEV_DEG:               float = 0.0
INC_SCREEN_STEP_DEG:        float = 1.0
N_RAAN_OFFSETS:             int   = 10
SEED_WORKERS:               int   = 8
# mip_focus/heuristics_frac/solver_params are intentionally not exposed here
# -- batch_worker.py's config schema defaults to vanilla Gurobi tuning
# (MIPFocus=0, Heuristics=0.05, no solver_params overrides) when these keys
# are absent from the uploaded config, which is what omitting them here
# achieves. The schema itself still accepts them if per-run tuning is worth
# reintroducing.

# ---------------------------------------------------------------------------
# Target mode -- "country" (sweeps SWEEP_COUNTRIES below) or "prespecified"
# (one fixed target-point set, shared by every combo in the sweep; not itself
# a swept dimension).
# ---------------------------------------------------------------------------
TARGET_MODE: str = "prespecified"
TARGET_POINTS: list[list] = [
    ["Name", 39.79, 125.66],
]
# If set (and TARGET_MODE == "prespecified"), overrides TARGET_POINTS above by
# loading a CSV -- see sbi_coverage.core.target_points.load_target_points_csv.
TARGET_POINTS_CSV: str | Path | None = r"C:\Users\natha\OneDrive\Documents\GT 2026 Fall Semester\Research\Principle Sets\CN\Prespecified_100_ICBMS_CN_High_Priority.csv"

# Per-country inclination sweep bounds — lower bound ≈ minimum latitude of
# interest for each country, matching sweep_optimize.py. Prespecified mode
# instead computes its bounds dynamically via smart_inclination_bounds().
INCLINATION_SWEEP_BOUNDS_DEG: dict[str, tuple[float, float]] = {
    "China":       (20.0, 90.0),
    "North Korea": (35.0, 90.0),
    "Russia":      (40.0, 90.0),
    "Iran":        (20.0, 90.0)
}
DEFAULT_INCLINATION_BOUNDS: tuple[float, float] = (0, 90.0)

# ---------------------------------------------------------------------------
# Run metadata (optional; saved into the output JSON only -- never used for
# job/config naming, which is unaffected by these). Each combo's saved
# run_name gets "_<n>" appended, n being its 1-based position in the grid.
# ---------------------------------------------------------------------------
RUN_NAME: str | None = "Prespecified_100_ICBMS_CN_High_Priority_PRINCIPAL"
IS_PRINCIPAL_SET: bool = True

# ---------------------------------------------------------------------------
# Sweep grid — mirrors sweep_optimize.py defaults.
# Change these lists to define the combinations you want to run.
# ---------------------------------------------------------------------------
SWEEP_COUNTRIES:                list[str]   = []
SWEEP_ALTITUDES_KM:             list[float] = [400.0]
SWEEP_SALVO_SIZES:              list[int]   = [100]
SWEEP_INTERCEPTORS_PER_SAT:     list[int]   = [1, 2, 3, 4, 5, 6]
SWEEP_BURNOUT_VELOCITIES_KM_S:  list[float] = [4.0, 6.0]
# New sweepable dimensions -- default to a single element (the timing/accel
# value CoverageConfig would otherwise default to) so an unmodified sweep's
# grid size is unaffected.
SWEEP_DETECTION_TIMES_S:        list[float] = [0.0, 15.0, 30.0, 45.0]
SWEEP_DECISION_TIMES_S:         list[float] = [0.0]
SWEEP_BURNOUT_TIMES_S:          list[float] = [170.0]
SWEEP_A_G_LIST:                 list[float] = [10.0]
# margin_multiplier/trim_margin_multiplier/max_seeds_safety_cap/polar_band_deg/
# polar_band_divisor (the rarity-weighted, margin-driven build+trim seed
# selection -- see sweep_optimize.py/lee_test2.py) are intentionally not
# exposed here -- batch_worker.py's config schema already defaults to the
# same values (3.0/1.25/200/[70,90]/5.0) when these keys are absent from the
# uploaded config, which is what omitting them here achieves. The schema
# still accepts them if per-sweep overrides are worth reintroducing.

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sweep_tag(ts: str) -> str:
    """Short tag used to namespace this sweep's config files in S3."""
    return f"sweep_{ts}"


def _config_for(
    *,
    target_mode: str,
    country: str | None,
    target_points: list | None,
    inc_bounds: tuple[float, float],
    altitude_km: float,
    salvo_size: int,
    n_interceptors_per_sat: int,
    v_bo_km_s: float,
    detection_time_s: float,
    decision_time_s: float,
    target_missile_burnout_time_s: float,
    a_g: float,
    job_name: str,
    run_name: str | None,
    is_principal_set: bool,
) -> dict:
    """Build the JSON config dict for one batch_worker.py job."""
    return {
        "job_name":                     job_name,
        "run_name":                     run_name,
        "is_principal_set":             is_principal_set,
        "target_mode":                  target_mode,
        "target_country":               country if target_mode == "country" else None,
        "target_points":                target_points if target_mode == "prespecified" else None,
        "target_shell_n_points":        TARGET_SHELL_N_POINTS if target_mode == "country" else None,
        "target_altitude_km":           altitude_km,
        "inclination_sweep_bounds_deg": list(inc_bounds),
        "v_bo_km_s":                    v_bo_km_s,
        "salvo_size":                   salvo_size,
        "n_interceptors_per_sat":       n_interceptors_per_sat,
        "mip_gap":                      MIP_GAP,
        "time_limit_s":                 TIME_LIMIT_S,
        "altitude_max_repeat_days":     ALTITUDE_MAX_REPEAT_DAYS,
        "dt_s":                         DT_S,
        "detection_time_s":             detection_time_s,
        "decision_time_s":              decision_time_s,
        "target_missile_burnout_time_s": target_missile_burnout_time_s,
        "a_g":                          a_g,
        "intercept_alt_km":             INTERCEPT_ALT_KM,
        "min_elev_deg":                 MIN_ELEV_DEG,
        "inc_screen_step_deg":          INC_SCREEN_STEP_DEG,
        "n_raan_offsets":               N_RAAN_OFFSETS,
        "seed_workers":                 SEED_WORKERS,
    }


def _results_exist(
    s3: "boto3.client", bucket: str, *, target_mode: str, country: str | None,
    target_points: list | None, altitude_km: float, salvo: int, n_int: int, v_bo: float,
    detection_time_s: float, decision_time_s: float,
    target_missile_burnout_time_s: float, a_g: float,
) -> bool:
    """Return True if a .csv result file already exists in S3 for this combination.

    Checks burnout vel, interceptors, salvo, intercept alt, orbit altitude, AND
    (item 9) the burnout/detection/decision-time and max-accel segments -- once
    those can vary per combo, omitting them here would falsely treat combos
    that only differ in one of these as duplicates of each other.
    The snapped orbit altitude may differ slightly from the requested one, so we
    match on a rounded integer (e.g. 'orbit-alt-400km') — sufficient because
    sweep altitudes are spaced further apart than snapping precision.
    """
    if target_mode == "prespecified":
        pts = target_points or []
        first_name = str(pts[0][0]).strip().lower().replace(" ", "-").replace(":", "").replace("/", "-") if pts else "prespecified"
        broad_prefix = f"prespecified/{first_name[:40]}/"
    else:
        country_slug = (country or "").lower().replace(" ", "-")
        broad_prefix = f"country/{country_slug}/"
    intercept_alt = round(INTERCEPT_ALT_KM)
    t_window      = round(target_missile_burnout_time_s)
    detect_time   = round(detection_time_s)
    decide_time   = round(decision_time_s)
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
                and f"intercept-window-{t_window}s/" in key
                and f"detection-time-{detect_time}s/" in key
                and f"decision-time-{decide_time}s/" in key
                and f"max-accel-{a_g:.1f}g/" in key
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

    # Resolve target mode once. Prespecified mode uses ONE fixed target-point
    # set for the whole sweep (not itself a swept dimension -- there's no
    # concept elsewhere in this codebase of sweeping over multiple
    # prespecified sets, and none was requested).
    if TARGET_MODE == "prespecified":
        fixed_target_points = load_target_points_csv(TARGET_POINTS_CSV) if TARGET_POINTS_CSV else TARGET_POINTS
        fixed_inc_bounds = smart_inclination_bounds(fixed_target_points)
        fixed_target_slug = (
            str(fixed_target_points[0][0]).strip().lower()
            .replace(" ", "-").replace(":", "").replace("/", "-")
        )[:20]
    else:
        fixed_target_points = None
        fixed_inc_bounds = None
        fixed_target_slug = None

    # Grid as a list of per-combo dicts rather than positional tuples --
    # "country" only participates as a dimension in country mode, and this
    # avoids extending an already-wide positional tuple with a conditional slot.
    dims: dict[str, list] = {}
    if TARGET_MODE == "country":
        dims["country"] = SWEEP_COUNTRIES
    dims.update({
        "altitude_km":                    SWEEP_ALTITUDES_KM,
        "salvo_size":                     SWEEP_SALVO_SIZES,
        "n_interceptors_per_sat":         SWEEP_INTERCEPTORS_PER_SAT,
        "v_bo_km_s":                      SWEEP_BURNOUT_VELOCITIES_KM_S,
        "detection_time_s":               SWEEP_DETECTION_TIMES_S,
        "decision_time_s":                SWEEP_DECISION_TIMES_S,
        "target_missile_burnout_time_s":  SWEEP_BURNOUT_TIMES_S,
        "a_g":                            SWEEP_A_G_LIST,
    })
    grid = [dict(zip(dims.keys(), values)) for values in itertools.product(*dims.values())]
    n_total = len(grid)

    dim_str = " × ".join(f"{len(v)}" for v in dims.values())

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

    for idx, combo in enumerate(grid, 1):
        if TARGET_MODE == "country":
            country       = combo["country"]
            target_points = None
            inc_bounds    = INCLINATION_SWEEP_BOUNDS_DEG.get(country, DEFAULT_INCLINATION_BOUNDS)
            target_slug   = country.lower().replace(" ", "-")
            target_label  = country
        else:
            country       = None
            target_points = fixed_target_points
            inc_bounds    = fixed_inc_bounds
            target_slug   = fixed_target_slug
            target_label  = fixed_target_slug

        alt_km    = combo["altitude_km"]
        salvo     = combo["salvo_size"]
        n_int     = combo["n_interceptors_per_sat"]
        v_bo      = combo["v_bo_km_s"]
        detect_s  = combo["detection_time_s"]
        decide_s  = combo["decision_time_s"]
        burnout_s = combo["target_missile_burnout_time_s"]
        a_g_val   = combo["a_g"]

        # Every swept dimension (including the 4 new item-9 ones) must appear
        # in job_name -- two combos differing only in one of these would
        # otherwise share an S3 config key, and since Batch jobs run
        # asynchronously, a later upload can silently clobber an earlier job's
        # config before its container downloads it.
        job_name = (
            f"{sweep_tag}"
            f"_{target_slug}"
            f"_alt{int(alt_km)}km"
            f"_s{salvo}"
            f"_i{n_int}"
            f"_v{v_bo:.0f}"
            f"_dt{int(detect_s)}"
            f"_dc{int(decide_s)}"
            f"_bt{int(burnout_s)}"
            f"_ag{a_g_val:.1f}".replace(".", "p")
        )
        # AWS Batch job names must match [a-zA-Z0-9_-] and be <=128 chars.
        # a_g's decimal point is handled above ("10.0" -> "10p0") since "p" is
        # more legible than a blanket substitution there. Everything else --
        # in particular target_slug, which comes from a free-text target/point
        # name (country names are safe, but a prespecified point's name can
        # contain parentheses, commas, etc. -- confirmed to break job
        # submission with a real Russian target set) -- is caught by this
        # final whitelist substitution instead of trying to blacklist every
        # punctuation character that might show up in a target name.
        job_name = re.sub(r"[^A-Za-z0-9_-]", "-", job_name)
        job_name = re.sub(r"-{2,}", "-", job_name)
        job_name = job_name[:128]

        combo_run_name = f"{RUN_NAME}_{idx}" if RUN_NAME else None

        config_key = f"{S3_CONFIG_PREFIX}/{sweep_tag}/{job_name}.json"
        cfg        = _config_for(
            target_mode=TARGET_MODE,
            country=country,
            target_points=target_points,
            inc_bounds=inc_bounds,
            altitude_km=alt_km,
            salvo_size=salvo,
            n_interceptors_per_sat=n_int,
            v_bo_km_s=v_bo,
            detection_time_s=detect_s,
            decision_time_s=decide_s,
            target_missile_burnout_time_s=burnout_s,
            a_g=a_g_val,
            job_name=job_name,
            run_name=combo_run_name,
            is_principal_set=IS_PRINCIPAL_SET,
        )

        label = (
            f"[{idx:3d}/{n_total}]  {target_label:<12}  "
            f"alt={alt_km:.0f}km  salvo={salvo:<4}  int/sat={n_int}  v_bo={v_bo:.0f}km/s  "
            f"detect={detect_s:.0f}s  decide={decide_s:.0f}s  burnout={burnout_s:.0f}s  a_g={a_g_val:.1f}"
        )

        if dry_run:
            print(f"  DRY  {label}")
            print(f"         config  → s3://{S3_BUCKET}/{config_key}")
            print(f"         job     → {job_name}")
            continue

        # Check if results already exist in S3 — skip if so
        if _results_exist(
            s3, S3_BUCKET,
            target_mode=TARGET_MODE, country=country, target_points=target_points,
            altitude_km=alt_km, salvo=salvo, n_int=n_int, v_bo=v_bo,
            detection_time_s=detect_s, decision_time_s=decide_s,
            target_missile_burnout_time_s=burnout_s, a_g=a_g_val,
        ):
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
