"""
cloud_run.py

Local launcher for a single SBI SCLP optimization job on AWS Batch.

Mirrors lee_test2.py's parameter set — specify either a country target or a
list of prespecified lat/lon points.  Builds a config JSON, uploads it to S3,
submits one AWS Batch job pointing at batch_worker.py, and prints the job ID.
No computation is done locally.

Usage (country mode):
    python scripts/cloud_run.py --country China --altitude 250 --salvo 1 --interceptors 1 --vbo 4.0

Usage (prespecified mode):
    python scripts/cloud_run.py --target-points "NK: Yongbyon,39.79,125.66" --altitude 250 --salvo 1 --interceptors 1 --vbo 4.0

Optional arguments mirror lee_test2.py defaults — edit the constants below
to change defaults without having to pass flags every time.
"""
from __future__ import annotations

import argparse
import datetime
import json

import boto3

# ---------------------------------------------------------------------------
# AWS / S3 configuration
# ---------------------------------------------------------------------------
S3_BUCKET:        str = "sbi-optimization-runs"
S3_CONFIG_PREFIX: str = "configs/runs"
JOB_QUEUE:        str = "sbi-queue"
JOB_DEFINITION:   str = "sbi-optimizer-job"
AWS_REGION:       str = "us-east-1"

# ---------------------------------------------------------------------------
# Default parameters — mirrors lee_test2.py exactly.
# Edit here to change defaults; all can also be overridden via CLI flags.
# ---------------------------------------------------------------------------

# Target
TARGET_MODE:            str   = "country"       # "country" or "prespecified"
TARGET_COUNTRY:         str   = "China"
TARGET_SHELL_N_POINTS:  int   = 15000
INCLINATION_SWEEP_BOUNDS_DEG: dict[str, tuple[float, float]] = {
    "China":       (20.0, 90.0),
    "North Korea": (35.0, 90.0),
}
DEFAULT_INCLINATION_BOUNDS: tuple[float, float] = (20.0, 90.0)

# Engagement geometry
V_BO_KM_S:          float = 4.0
SALVO_SIZE:         int   = 1
N_INTERCEPTORS_PER_SAT: int = 1
T_WINDOW_S:         float = 170.0
A_G:                float = 10.0
INTERCEPT_ALT_KM:   float = 200.0
MIN_ELEV_DEG:       float = 0.0

# Orbit
ALTITUDE_OVERRIDE_KM:     float | None = 250.0   # None → use geometric h*
ALTITUDE_MAX_REPEAT_DAYS: int          = 1
USE_J2:                   bool         = False

# Seed grid
DT_S:               float = 120.0
INC_SCREEN_STEP_DEG: float = 1.0
N_RAAN_OFFSETS:     int   = 10
SEED_WORKERS:       int   = 8
N_SEEDS_FOR_MILP:   int   = 10
RESERVE_POLAR_ORBITS: bool = False

# Solver
MIP_GAP:        float = 0.01
TIME_LIMIT_S:   float = 300.0
MIP_FOCUS:      int   = 1
HEURISTICS_FRAC: float = 0.2
SOLVER_PARAMS:  dict  = {
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Submit a single SBI SCLP optimization job to AWS Batch."
    )

    # Target — mutually exclusive: country or prespecified points
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--country",
        help="Target country name (e.g. 'China'). Sets country mode.",
    )
    target_group.add_argument(
        "--target-points",
        nargs="+",
        metavar="NAME,LAT,LON",
        help=(
            "One or more prespecified target points in 'Name,lat,lon' format "
            "(e.g. \"NK: Yongbyon,39.79,125.66\"). Sets prespecified mode."
        ),
    )

    # Engagement
    parser.add_argument("--altitude",     required=True,  type=float, help="Orbit altitude override in km (required)")
    parser.add_argument("--salvo",        default=SALVO_SIZE,          type=int,   help=f"Salvo size (default: {SALVO_SIZE})")
    parser.add_argument("--interceptors", default=N_INTERCEPTORS_PER_SAT, type=int, help=f"Interceptors per satellite (default: {N_INTERCEPTORS_PER_SAT})")
    parser.add_argument("--vbo",          default=V_BO_KM_S,           type=float, help=f"Burnout velocity in km/s (default: {V_BO_KM_S})")
    parser.add_argument("--t-window",     default=T_WINDOW_S,          type=float, help=f"Intercept time window in s (default: {T_WINDOW_S})")
    parser.add_argument("--a-g",          default=A_G,                 type=float, help=f"Max accel in g (default: {A_G})")
    parser.add_argument("--intercept-alt", default=INTERCEPT_ALT_KM,  type=float, help=f"Intercept altitude in km (default: {INTERCEPT_ALT_KM})")
    parser.add_argument("--min-elev",     default=MIN_ELEV_DEG,        type=float, help=f"Min elevation angle in deg (default: {MIN_ELEV_DEG})")

    # Orbit
    parser.add_argument("--altitude-max-repeat-days", default=ALTITUDE_MAX_REPEAT_DAYS, type=int,   help=f"Max RGT repeat days (default: {ALTITUDE_MAX_REPEAT_DAYS})")
    parser.add_argument("--use-j2",       action="store_true", default=USE_J2,  help="Enable J2 perturbation model")
    parser.add_argument("--inc-bounds",   default=None, nargs=2, type=float, metavar=("LO", "HI"),
                        help="Inclination sweep bounds in deg (default: per-country or 20 90)")

    # Seed grid
    parser.add_argument("--dt",           default=DT_S,                type=float, help=f"Time step in s (default: {DT_S})")
    parser.add_argument("--inc-step",     default=INC_SCREEN_STEP_DEG, type=float, help=f"Inclination step in deg (default: {INC_SCREEN_STEP_DEG})")
    parser.add_argument("--n-raan",       default=N_RAAN_OFFSETS,      type=int,   help=f"RAAN offsets per inclination (default: {N_RAAN_OFFSETS})")
    parser.add_argument("--seed-workers", default=SEED_WORKERS,        type=int,   help=f"Parallel seed workers (default: {SEED_WORKERS})")
    parser.add_argument("--n-seeds",      default=N_SEEDS_FOR_MILP,    type=int,   help=f"Seeds passed to MILP (default: {N_SEEDS_FOR_MILP})")
    parser.add_argument("--reserve-polar", action="store_true", default=RESERVE_POLAR_ORBITS, help="Reserve polar orbit seeds")
    parser.add_argument("--n-points",     default=TARGET_SHELL_N_POINTS, type=int, help=f"Target shell points (country mode, default: {TARGET_SHELL_N_POINTS})")

    # Solver
    parser.add_argument("--mip-gap",      default=MIP_GAP,        type=float, help=f"MIP gap target (default: {MIP_GAP})")
    parser.add_argument("--time-limit",   default=TIME_LIMIT_S,   type=float, help=f"Solver time limit in s (default: {TIME_LIMIT_S})")
    parser.add_argument("--mip-focus",    default=MIP_FOCUS,       type=int,   help=f"Gurobi MIPFocus (default: {MIP_FOCUS})")
    parser.add_argument("--heuristics",   default=HEURISTICS_FRAC, type=float, help=f"Gurobi heuristics fraction (default: {HEURISTICS_FRAC})")

    # AWS
    parser.add_argument("--job-name", default=None, help="Custom Batch job name (auto-generated if omitted)")

    args = parser.parse_args()

    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")

    # Resolve target mode
    if args.country:
        target_mode = "country"
        country     = args.country
        target_points = None
        inc_bounds = list(args.inc_bounds) if args.inc_bounds else list(
            INCLINATION_SWEEP_BOUNDS_DEG.get(country, DEFAULT_INCLINATION_BOUNDS)
        )
        slug = country.lower().replace(" ", "-")
    else:
        target_mode   = "prespecified"
        country       = None
        target_points = []
        for entry in args.target_points:
            parts = entry.split(",")
            if len(parts) != 3:
                raise ValueError(f"--target-points entries must be 'Name,lat,lon', got: {entry!r}")
            name, lat, lon = parts[0].strip(), float(parts[1]), float(parts[2])
            target_points.append([name, lat, lon])
        inc_bounds = list(args.inc_bounds) if args.inc_bounds else list(DEFAULT_INCLINATION_BOUNDS)
        slug = target_points[0][0].lower().replace(" ", "-").replace(":", "").replace("/", "-")[:20]

    job_name = args.job_name or (
        f"run_{ts}_{slug}"
        f"_alt{int(args.altitude)}km"
        f"_s{args.salvo}"
        f"_i{args.interceptors}"
        f"_v{args.vbo:.0f}"
    )
    job_name = job_name[:128]

    cfg = {
        # Target
        "job_name":                     job_name,
        "target_mode":                  target_mode,
        "target_country":               country,
        "target_points":                target_points,
        "target_shell_n_points":        args.n_points,
        "inclination_sweep_bounds_deg": inc_bounds,
        # Engagement
        "v_bo_km_s":                    args.vbo,
        "salvo_size":                   args.salvo,
        "n_interceptors_per_sat":       args.interceptors,
        "t_window_s":                   args.t_window,
        "a_g":                          args.a_g,
        "intercept_alt_km":             args.intercept_alt,
        "min_elev_deg":                 args.min_elev,
        # Orbit
        "altitude_override_km":         args.altitude,
        "altitude_max_repeat_days":     args.altitude_max_repeat_days,
        "use_j2":                       args.use_j2,
        # Seed grid
        "dt_s":                         args.dt,
        "inc_screen_step_deg":          args.inc_step,
        "n_raan_offsets":               args.n_raan,
        "seed_workers":                 args.seed_workers,
        "n_seeds_for_milp":             args.n_seeds,
        "reserve_polar_orbits":         args.reserve_polar,
        # Solver
        "mip_gap":                      args.mip_gap,
        "time_limit_s":                 args.time_limit,
        "mip_focus":                    args.mip_focus,
        "heuristics_frac":              args.heuristics,
        "solver_params":                SOLVER_PARAMS,
    }

    config_key    = f"{S3_CONFIG_PREFIX}/{job_name}.json"
    s3_config_uri = f"s3://{S3_BUCKET}/{config_key}"

    print(f"Uploading config to {s3_config_uri} ...")
    s3 = boto3.client("s3", region_name=AWS_REGION)
    s3.put_object(Body=json.dumps(cfg, indent=2).encode("utf-8"), Bucket=S3_BUCKET, Key=config_key)

    print(f"Submitting Batch job: {job_name}")
    batch = boto3.client("batch", region_name=AWS_REGION)
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
    job_id = response["jobId"]

    print(f"\nJob submitted successfully.")
    print(f"  Job name : {job_name}")
    print(f"  Job ID   : {job_id}")
    print(f"\nCheck status:")
    print(f"  aws batch describe-jobs --jobs {job_id} --query \"jobs[0].status\"")


if __name__ == "__main__":
    main()
