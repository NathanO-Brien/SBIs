"""
s3_cleanup_bad_runs.py

Scans all SCLP result JSON files in S3 and deletes any run where the
wall time (modeling + optimization) is below a threshold, or where the
MIP gap is null or 1.0.

For each run, prints the summary line plus the full Gurobi solver log.
Deletes the entire run prefix: .csv, .json, _gurobi.log, _config.json.

Usage:
    python scripts/s3_cleanup_bad_runs.py                        # dry run, show all wall times + logs
    python scripts/s3_cleanup_bad_runs.py --min-time 60          # flag runs shorter than 60s
    python scripts/s3_cleanup_bad_runs.py --min-time 60 --delete # actually delete short runs
"""
from __future__ import annotations

import argparse
import json

import boto3

S3_BUCKET  = "sbi-optimization-runs"
AWS_REGION = "us-east-1"

RESULT_PREFIX = "country/"


def _wall_time(rc: dict) -> float | None:
    t_model = rc.get("time_modeling_s")
    t_solve = rc.get("time_optimization_s")
    if t_model is None and t_solve is None:
        return None
    return (t_model or 0.0) + (t_solve or 0.0)


def _fetch_gurobi_log(s3, prefix: str, job_name: str) -> str | None:
    """Download the Gurobi log for a run. Returns text or None if not found."""
    key = f"{prefix}{job_name}_gurobi.log"
    try:
        resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return resp["Body"].read().decode("utf-8", errors="replace")
    except s3.exceptions.NoSuchKey:
        return None
    except Exception as exc:
        return f"(could not fetch log: {exc})"


def main(delete: bool, min_time: float | None, show_logs: bool = False) -> None:
    s3        = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")

    bad_prefixes: list[str] = []
    checked = 0
    # Each row: (wall_val, key, prefix, job_name, gap_str, wall_str, rc)
    rows: list[tuple] = []

    print(f"Scanning s3://{S3_BUCKET}/{RESULT_PREFIX} for result JSON files ...")

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=RESULT_PREFIX):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith(".json") or key.endswith("_config.json"):
                continue

            try:
                resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
                data = json.loads(resp["Body"].read().decode("utf-8"))
            except Exception as exc:
                print(f"  WARN  could not read {key}: {exc}")
                continue

            checked += 1
            rc       = data.get("run_config", {})
            wall     = _wall_time(rc)
            gap      = rc.get("mip_gap_achieved")
            gap_str  = f"{gap:.4f}" if gap is not None else "null"
            wall_str = f"{wall:7.1f}s" if wall is not None else "   null"
            prefix   = key.rsplit("/", 1)[0] + "/"
            job_name = key.split("/")[-1].removesuffix(".json")

            rows.append((wall if wall is not None else -1.0, key, prefix, job_name, gap_str, wall_str, rc))

    # Sort shortest wall time first
    rows.sort(key=lambda r: r[0])

    for wall_val, key, prefix, job_name, gap_str, wall_str, rc in rows:
        gap_val = rc.get("mip_gap_achieved")
        is_bad = (
            (min_time is not None and wall_val < min_time)
            or wall_val < 0
            or gap_val is None
            or gap_val >= 1.0
        )

        tag       = "BAD" if is_bad else "OK "
        n_sat     = rc.get("n_satellites_selected")
        obj_bound = rc.get("obj_bound")
        sat_str   = f"{n_sat}"          if n_sat     is not None else "?"
        bnd_str   = f"{obj_bound:.2f}"  if obj_bound is not None else "?"

        print()
        print(f"  {tag}  wall={wall_str}  gap={gap_str:<10}  sats={sat_str:>4}  bound={bnd_str:<8}  {job_name}")

        # Gurobi log (only if --logs flag passed)
        if show_logs:
            log_text = _fetch_gurobi_log(s3, prefix, job_name)
            if log_text:
                print("  " + "-" * 66)
                for line in log_text.splitlines():
                    print(f"  {line}")
                print("  " + "-" * 66)
            else:
                print("  (no Gurobi log found)")

        if is_bad and prefix not in bad_prefixes:
            bad_prefixes.append(prefix)

    print(f"\nChecked {checked} result JSON(s).  Bad runs: {len(bad_prefixes)}")

    if not bad_prefixes:
        print("Nothing to delete.")
        return

    print("\nRun prefixes to delete:")
    for p in bad_prefixes:
        print(f"  s3://{S3_BUCKET}/{p}")

    if not delete:
        print("\nDry run — pass --delete to actually remove these.")
        return

    print("\nDeleting ...")
    deleted_total = 0
    for prefix in bad_prefixes:
        keys_to_delete = []
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys_to_delete.append({"Key": obj["Key"]})

        if not keys_to_delete:
            print(f"  (no objects found under {prefix})")
            continue

        s3.delete_objects(
            Bucket=S3_BUCKET,
            Delete={"Objects": keys_to_delete, "Quiet": True},
        )
        deleted_total += len(keys_to_delete)
        print(f"  Deleted {len(keys_to_delete)} object(s) from {prefix}")

    print(f"\nDone. Removed {deleted_total} object(s) across {len(bad_prefixes)} run prefix(es).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit and delete short/bad S3 runs.")
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Actually delete bad runs. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--min-time",
        type=float,
        default=None,
        metavar="SECONDS",
        help="Flag runs with wall time below this threshold as bad (e.g. --min-time 60).",
    )
    parser.add_argument(
        "--logs",
        action="store_true",
        help="Print the full Gurobi log for each run.",
    )
    args = parser.parse_args()
    main(delete=args.delete, min_time=args.min_time, show_logs=args.logs)
