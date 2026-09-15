"""
sync_results_from_s3.py

Downloads the best result for every unique scenario from S3 into the local
results/ folder using an abbreviated hierarchy to stay under Windows MAX_PATH.

Scans both target modes:
    country/{country}/...          -> results/{country}/...
    prespecified/{target-name}/... -> results/prespecified/{target-name}/...

A "scenario" is fully defined by the S3 leaf folder:
    <mode>/<target-name>/target-points-{N}/intercept-alt-{h}km/orbit-alt-{alt}km/
    burnout-vel-{v}km-s/intercept-window-{w}s/max-accel-{g}g/
    interceptors-per-sat-{n}/doctrine-{d}/salvo-size-{s}/

Locally the same hierarchy is stored with abbreviated segment names:
    results/{target}/tp-{N}/ia-{h}/oa-{alt}/vbo-{v}/tw-{w}/ag-{g}/int-{n}/doc-{d}/sal-{s}/
(the leading "country" segment is dropped locally; "prespecified" is kept so
blob runs are clearly separated from country runs)

When multiple runs exist in the same leaf folder, the one with the fewest
satellites (lowest n_satellites in the JSON) is downloaded.  All four files
associated with that run are pulled:
    <job_name>.csv
    <job_name>.json
    <job_name>_gurobi.log
    <job_name>_config.json

Usage:
    python scripts/sync_results_from_s3.py            # dry run — shows what would be downloaded
    python scripts/sync_results_from_s3.py --download  # actually download
    python scripts/sync_results_from_s3.py --download --overwrite  # re-download existing files
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import boto3

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
S3_BUCKET:    str  = "sbi-optimization-runs"
AWS_REGION:   str  = "us-east-1"
S3_PREFIXES:  list[str] = ["country/", "prespecified/"]
LOCAL_ROOT:   Path = Path("results")

# Suffixes that constitute a complete run
RUN_SUFFIXES: list[str] = [".csv", ".json", "_gurobi.log", "_config.json"]

# Mapping of S3 segment prefix → local abbreviated prefix
_SEG_ABBREVS: list[tuple[str, str]] = [
    ("target-points-",       "tp-"),
    ("intercept-alt-",       "ia-"),
    ("orbit-alt-",           "oa-"),
    ("burnout-vel-",         "vbo-"),
    ("intercept-window-",    "tw-"),
    ("max-accel-",           "ag-"),
    ("interceptors-per-sat-","int-"),
    ("doctrine-",            "doc-"),
    ("salvo-size-",          "sal-"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _abbreviate_segment(seg: str) -> str:
    """Replace a long S3 folder segment name with its abbreviated form."""
    for long, short in _SEG_ABBREVS:
        if seg.startswith(long):
            return short + seg[len(long):]
    return seg


def _s3_leaf_to_local(s3_leaf: str) -> Path:
    """Convert an S3 leaf folder path to an abbreviated local Path.

    S3:    country/china/target-points-273/intercept-alt-200km/orbit-alt-404km/
           burnout-vel-10.0km-s/intercept-window-170s/max-accel-10.0g/
           interceptors-per-sat-1/doctrine-1x/salvo-size-10/
    Local: results/china/tp-273/ia-200km/oa-404km/vbo-10.0km-s/tw-170s/
           ag-10.0g/int-1/doc-1x/sal-10/
    """
    parts = [p for p in s3_leaf.split("/") if p]
    # Drop the leading "country" segment — redundant with LOCAL_ROOT.
    # Keep "prespecified" so blob runs stay clearly separated locally.
    if parts and parts[0] == "country":
        parts = parts[1:]
    abbreviated = [_abbreviate_segment(p) for p in parts]
    return LOCAL_ROOT / Path(*abbreviated)


def _leaf_folder(key: str) -> str:
    """Return the S3 leaf scenario folder (up to and including salvo-size-N/)."""
    parts = key.split("/")
    for i, part in enumerate(parts):
        if part.startswith("salvo-size-"):
            return "/".join(parts[: i + 1]) + "/"
    return ""


def _job_name_from_key(key: str) -> str:
    """Return the bare filename stem (job name) from a full S3 key."""
    filename = key.split("/")[-1]
    for suffix in RUN_SUFFIXES:
        if filename.endswith(suffix):
            return filename[: -len(suffix)]
    return filename


def _is_result_json(key: str) -> bool:
    """True for the primary result JSON (not _config.json)."""
    filename = key.split("/")[-1]
    return filename.endswith(".json") and not filename.endswith("_config.json")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(download: bool, overwrite: bool) -> None:
    s3        = boto3.client("s3", region_name=AWS_REGION)
    paginator = s3.get_paginator("list_objects_v2")

    # ------------------------------------------------------------------
    # Pass 1: collect all S3 keys grouped by leaf folder (both modes)
    # ------------------------------------------------------------------
    # leaf_folder -> list of all keys under that folder
    folder_keys: dict[str, list[str]] = defaultdict(list)

    for prefix in S3_PREFIXES:
        print(f"Scanning s3://{S3_BUCKET}/{prefix} ...")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                leaf = _leaf_folder(key)
                if leaf:
                    folder_keys[leaf].append(key)

    print(f"Found {len(folder_keys)} unique leaf scenario folder(s).\n")

    # ------------------------------------------------------------------
    # Pass 2: for each folder, find the best run (fewest satellites)
    # ------------------------------------------------------------------
    # leaf_folder -> job_name of best run
    best_job: dict[str, str] = {}
    # leaf_folder -> n_satellites of best run
    best_n: dict[str, int] = {}
    skipped_no_json: int = 0

    for leaf, keys in sorted(folder_keys.items()):
        result_json_keys = [k for k in keys if _is_result_json(k)]
        if not result_json_keys:
            print(f"  [warn] no result JSON in {leaf} — skipping")
            skipped_no_json += 1
            continue

        for json_key in result_json_keys:
            try:
                resp = s3.get_object(Bucket=S3_BUCKET, Key=json_key)
                data = json.loads(resp["Body"].read().decode("utf-8"))
            except Exception as exc:
                print(f"  [warn] could not read {json_key}: {exc}")
                continue

            n_sat = data.get("n_satellites")
            if n_sat is None:
                continue  # no solution — skip

            job_name = _job_name_from_key(json_key)
            if leaf not in best_job or int(n_sat) < best_n[leaf]:
                best_job[leaf] = job_name
                best_n[leaf]   = int(n_sat)

    print(f"Best run resolved for {len(best_job)} folder(s).")
    if skipped_no_json:
        print(f"  ({skipped_no_json} folder(s) skipped — no result JSON found)\n")
    else:
        print()

    # ------------------------------------------------------------------
    # Pass 3: download all files for the winning run in each folder
    # ------------------------------------------------------------------
    total_files  = 0
    already_have = 0
    downloaded   = 0
    errors       = 0

    for leaf, job_name in sorted(best_job.items()):
        keys_for_run = [
            f"{leaf}{job_name}{suffix}" for suffix in RUN_SUFFIXES
        ]
        local_dir = _s3_leaf_to_local(leaf)

        for key in keys_for_run:
            filename   = key.split("/")[-1]
            local_path = local_dir / filename
            total_files += 1

            if local_path.exists() and not overwrite:
                already_have += 1
                continue

            if not download:
                print(f"  DRY  s3://{S3_BUCKET}/{key}")
                print(f"       → {local_path}")
                continue

            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # boto3 writes a temp file alongside the destination before renaming.
                # On Windows, MAX_PATH (260 chars) is easily exceeded by the full
                # nested path + temp suffix.  Work around by downloading to a short
                # temp name in the same directory, then renaming.
                tmp_path = local_path.parent / "_dl_tmp"
                s3.download_file(S3_BUCKET, key, str(tmp_path))
                tmp_path.replace(local_path)
                downloaded += 1
                print(f"  OK   {key.split('/')[-1]}  →  {local_path.parent}")
            except s3.exceptions.NoSuchKey:
                # Not all runs have all four files (e.g. gurobi.log may be absent)
                print(f"  MISS {key.split('/')[-1]}  (not in S3 — skipping)")
                total_files -= 1
            except Exception as exc:
                print(f"  ERR  {key}: {exc}")
                errors += 1

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    if not download:
        print(f"Dry run complete.")
        print(f"  Scenarios found : {len(best_job)}")
        print(f"  Files to fetch  : {total_files}")
        print(f"\nRe-run with --download to actually pull the files.")
    else:
        print(f"Sync complete.")
        print(f"  Scenarios synced : {len(best_job)}")
        print(f"  Files downloaded : {downloaded}")
        print(f"  Already present  : {already_have}")
        print(f"  Errors           : {errors}")
        if already_have:
            print(f"\n  Tip: pass --overwrite to re-download files you already have.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sync best S3 results to local results/ folder."
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="Actually download files. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-download files that already exist locally.",
    )
    args = parser.parse_args()
    main(download=args.download, overwrite=args.overwrite)
