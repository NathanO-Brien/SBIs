"""
filter_principal_set_results.py

List every stored SCLP result whose run_config.is_principal_set is true.

There is no database — results are plain JSON sidecar files, locally under
results/ or in S3 (one <job_name>.json per run, uploaded by batch_worker.py).
"Filtering" means reading each sidecar's run_config.is_principal_set field.
Locally this is a fast glob + json.load. In S3 there's no server-side
filtering on JSON content (no Athena/S3 Select wired up), so this does one
GetObject per candidate key -- fine for hundreds of results, would need
batching/Athena if the result count grows into the tens of thousands.

Usage:
    python scripts/filter_principal_set_results.py --local
    python scripts/filter_principal_set_results.py --s3
    python scripts/filter_principal_set_results.py --local --s3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path("results")
S3_BUCKET = "sbi-optimization-runs"


def _summarize(run_config: dict, source: str) -> str:
    target = run_config.get("target_country") or (
        (run_config.get("target_points") or [[None]])[0][0]
        if run_config.get("target_points") else None
    ) or run_config.get("target_mode", "?")
    return (
        f"  {source}\n"
        f"    run_name={run_config.get('run_name')!r}  target={target!r}  "
        f"salvo_size={run_config.get('salvo_size')}  "
        f"n_interceptors_per_sat={run_config.get('n_interceptors_per_sat')}  "
        f"v_bo_km_s={run_config.get('v_bo_km_s')}  "
        f"n_satellites={run_config.get('n_satellites_selected')}"
    )


def _scan_local() -> list[str]:
    matches: list[str] = []
    for json_path in sorted(RESULTS_DIR.rglob("*_result.json")):
        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"  [warn] could not read {json_path}: {exc}")
            continue
        run_config = data.get("run_config", {})
        if run_config.get("is_principal_set") is True:
            matches.append(_summarize(run_config, str(json_path)))
    return matches


def _scan_s3() -> list[str]:
    import boto3

    s3 = boto3.client("s3")
    paginator = s3.get_paginator("list_objects_v2")
    matches: list[str] = []
    checked = 0
    for prefix in ("country/", "prespecified/"):
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".json") or key.endswith("_config.json"):
                    continue
                checked += 1
                body = s3.get_object(Bucket=S3_BUCKET, Key=key)["Body"].read()
                try:
                    data = json.loads(body)
                except json.JSONDecodeError as exc:
                    print(f"  [warn] could not parse s3://{S3_BUCKET}/{key}: {exc}")
                    continue
                run_config = data.get("run_config", {})
                if run_config.get("is_principal_set") is True:
                    matches.append(_summarize(run_config, f"s3://{S3_BUCKET}/{key}"))
    print(f"  (checked {checked} S3 result JSONs)")
    return matches


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local", action="store_true", help="Scan local results/ directory.")
    parser.add_argument("--s3", action="store_true", help="Scan the S3 results bucket.")
    args = parser.parse_args()
    if not args.local and not args.s3:
        parser.error("Specify at least one of --local or --s3.")

    all_matches: list[str] = []
    if args.local:
        print(f"Scanning local results under {RESULTS_DIR.resolve()} ...")
        all_matches.extend(_scan_local())
    if args.s3:
        print(f"Scanning s3://{S3_BUCKET} ...")
        all_matches.extend(_scan_s3())

    print(f"\n{len(all_matches)} principal-set result(s) found:\n")
    for m in all_matches:
        print(m)


if __name__ == "__main__":
    main()
