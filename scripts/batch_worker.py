"""
batch_worker.py -- AWS Batch container entry point for the SBI SCLP optimizer.

Runs inside the Docker container (see docker/Dockerfile), reads one JSON configuration from S3, runs
the full optimization (Lee/run_pipeline.run_sclp -- the same code lee_test2.py and
cloud_sweep.py --local use), and uploads every output product flat to s3://<bucket>/runs/.

Environment variables
---------------------
    S3_CONFIG   s3://bucket/key of the JSON configuration (set by the launchers)
    S3_BUCKET   bucket for the outputs (set by the launchers)
    GRB_WLSACCESSID / GRB_WLSSECRET / GRB_LICENSEID   Gurobi WLS credentials (Batch job secrets)
    GIT_COMMIT  optional; recorded in the result JSON (the image has no .git directory)

Configuration
-------------
The accepted keys, their defaults and type coercions live in sbi_coverage.runs.config
(CONFIG_DEFAULTS); configs from before the vocabulary rename (a_g, v_bo_km_s, dt_s, ...) are still
accepted. Required: salvo_size, n_interceptors_per_sat, burnout_velocity_km_s, time_limit_s, and either
target_points ([[name, lat, lon], ...]) or, in the deprecated country mode, target_country.

Outputs
-------
Flat under runs/ (no subfolders), all sharing the short run name (sbi_coverage.runs.names), e.g.
CNp-s046-d15-v6-g10-b170-k0-i1-261009a-6dff:
    <run>_result.csv     selected satellites (orbital elements, family, seed)
    <run>_result.json    sidecar for replay/analysis (schema_version 2)
    <run>_gurobi.log     Gurobi solver log
    <run>_config.json    the normalized input configuration (written first: it reserves the name)
    <run>_failed.json    only if the run failed (error + traceback); the job then exits non-zero
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from pathlib import Path

import boto3

from sbi_coverage.Lee.run_pipeline import run_sclp, write_failure_record
from sbi_coverage.runs.config import S3_RUNS_PREFIX, normalize_config, settings_from_config
from sbi_coverage.runs.index import reserve_run_name


def _require_env(name: str) -> str:
    val = os.environ.get(name, "").strip()
    if not val:
        print(f"[batch_worker] ERROR: required environment variable {name} is not set.", flush=True)
        sys.exit(1)
    return val


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    if not uri.startswith("s3://"):
        raise ValueError(f"expected an s3:// URI, got {uri!r}")
    bucket, _, key = uri[5:].partition("/")
    if not bucket or not key:
        raise ValueError(f"malformed S3 URI: {uri!r}")
    return bucket, key


def _s3_download_json(bucket: str, key: str) -> dict:
    buf = io.BytesIO()
    boto3.client("s3").download_fileobj(bucket, key, buf)
    return json.loads(buf.getvalue().decode("utf-8"))


def _s3_upload(local_path: str | Path, bucket: str, key: str) -> None:
    boto3.client("s3").upload_file(str(local_path), bucket, key)
    print(f"[batch_worker] Uploaded  s3://{bucket}/{key}", flush=True)


def main() -> None:
    s3_config_uri = _require_env("S3_CONFIG")
    bucket = _require_env("S3_BUCKET")

    _, config_key = _parse_s3_uri(s3_config_uri)
    print(f"[batch_worker] Loading config from {s3_config_uri}", flush=True)
    raw = _s3_download_json(*_parse_s3_uri(s3_config_uri))
    cfg = normalize_config(raw)

    # Reserve the run name atomically (conditional put of <run>_config.json) so two jobs with identical
    # settings can never both claim the same sequence letter.
    settings = settings_from_config(cfg)
    echo = json.dumps({k: v for k, v in cfg.items()}, indent=2, default=str).encode("utf-8")
    run_name = reserve_run_name(boto3.client("s3"), bucket, settings, echo)
    prefix = S3_RUNS_PREFIX.rstrip("/")

    print("=" * 60, flush=True)
    print("batch_worker.py  --  SBI SCLP Cloud Runner", flush=True)
    print(f"  Run name   : {run_name}", flush=True)
    print(f"  Config     : {s3_config_uri}", flush=True)
    print(f"  Target     : {cfg['target_mode']}"
          + (f" ({len(cfg['target_points'])} points)" if cfg["target_mode"] == "prespecified" else f" {cfg['target_country']}"),
          flush=True)
    print(f"  Salvo      : {cfg['salvo_size']}  x {cfg['n_interceptors_per_sat']} interceptor(s)/sat", flush=True)
    print(f"  Time limit : {cfg['time_limit_s']:.0f} s,  MIP gap {cfg['mip_gap'] * 100:.2f}%", flush=True)
    print("=" * 60, flush=True)

    out_dir = Path(tempfile.mkdtemp(prefix="sbi_run_"))
    try:
        summary = run_sclp(cfg, out_dir=out_dir, run_name=run_name, log=lambda m: print(m, flush=True))
    except BaseException as exc:
        failed = write_failure_record(out_dir, run_name, cfg, exc)
        _s3_upload(failed, bucket, f"{prefix}/{failed.name}")
        raise

    print(f"\n[batch_worker] Uploading results to s3://{bucket}/{prefix}/", flush=True)
    for path in (summary["csv_path"], summary["json_path"], summary["gurobi_log"]):
        if Path(path).exists():
            _s3_upload(path, bucket, f"{prefix}/{Path(path).name}")
    print("[batch_worker] Done.", flush=True)


if __name__ == "__main__":
    main()
