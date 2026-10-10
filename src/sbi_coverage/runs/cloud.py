"""AWS helpers shared by the launch scripts (cloud_run.py, cloud_sweep.py): config upload, Batch submit,
job-name sanitising, and the "does this run already exist?" check.
"""
from __future__ import annotations

import json
import re
from typing import Any, Mapping

from .config import (AWS_REGION, JOB_DEFINITION, JOB_QUEUE, S3_BUCKET, S3_RUNS_PREFIX, normalize_config,
                      settings_from_config)
from .names import decode_run_name, parse_filename, settings_key

S3_CONFIG_PREFIX = "configs/runs"


def sanitize_job_name(name: str) -> str:
    """AWS Batch job names must match [A-Za-z0-9_-] and be <= 128 characters."""
    name = re.sub(r"[^A-Za-z0-9_-]", "-", name)
    return re.sub(r"-{2,}", "-", name)[:128]


def clients():
    """(s3, batch) boto3 clients for the configured region."""
    import boto3
    return boto3.client("s3", region_name=AWS_REGION), boto3.client("batch", region_name=AWS_REGION)


def upload_config(s3, cfg: Mapping[str, Any], key: str, bucket: str = S3_BUCKET) -> str:
    """Upload the configuration JSON; returns its s3:// URI."""
    s3.put_object(Body=json.dumps(cfg, indent=2).encode("utf-8"), Bucket=bucket, Key=key)
    return f"s3://{bucket}/{key}"


def submit_job(batch, *, job_name: str, s3_config_uri: str, bucket: str = S3_BUCKET) -> str:
    """Submit one AWS Batch job running batch_worker.py on the configuration; returns the job id."""
    response = batch.submit_job(
        jobName=job_name, jobQueue=JOB_QUEUE, jobDefinition=JOB_DEFINITION,
        containerOverrides={"environment": [{"name": "S3_CONFIG", "value": s3_config_uri},
                                            {"name": "S3_BUCKET", "value": bucket}]},
    )
    return response["jobId"]


def s3_run_exists(s3, cfg: Mapping[str, Any], bucket: str = S3_BUCKET) -> list[str]:
    """Names of finished runs in S3 ``runs/`` with the same identity (any date) as ``cfg``; [] if none.

    Identity = readable settings tokens + result-affecting hash (see runs/names.py); solver limits are
    not part of it. Only runs that have a ``_result.json`` count (failed or queued jobs do not).
    """
    key = settings_key(settings_from_config(normalize_config(cfg)))
    found: set[str] = set()
    head = key.rsplit("-", 1)[0]                       # readable tokens; the date follows
    name_prefix = head[: head.index("-s")] if "-s" in head else ""   # scenario (+TEST-) prefix narrows the listing
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{S3_RUNS_PREFIX}{name_prefix}"):
        for obj in page.get("Contents", []):
            parsed = parse_filename(obj["Key"])
            if parsed and parsed[1] == "result.json" and decode_run_name(parsed[0])["key"] == key:
                found.add(parsed[0])
    return sorted(found)
