"""Cloud-only view of the run results: list, index and read runs straight from S3 (no local files).

The GUI uses this module exclusively. A run's inputs come from its file name; its results (satellite count,
status, families, ...) come from the small ``_result.json`` sidecar, fetched in parallel and cached by
(key, ETag) so a refresh only downloads new or changed objects.
"""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .config import S3_BUCKET, S3_RUNS_PREFIX
from .index import _COLUMNS, fill_row_from_sidecar, normalize_sidecar
from .names import decode_run_name, parse_filename

_CACHE: dict[tuple[str, str, str], dict[str, Any]] = {}
_CACHE_LOCK = threading.Lock()


def list_run_files(s3, bucket: str = S3_BUCKET, prefix: str = S3_RUNS_PREFIX) -> dict[str, dict[str, dict[str, Any]]]:
    """{run name: {product: {key, size, etag, last_modified}}} for every recognizable file under ``prefix``."""
    runs: dict[str, dict[str, dict[str, Any]]] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            parsed = parse_filename(obj["Key"])
            if parsed and parsed[1]:
                runs.setdefault(parsed[0], {})[parsed[1]] = {
                    "key": obj["Key"], "size": obj.get("Size"), "etag": obj.get("ETag", ""),
                    "last_modified": obj.get("LastModified"),
                }
    return runs


def read_json(s3, bucket: str, key: str, etag: str = "") -> dict[str, Any]:
    """Download and parse a JSON object (cached by key+etag)."""
    ck = (bucket, key, etag)
    with _CACHE_LOCK:
        if etag and ck in _CACHE:
            return _CACHE[ck]
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    data = json.loads(body.decode("utf-8"))
    if etag:
        with _CACHE_LOCK:
            _CACHE[ck] = data
    return data


def read_bytes(s3, bucket: str, key: str) -> bytes:
    return s3.get_object(Bucket=bucket, Key=key)["Body"].read()


def read_sidecar(s3, bucket: str, files: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Normalized (schema-2 vocabulary) sidecar of one run from its file listing."""
    info = files["result.json"]
    return normalize_sidecar(read_json(s3, bucket, info["key"], info["etag"]))


def build_index(s3, bucket: str = S3_BUCKET, prefix: str = S3_RUNS_PREFIX, max_workers: int = 16,
                listing: dict | None = None):
    """DataFrame, one row per run in S3 (newest first); unreadable/failed runs are kept and flagged.

    Columns: the decoded name fields, results (n_satellites, result_status, mip_gap_achieved, ...), families,
    ``has_csv``, ``files`` (product -> key) and ``last_modified``.
    """
    import pandas as pd
    listing = listing if listing is not None else list_run_files(s3, bucket, prefix)

    def one(item):
        name, files = item
        row: dict[str, Any] = decode_run_name(name)
        row["files"] = {p: f["key"] for p, f in files.items()}
        row["last_modified"] = max((f["last_modified"] for f in files.values() if f.get("last_modified")), default=None)
        if "result.json" in files:
            try:
                fill_row_from_sidecar(row, read_json(s3, bucket, files["result.json"]["key"], files["result.json"]["etag"]),
                                      "result.csv" in files)
            except Exception as exc:
                row["result_status"] = "UNREADABLE"
                row["status_note"] = f"{type(exc).__name__}: {exc}"[:200]
        elif "failed.json" in files:
            row["result_status"] = "FAILED"
            try:
                row["status_note"] = str(read_json(s3, bucket, files["failed.json"]["key"], files["failed.json"]["etag"]).get("error", ""))[:200]
            except Exception:
                row["status_note"] = "failed (unreadable record)"
        else:
            row["result_status"] = "RUNNING / INCOMPLETE"   # config reserved the name but no result yet
        return row

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        rows = list(ex.map(one, listing.items()))
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_COLUMNS + ["files", "last_modified"])
    for c in _COLUMNS:
        if c not in df:
            df[c] = None
    extra = [c for c in df.columns if c not in _COLUMNS]
    return df[_COLUMNS + extra].sort_values(["date_iso", "seq"], ascending=False).reset_index(drop=True)
