"""Index of run results in a flat folder (and the flat S3 ``runs/`` prefix).

Run files are named ``<run name>_<product>`` (see :mod:`sbi_coverage.runnames`), so the
index is built from the file names plus each run's ``_result.json`` sidecar -- no database.
Used by ``scripts/run_browser.py``; importable on its own for scripting::

    from sbi_coverage.runindex import load_index
    df = load_index("results")
    df[(df.scenario == "CN") & (df.detection_time_s == 15)]
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from .runnames import decode_run_name, parse_filename

S3_BUCKET = "sbi-optimization-runs"
S3_RUNS_PREFIX = "runs/"
PRODUCTS = ("result.json", "result.csv", "config.json", "gurobi.log")

# Columns pulled from run_config in the JSON sidecar (None when absent).
_RC_FIELDS = (
    "n_satellites_selected", "result_status", "mip_gap_achieved", "time_limit_s", "n_targets",
    "orbit_model", "intercept_alt_km", "r_max_km", "actual_alt_km", "anchor_inc_deg", "nd_repeat",
    "raan_step_deg", "margin_multiplier", "trim_margin_multiplier", "r_required", "run_name",
)


def _row(json_path: Path) -> dict[str, Any] | None:
    parsed = parse_filename(json_path.name)
    if not parsed or parsed[1] != "result.json":
        return None
    row: dict[str, Any] = decode_run_name(parsed[0])
    row["path"] = str(json_path)
    try:
        data = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        row["status_note"] = "unreadable json"
        return row
    rc = data.get("run_config", {}) or {}
    for k in _RC_FIELDS:
        row[k] = rc.get(k)
    row["n_satellites"] = rc.get("n_satellites_selected", data.get("n_satellites"))
    fams = rc.get("families") or []
    row["families"] = ", ".join(f"{f['N_P']}:{f['N_D']}@{f['alt_km']:.0f}km x{f.get('satellites_selected', 0)}" for f in fams)
    sg = rc.get("seed_generation") or {}
    row["target_lat_span_deg"] = sg.get("target_lat_span_deg")
    row["n_alt_inc_combos"] = sg.get("n_unique_alt_inc_combos_all_candidates")
    row["has_csv"] = json_path.with_suffix(".csv").exists()
    return row


def load_index(folder: str | Path):
    """DataFrame (one row per run, newest first) of every ``*_result.json`` in ``folder``."""
    import pandas as pd
    rows = [r for p in sorted(Path(folder).glob("*_result.json")) if (r := _row(p))]
    cols = ["name", "scenario", "principal", "test", "date_iso", "seq", "salvo_size", "detection_time_s", "decision_time_s",
            "v_bo_km_s", "a_g", "target_missile_burnout_time_s", "n_interceptors_per_sat", "n_satellites",
            "result_status", "mip_gap_achieved", "families", "hash", "key", "path"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=cols)
    extra = [c for c in df.columns if c not in cols]
    return df[cols + extra].sort_values(["date", "seq"], ascending=False).reset_index(drop=True)


def list_s3_runs(bucket: str = S3_BUCKET, prefix: str = S3_RUNS_PREFIX, s3=None) -> dict[str, list[str]]:
    """{run name: [S3 keys of its products]} for the flat ``runs/`` prefix."""
    import boto3
    s3 = s3 or boto3.client("s3")
    runs: dict[str, list[str]] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            parsed = parse_filename(obj["Key"])
            if parsed:
                runs.setdefault(parsed[0], []).append(obj["Key"])
    return runs


def sync_from_s3(folder: str | Path, bucket: str = S3_BUCKET, prefix: str = S3_RUNS_PREFIX,
                 products: tuple[str, ...] = ("result.json", "result.csv", "config.json"),
                 names: list[str] | None = None, progress: Callable[[str], None] | None = None) -> list[str]:
    """Download runs missing from ``folder`` (existing files are never overwritten).

    Returns the names of runs that gained at least one file.
    """
    import boto3
    s3 = boto3.client("s3")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    got: list[str] = []
    for name, keys in list_s3_runs(bucket, prefix, s3).items():
        if names is not None and name not in names:
            continue
        new = False
        for key in keys:
            prod = parse_filename(key)[1]
            dest = folder / key.rsplit("/", 1)[-1]
            if prod in products and not dest.exists():
                s3.download_file(bucket, key, str(dest))
                new = True
        if new:
            got.append(name)
            if progress:
                progress(name)
    return got


def resolve_run_files(target: str | Path | None, folder: str | Path = "results") -> tuple[Path, Path]:
    """(result.csv, result.json) local paths for a run, downloading from S3 ``runs/`` if needed.

    ``target`` may be a run name, a local path to the ``_result.csv``/``_result.json`` (or a
    folder holding one), an S3 URI/key such as ``s3://sbi-optimization-runs/runs/<name>_result.csv``,
    or None for the newest result in ``folder``.  Files fetched from S3 are saved in ``folder``.
    """
    folder = Path(folder)
    if target is None or str(target).strip() == "":
        found = sorted(folder.glob("*_result.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not found:
            raise FileNotFoundError(f"No *_result.csv in {folder}; pass a run name, path or S3 URI")
        print(f"Auto-selected newest result: {found[0].name}")
        return found[0], found[0].with_suffix(".json")
    t = str(target).strip()
    local = Path(t)
    if not t.lower().startswith("s3://") and local.is_dir():
        found = sorted(local.glob("*_result.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not found:
            raise FileNotFoundError(f"No *_result.csv in {local}")
        return found[0], found[0].with_suffix(".json")
    if not t.lower().startswith("s3://") and local.suffix in (".csv", ".json") and local.exists():
        stem = local.with_suffix("")
        return stem.with_suffix(".csv"), stem.with_suffix(".json")
    parsed = parse_filename(t.replace("\\", "/").rsplit("/", 1)[-1])
    if not parsed:
        raise ValueError(f"{target!r} is not a run name, existing result file/folder or S3 URI/key of a run")
    name = parsed[0]
    csv_p, json_p = folder / f"{name}_result.csv", folder / f"{name}_result.json"
    if not (csv_p.exists() and json_p.exists()):
        print(f"[note] {name} not in {folder}; downloading from s3://{S3_BUCKET}/{S3_RUNS_PREFIX} ...")
        sync_from_s3(folder, names=[name])
        if not (csv_p.exists() and json_p.exists()):
            raise FileNotFoundError(f"run {name} not found locally or in S3 runs/")
    return csv_p, json_p
