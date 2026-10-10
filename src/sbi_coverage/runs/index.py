"""Index of run results in a flat folder and the flat S3 ``runs/`` prefix.

Run files are named ``<run name>_<product>`` (see :mod:`sbi_coverage.runs.names`), so the index is built
from the file names plus each run's ``_result.json`` sidecar -- no database. Importable on its own for
scripting::

    from sbi_coverage.runs.index import load_index
    df = load_index("results")
    df[(df.scenario == "CN") & (df.detection_time_s == 15)]

Sidecars written before the schema-2 vocabulary rename are read through ``normalize_sidecar``, so old
and new files appear side by side.
"""
from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any, Callable

from .config import S3_BUCKET, S3_RUNS_PREFIX
from .names import decode_run_name, encode_run_name, parse_filename

PRODUCTS = ("result.json", "result.csv", "config.json", "gurobi.log", "failed.json")
DEFAULT_SYNC_PRODUCTS = ("result.json", "result.csv", "config.json", "failed.json")

# pre-rename run_config key -> current key
_LEGACY_RC = {
    "v_bo_km_s": "burnout_velocity_km_s", "a_g": "max_accel_g", "actual_alt_km": "anchor_alt_km",
    "rgt_ratio": "anchor_ratio", "seeds_selected": "seeds_kept", "run_id": "run_name",
    "max_seeds_safety_cap": "max_seeds",
}


def normalize_sidecar(data: dict[str, Any]) -> dict[str, Any]:
    """View of a result JSON in the current (schema 2) vocabulary.

    Handles schema-1 files (no ``schema_version``): top-level n_satellites/status_str/mip_gap move into
    ``result``, cov keys ``v_bo_km_s``/``a_g`` are renamed, run_config keys are mapped (see _LEGACY_RC),
    and the old free-text ``run_name`` becomes ``label``. Unknown keys (drag/ballistic coefficient) are kept
    untouched and ignored by readers.
    """
    if data.get("schema_version", 1) >= 2:
        return data
    rc_in = dict(data.get("run_config") or {})
    rc: dict[str, Any] = {}
    for k, v in rc_in.items():
        rc[_LEGACY_RC.get(k, k)] = v
    if "label" not in rc and "run_name" in rc_in:
        rc["label"] = rc_in["run_name"]
    cov = dict(data.get("cov") or {})
    if "v_bo_km_s" in cov:
        cov["burnout_velocity_km_s"] = cov.pop("v_bo_km_s")
    if "a_g" in cov:
        cov["max_accel_g"] = cov.pop("a_g")
    result = {
        "n_satellites": data.get("n_satellites", rc_in.get("n_satellites_selected")),
        "status": data.get("status_str", rc_in.get("result_status")),
        "obj_val": data.get("obj_val"), "obj_bound": rc_in.get("obj_bound"),
        "mip_gap_achieved": data.get("mip_gap", rc_in.get("mip_gap_achieved")),
        "time_modeling_s": rc_in.get("time_modeling_s"), "time_optimization_s": rc_in.get("time_optimization_s"),
    }
    sim = dict(data.get("sim") or {})
    return {
        "schema_version": 1, "run_name": rc.get("run_name"), "timestamp_utc": rc_in.get("timestamp_utc", data.get("timestamp")),
        "result": result, "cov": cov, "sim": sim, "targets": data.get("targets", []),
        "countries": data.get("countries", []), "run_config": rc, "provenance": {},
    }


def read_sidecar(json_path: str | Path) -> dict[str, Any]:
    """Load a result JSON and normalize it to the current vocabulary."""
    return normalize_sidecar(json.loads(Path(json_path).read_text(encoding="utf-8")))


def _fill_from_sidecar(row: dict[str, Any], data: dict[str, Any], json_path: Path) -> dict[str, Any]:
    sc = normalize_sidecar(data)
    rc, res = sc["run_config"], sc["result"]
    row["n_satellites"] = res.get("n_satellites")
    row["result_status"] = res.get("status")
    row["mip_gap_achieved"] = res.get("mip_gap_achieved")
    row["time_optimization_s"] = res.get("time_optimization_s")
    for k in ("label", "orbit_model", "intercept_alt_km", "r_max_km", "anchor_alt_km", "anchor_inc_deg", "nd_repeat",
              "raan_step_deg", "margin_multiplier", "trim_margin_multiplier", "r_required", "n_targets", "wall_time_s"):
        row[k] = rc.get(k)
    solver = rc.get("solver") or {}
    row["time_limit_s"] = solver.get("time_limit_s", rc.get("time_limit_s"))
    fams = rc.get("families") or []
    row["families"] = ", ".join(
        f"{f.get('N_P', '?')}:{f.get('N_D', '?')}@{float(f.get('alt_km', float('nan'))):.0f}km x{f.get('satellites_selected', 0)}"
        for f in fams if isinstance(f, dict)
    )
    sg = rc.get("seed_generation") or {}
    row["target_lat_span_deg"] = sg.get("target_lat_span_deg")
    row["n_alt_inc_combos"] = sg.get("n_unique_alt_inc_combos_all_candidates")
    row["git_commit"] = (sc.get("provenance") or {}).get("git_commit")
    row["has_csv"] = json_path.with_suffix(".csv").exists()
    return row


def _row(json_path: Path) -> dict[str, Any] | None:
    parsed = parse_filename(json_path.name)
    if not parsed or parsed[1] not in ("result.json", "failed.json"):
        return None
    row: dict[str, Any] = decode_run_name(parsed[0])
    row["path"] = str(json_path)
    if parsed[1] == "failed.json":
        row["result_status"] = "FAILED"
        try:
            row["status_note"] = json.loads(json_path.read_text(encoding="utf-8")).get("error", "")[:200]
        except Exception:
            row["status_note"] = "failed (unreadable record)"
        return row
    try:
        return _fill_from_sidecar(row, json.loads(json_path.read_text(encoding="utf-8")), json_path)
    except Exception as exc:  # one bad/legacy file must never break the whole index
        row["status_note"] = f"unreadable sidecar: {type(exc).__name__}"
        return row


_COLUMNS = ["name", "scenario", "principal", "test", "date_iso", "seq", "salvo_size", "detection_time_s",
            "decision_time_s", "burnout_velocity_km_s", "max_accel_g", "target_missile_burnout_time_s",
            "n_interceptors_per_sat", "n_satellites", "result_status", "mip_gap_achieved", "families", "hash",
            "key", "path"]


def load_index(folder: str | Path):
    """DataFrame (one row per run, newest first) of every result/failed record in ``folder``."""
    import pandas as pd
    folder = Path(folder)
    paths = sorted(folder.glob("*_result.json")) + sorted(folder.glob("*_failed.json"))
    rows = [r for p in paths if (r := _row(p))]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=_COLUMNS)
    for c in _COLUMNS:
        if c not in df:
            df[c] = None
    extra = [c for c in df.columns if c not in _COLUMNS]
    return df[_COLUMNS + extra].sort_values(["date_iso", "seq"], ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# S3
# ---------------------------------------------------------------------------

def list_s3_runs(bucket: str = S3_BUCKET, prefix: str = S3_RUNS_PREFIX, s3=None,
                 name_prefix: str = "") -> dict[str, list[str]]:
    """{run name: [S3 keys of its products]} under the flat ``runs/`` prefix (optionally one name prefix)."""
    import boto3
    s3 = s3 or boto3.client("s3")
    runs: dict[str, list[str]] = {}
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=prefix + name_prefix):
        for obj in page.get("Contents", []):
            parsed = parse_filename(obj["Key"])
            if parsed:
                runs.setdefault(parsed[0], []).append(obj["Key"])
    return runs


def sync_from_s3(folder: str | Path, bucket: str = S3_BUCKET, prefix: str = S3_RUNS_PREFIX,
                 products: tuple[str, ...] = DEFAULT_SYNC_PRODUCTS, names: list[str] | None = None,
                 overwrite: bool = False, progress: Callable[[str], None] | None = None) -> list[str]:
    """Download runs (or files) missing from ``folder``; existing files are kept unless ``overwrite``.

    ``names`` restricts the sync to those run names (one targeted listing per name). Returns the names of
    runs that gained at least one file.
    """
    import boto3
    s3 = boto3.client("s3")
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    if names is None:
        listings = list_s3_runs(bucket, prefix, s3)
    else:
        listings = {}
        for n in dict.fromkeys(names):
            listings.update({k: v for k, v in list_s3_runs(bucket, prefix, s3, name_prefix=n).items() if k == n})
    got: list[str] = []
    for name, keys in listings.items():
        new = False
        for key in keys:
            prod = parse_filename(key)[1]
            dest = folder / key.rsplit("/", 1)[-1]
            if prod in products and (overwrite or not dest.exists()):
                s3.download_file(bucket, key, str(dest))
                new = True
        if new:
            got.append(name)
            if progress:
                progress(name)
    return got


def reserve_run_name(s3, bucket: str, settings: dict, config_echo: bytes, prefix: str = S3_RUNS_PREFIX) -> str:
    """Atomically claim the first free run name for these settings (today, UTC) and return it.

    The claim is a conditional create (``If-None-Match: *``) of ``<name>_config.json``, so two simultaneous
    jobs with identical settings get different sequence letters instead of overwriting each other.
    """
    today = datetime.datetime.now(datetime.timezone.utc).date()
    for seq in "abcdefghijklmnopqrstuvwxyz":
        name = encode_run_name(settings, today, seq)
        try:
            s3.put_object(Bucket=bucket, Key=f"{prefix}{name}_config.json", Body=config_echo, IfNoneMatch="*")
            return name
        except Exception as exc:  # botocore ClientError: 412 PreconditionFailed / 409 conflict -> try next letter
            code = str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))
            if code in ("PreconditionFailed", "ConditionalRequestConflict", "412", "409"):
                continue
            raise
    raise RuntimeError("more than 26 runs with identical settings today")


def _split_s3_target(t: str) -> tuple[str, str]:
    """(bucket, key prefix) from an s3:// URI; ('' , t) for a bare key."""
    if t.lower().startswith("s3://"):
        bucket, _, key = t[5:].partition("/")
        return bucket, key
    return "", t


def resolve_run_files(target: str | Path | None, folder: str | Path = "results") -> tuple[Path, Path]:
    """(result.csv, result.json) local paths for a run, downloading from S3 ``runs/`` if needed.

    ``target`` may be a run name, a local path to the ``_result.csv``/``_result.json`` (or a folder
    holding one), an S3 URI/key such as ``s3://<bucket>/runs/<name>_result.csv`` (its bucket and key
    prefix are honored), or None for the newest result in ``folder``. Files fetched from S3 are saved
    in ``folder``.
    """
    folder = Path(folder)
    if target is None or str(target).strip() == "":
        found = sorted(folder.glob("*_result.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not found:
            raise FileNotFoundError(f"No *_result.csv in {folder}; pass a run name, path or S3 URI")
        print(f"Auto-selected newest result: {found[0].name}")
        return found[0], found[0].with_suffix(".json")
    t = str(target).strip()
    is_s3 = t.lower().startswith("s3://")
    local = Path(t)
    if not is_s3 and local.is_dir():
        found = sorted(local.glob("*_result.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not found:
            raise FileNotFoundError(f"No *_result.csv in {local}")
        return found[0], found[0].with_suffix(".json")
    if not is_s3 and local.suffix in (".csv", ".json") and local.exists():
        stem = local.with_suffix("")
        return stem.with_suffix(".csv"), stem.with_suffix(".json")
    bucket, key = _split_s3_target(t)
    parsed = parse_filename(t.replace("\\", "/").rsplit("/", 1)[-1])
    if not parsed:
        raise ValueError(f"{target!r} is not a run name, existing result file/folder or S3 URI/key of a run")
    name = parsed[0]
    csv_p, json_p = folder / f"{name}_result.csv", folder / f"{name}_result.json"
    if not (csv_p.exists() and json_p.exists()):
        prefix = key.rsplit("/", 1)[0] + "/" if "/" in key else S3_RUNS_PREFIX
        bucket = bucket or S3_BUCKET
        print(f"[note] {name} not in {folder}; downloading from s3://{bucket}/{prefix} ...")
        sync_from_s3(folder, bucket=bucket, prefix=prefix, names=[name])
        if not (csv_p.exists() and json_p.exists()):
            raise FileNotFoundError(f"run {name} not found locally or in s3://{bucket}/{prefix}")
    return csv_p, json_p
