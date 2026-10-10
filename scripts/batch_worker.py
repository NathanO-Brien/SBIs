"""
batch_worker.py

Cloud-native SCLP optimization script for AWS Batch.

Mirrors lee_test2.py exactly — country-mode target shell or prespecified
target points, Lee exact-J2 seed families (anchored from the target latitudes,
shared repeat period) × RAAN seed grid, greedy seed selection — but
reads all problem inputs from a JSON config file stored in S3 and writes all
outputs back to S3.  Designed to run inside a Docker container on AWS Batch.
Submitted via cloud_run.py (single job) or cloud_sweep.py (parameter sweep).

Environment variables (the only three needed):
    S3_CONFIG   Full S3 URI of the JSON config file to load.
                Format: s3://<bucket>/<key>
    S3_BUCKET   S3 bucket where all outputs are written.
    S3_PREFIX   S3 key prefix for this job's outputs, e.g. "results/run-001"
                (no trailing slash).

Config file schema (JSON) -- required/most-relevant keys shown; see
_CONFIG_DEFAULTS below for the full set:
    {
        "job_name":                     "china_450km",   // optional; auto-timestamped if omitted
        "run_name":                     "campaign_7",    // optional; saved into run_config only
        "is_principal_set":             false,           // optional, default false
        "target_mode":                  "country",       // "country" or "prespecified", default "country"
        "target_country":               "China",         // required in country mode
        "target_points":                null,            // [[name, lat, lon], ...], required in prespecified mode
        "target_shell_n_points":        15000,           // optional, default 15000 (country mode)
        "nd_repeat":                    3,               // optional, default 3 (days in the shared repeat period)
        "altitude_min_km":              200.0,           // optional; the upper altitude bound is intercept_alt + R_max
        "anchor_lat_margin_deg":        0.0,             // optional, default 0
        "raan_step_deg":                0.25,            // optional, default 0.25
        "v_bo_km_s":                    6.0,             // required
        "salvo_size":                   1,               // required
        "n_interceptors_per_sat":       1,               // required
        "mip_gap":                      0.01,            // optional, default 0.01
        "time_limit_s":                 3600             // required
    }

Outputs uploaded flat to s3://<S3_BUCKET>/[<S3_PREFIX>/]runs/ (no subfolders). <run> is the
short run name from sbi_coverage.runnames, e.g. CNP-s046-d15-v6-g10-b170-k0-i1-261009a-6dff:
    <run>_result.csv        Orbital elements of selected satellites.
    <run>_result.json       Full simulation config sidecar
                            (compatible with coverage_from_sclp.py).
    <run>_gurobi.log        Gurobi solver log.
    <run>_config.json       Copy of the input config for traceability.
"""
from __future__ import annotations

import datetime
import io
import json
import math
import os
import sys
import time
from pathlib import Path

import boto3
import numpy as np

from sbi_coverage.core.config import CoverageConfig, EarthConstants, SimConfig
from sbi_coverage.core.shell import Shell
from sbi_coverage.runnames import encode_run_name, next_seq, parse_filename, settings_from_config, settings_key
from sbi_coverage.core.simulate import run_simulation
from sbi_coverage.Lee import (
    build_orb_elems_table,
    build_seed_pool,
    family_satellite_counts,
    filter_seed_pool,
    plan_seed_families,
    print_orb_table,
    save_sclp_result,
    seed_run_config,
    simulate_seed_pool,
    solve_sclp,
)


# ---------------------------------------------------------------------------
# Fixed algorithm parameters not exposed in the config file.
# ---------------------------------------------------------------------------
_PROPAGATOR: str                = "Nominal_Propagator"


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS: dict = {
    "job_name":                     None,    # auto-generated if omitted
    # Run metadata -- saved into run_config only, no other effect
    "run_name":                     None,
    "is_principal_set":             False,
    "is_test_run":                  False,   # prefixes the run name with TEST-
    "scenario":                     None,    # short scenario code for the run name (e.g. "NK"); inferred for countries
    # Target — exactly one of target_country or target_points must be set
    "target_mode":                  "country",   # "country" or "prespecified"
    "target_country":               None,
    "target_points":                None,        # list of [name, lat, lon] for prespecified mode
    "target_shell_n_points":        15000,
    # Seed families (Lee exact J2): altitude/inclination are derived from the
    # target latitudes, not given.  Upper altitude bound = intercept_alt + R_max.
    "nd_repeat":                    3,
    "altitude_min_km":              200.0,
    "anchor_lat_margin_deg":        0.0,
    "raan_step_deg":                0.25,
    "v_bo_km_s":                    None,    # required
    "salvo_size":                   None,    # required
    "n_interceptors_per_sat":       None,    # required
    "mip_gap":                      0.01,
    "time_limit_s":                 None,    # required
    # Seed selection: worst-case-first, margin-driven build + trim (shared with
    # lee_test2.py via sbi_coverage.Lee.seed_pipeline).
    "margin_multiplier":            5.0,
    "trim_margin_multiplier":       2.0,
    "max_seeds_safety_cap":         200,
    # Simulation parameters
    "dt_s":                         120.0,
    # Old "t_window_s" (time from target launch to intercept, assumed ==
    # interceptor flyout time) split into three inputs -- see CoverageConfig's
    # docstring for the derivation. Defaults (0, 0) leave
    # target_missile_burnout_time_s == the old t_window_s value, unchanged.
    "detection_time_s":             0.0,
    "decision_time_s":              0.0,
    "target_missile_burnout_time_s": 170.0,
    "a_g":                          10.0,
    "intercept_alt_km":             200.0,
    "min_elev_deg":                 0.0,
    "seed_workers":                 8,
    # Solver tuning -- intentionally neutral (Gurobi's own defaults) unless a
    # config explicitly overrides one of these; the schema still accepts
    # overrides if per-run tuning is worth reintroducing later.
    "mip_focus":                    0,
    "heuristics_frac":              0.05,
    "solver_params":                None,
    # If set, terminate the solve as soon as a live incumbent's gap drops
    # below this value, regardless of time_limit_s -- see solve_sclp's
    # stop_at_gap docstring. None (default): no early stop beyond
    # mip_gap/time_limit_s.
    "stop_at_gap":                  None,
}

_REQUIRED_CONFIG_KEYS = {
    "v_bo_km_s",
    "salvo_size",
    "n_interceptors_per_sat",
    "time_limit_s",
}


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def _resolve_run_id(settings: dict, bucket: str, flat_prefix: str) -> str:
    """Run name = readable settings + today's date + first unused same-day sequence letter."""
    date = datetime.datetime.now(datetime.timezone.utc).date()
    existing: list[str] = []
    try:
        s3 = boto3.client("s3")
        stem = encode_run_name(settings, date, "a")
        head = stem[: stem.index(date.strftime("%y%m%d"))]
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket, Prefix=f"{flat_prefix}/{head}"):
            existing += [o["Key"].rsplit("/", 1)[-1] for o in page.get("Contents", [])]
    except Exception as exc:  # listing is best-effort; the worker must not fail on it
        print(f"[batch_worker] WARNING: could not list existing runs ({exc}); using seq 'a'", flush=True)
    names = [parse_filename(k)[0] for k in existing if parse_filename(k)]
    return encode_run_name(settings, date, next_seq(names, settings, date))


def _require_env(name: str) -> str:
    """Return the value of a required environment variable or exit cleanly."""
    val = os.environ.get(name, "").strip()
    if not val:
        print(
            f"[batch_worker] ERROR: required environment variable '{name}' is not set.",
            flush=True,
        )
        sys.exit(1)
    return val


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split 's3://bucket/key/path' into ('bucket', 'key/path')."""
    if not uri.startswith("s3://"):
        print(
            f"[batch_worker] ERROR: S3_CONFIG must start with s3://, got: {uri}",
            flush=True,
        )
        sys.exit(1)
    parts = uri[5:].split("/", 1)
    if len(parts) != 2 or not parts[1]:
        print(
            f"[batch_worker] ERROR: S3_CONFIG must include a key, got: {uri}",
            flush=True,
        )
        sys.exit(1)
    return parts[0], parts[1]


def _s3_download_json(bucket: str, key: str) -> dict:
    """Download and parse a JSON file from S3."""
    s3  = boto3.client("s3")
    buf = io.BytesIO()
    s3.download_fileobj(bucket, key, buf)
    return json.loads(buf.getvalue().decode("utf-8"))


def _s3_upload(local_path: str | Path, bucket: str, key: str) -> None:
    """Upload a local file to S3 and print a confirmation line to CloudWatch."""
    s3 = boto3.client("s3")
    s3.upload_file(str(local_path), bucket, key)
    print(f"[batch_worker] Uploaded  s3://{bucket}/{key}", flush=True)


def _s3_upload_bytes(data: bytes, bucket: str, key: str) -> None:
    """Upload raw bytes to S3 (used for the config echo)."""
    s3 = boto3.client("s3")
    s3.put_object(Body=data, Bucket=bucket, Key=key)
    print(f"[batch_worker] Uploaded  s3://{bucket}/{key}", flush=True)


# ---------------------------------------------------------------------------
# Config loading and validation
# ---------------------------------------------------------------------------

def _load_config(s3_uri: str) -> dict:
    """Download the config JSON from S3, apply defaults, and validate."""
    bucket, key = _parse_s3_uri(s3_uri)
    print(f"[batch_worker] Loading config from s3://{bucket}/{key}", flush=True)
    raw = _s3_download_json(bucket, key)

    cfg = dict(_CONFIG_DEFAULTS)
    cfg.update(raw)

    missing = [k for k in _REQUIRED_CONFIG_KEYS if cfg.get(k) is None]
    if missing:
        print(
            f"[batch_worker] ERROR: config file is missing required keys: {missing}",
            flush=True,
        )
        sys.exit(1)

    # Validate target mode and required target inputs
    cfg["target_mode"] = str(cfg.get("target_mode", "country")).strip().lower()
    if cfg["target_mode"] == "country" and not cfg.get("target_country"):
        print(
            "[batch_worker] ERROR: target_mode='country' requires 'target_country' in config.",
            flush=True,
        )
        sys.exit(1)
    if cfg["target_mode"] == "prespecified":
        pts = cfg.get("target_points")
        if not pts or not isinstance(pts, list) or len(pts) == 0:
            print(
                "[batch_worker] ERROR: target_mode='prespecified' requires 'target_points' "
                "(non-empty list of [name, lat, lon]).",
                flush=True,
            )
            sys.exit(1)

    if cfg["job_name"] is None:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
        cfg["job_name"] = f"sclp_{ts}"

    # Type coercions
    cfg["time_limit_s"]                 = float(cfg["time_limit_s"])
    cfg["mip_gap"]                      = float(cfg["mip_gap"])
    cfg["salvo_size"]                   = int(cfg["salvo_size"])
    cfg["n_interceptors_per_sat"]       = int(cfg["n_interceptors_per_sat"])
    cfg["v_bo_km_s"]                    = float(cfg["v_bo_km_s"])
    cfg["target_shell_n_points"]        = (
        int(cfg["target_shell_n_points"]) if cfg.get("target_shell_n_points") is not None else None
    )
    cfg["target_country"]               = (
        str(cfg["target_country"]).strip() if cfg.get("target_country") is not None else None
    )
    cfg["nd_repeat"]                    = int(cfg["nd_repeat"])
    cfg["altitude_min_km"]              = float(cfg["altitude_min_km"])
    cfg["anchor_lat_margin_deg"]        = float(cfg["anchor_lat_margin_deg"])
    cfg["raan_step_deg"]                = float(cfg["raan_step_deg"])
    cfg["margin_multiplier"]            = float(cfg["margin_multiplier"])
    cfg["trim_margin_multiplier"]       = float(cfg["trim_margin_multiplier"])
    cfg["max_seeds_safety_cap"]         = int(cfg["max_seeds_safety_cap"])
    cfg["dt_s"]                         = float(cfg["dt_s"])
    cfg["detection_time_s"]             = float(cfg["detection_time_s"])
    cfg["decision_time_s"]              = float(cfg["decision_time_s"])
    cfg["target_missile_burnout_time_s"] = float(cfg["target_missile_burnout_time_s"])
    cfg["a_g"]                          = float(cfg["a_g"])
    cfg["intercept_alt_km"]             = float(cfg["intercept_alt_km"])
    cfg["min_elev_deg"]                 = float(cfg["min_elev_deg"])
    cfg["seed_workers"]                 = int(cfg["seed_workers"])
    cfg["mip_focus"]                    = int(cfg["mip_focus"])
    cfg["heuristics_frac"]              = float(cfg["heuristics_frac"])
    cfg["solver_params"]                = (
        dict(cfg["solver_params"]) if cfg.get("solver_params") is not None else None
    )
    cfg["run_name"]                     = str(cfg["run_name"]) if cfg.get("run_name") is not None else None
    cfg["is_principal_set"]             = bool(cfg.get("is_principal_set", False))
    cfg["is_test_run"]                  = bool(cfg.get("is_test_run", False))
    cfg["stop_at_gap"]                  = (
        float(cfg["stop_at_gap"]) if cfg.get("stop_at_gap") is not None else None
    )

    return cfg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # Load config from S3
    # ------------------------------------------------------------------
    s3_config_uri = _require_env("S3_CONFIG")
    s3_bucket     = _require_env("S3_BUCKET")
    s3_base       = os.environ.get("S3_PREFIX", "").strip().rstrip("/")

    cfg    = _load_config(s3_config_uri)
    job    = cfg["job_name"]
    bucket = s3_bucket
    run_settings = settings_from_config(cfg)
    flat_prefix  = f"{s3_base}/runs" if s3_base else "runs"
    run_id       = _resolve_run_id(run_settings, bucket, flat_prefix)

    target_mode     = cfg["target_mode"]
    country         = cfg.get("target_country")
    target_points   = cfg.get("target_points")   # list of [name, lat, lon] or None

    # ------------------------------------------------------------------
    # Startup banner — fully visible in CloudWatch Logs
    # ------------------------------------------------------------------
    if target_mode == "prespecified":
        _target_desc = f"prespecified ({len(target_points)} points)" if target_points else "prespecified"
    else:
        _target_desc = f"country: {country}"

    print("=" * 60,                                                              flush=True)
    print("batch_worker.py  —  SBI SCLP Cloud Runner",                          flush=True)
    print("=" * 60,                                                              flush=True)
    print(f"  Job name       : {job}",                                           flush=True)
    print(f"  Run name       : {run_id}",                                        flush=True)
    print(f"  Config source  : {s3_config_uri}",                                 flush=True)
    print(f"  Target         : {_target_desc}",                                  flush=True)
    print(f"  Seed families  : N_D={cfg['nd_repeat']}, altitude from {cfg['altitude_min_km']:.0f} km to intercept alt + R_max",  flush=True)
    print(f"  v_bo           : {cfg['v_bo_km_s']} km/s",                        flush=True)
    print(f"  Salvo size     : {cfg['salvo_size']}",                             flush=True)
    print(f"  Interceptors   : {cfg['n_interceptors_per_sat']} per sat",         flush=True)
    print(f"  MIP gap        : {cfg['mip_gap'] * 100:.2f}%",                     flush=True)
    print(f"  Time limit     : {cfg['time_limit_s']:.0f} s"
          f"  ({cfg['time_limit_s'] / 3600:.2f} h)",                             flush=True)
    print(f"  S3 bucket      : {bucket}  (all outputs flat under runs/)",  flush=True)
    print("=" * 60,                                                              flush=True)

    # ------------------------------------------------------------------
    # Coverage geometry
    # ------------------------------------------------------------------
    earth = EarthConstants()

    cov = CoverageConfig(
        earth=earth,
        detection_time_s=cfg["detection_time_s"],
        decision_time_s=cfg["decision_time_s"],
        target_missile_burnout_time_s=cfg["target_missile_burnout_time_s"],
        v_bo_km_s=cfg["v_bo_km_s"],
        a_g=cfg["a_g"],
        intercept_alt_km=cfg["intercept_alt_km"],
        min_elev_deg=cfg["min_elev_deg"],
    )
    print(
        f"Interceptor engagement time: {cov.interceptor_engagement_time_s:.2f} s "
        f"(burnout {cfg['target_missile_burnout_time_s']:.0f}s - detection {cfg['detection_time_s']:.0f}s "
        f"- decision {cfg['decision_time_s']:.0f}s)",
        flush=True,
    )
    print(f"\nComputed R_max: {cov.max_range_km:.2f} km", flush=True)

    # ------------------------------------------------------------------
    # Target shell — country or prespecified
    # ------------------------------------------------------------------
    if target_mode == "prespecified":
        labels    = [str(p[0]) for p in target_points]
        lats_deg  = np.array([float(p[1]) for p in target_points], dtype=np.float64)
        lons_deg  = np.array([float(p[2]) for p in target_points], dtype=np.float64)
        shell     = Shell.from_points(
            earth,
            lat_deg=lats_deg,
            lon_deg=lons_deg,
            shell_alt_km=cov.intercept_alt_km,
            meta_extra={"labels": labels, "roi": "prespecified target points"},
        )
        n_targets = len(labels)
    else:
        base_shell = Shell(
            earth,
            n_points=cfg["target_shell_n_points"],
            shell_alt_km=cov.intercept_alt_km,
            analysis_shell_mode="full",
        )
        shell = base_shell.mask_country([country])
        if shell.n_points_used <= 0:
            print(
                f"[batch_worker] ERROR: No shell points found inside country={country!r}. "
                "Check the country name against the geodata database.",
                flush=True,
            )
            sys.exit(1)
        lats_deg  = np.asarray(shell.lat_deg, dtype=np.float64)
        lons_deg  = np.asarray(shell.lon_deg, dtype=np.float64)
        n_targets = shell.n_points_used
        labels    = [f"{country} target {i}" for i in range(n_targets)]
        shell.meta["labels"] = labels

    # ------------------------------------------------------------------
    # Seed families: anchored top-down from the target latitudes, synchronized
    # repeat period (Lee Eq. 33), exact secular J2.  Altitude is an OUTPUT.
    # ------------------------------------------------------------------
    try:
        plan = plan_seed_families(
            [[labels[i], float(lats_deg[i]), float(lons_deg[i])] for i in range(n_targets)],
            earth=earth,
            N_D=cfg["nd_repeat"],
            alt_min_km=cfg["altitude_min_km"],
            anchor_lat_margin_deg=cfg["anchor_lat_margin_deg"],
            cov=cov,
            dt_target_s=cfg["dt_s"],
        )
    except ValueError as exc:
        print(f"[batch_worker] ERROR: seed family planning failed: {exc}", flush=True)
        sys.exit(1)
    families      = plan["families"]
    N_D           = int(plan["N_D"])
    L             = int(plan["L"])
    _dt_s_actual  = float(plan["dt_s"])
    horizon_s     = (L - 1) * _dt_s_actual
    anchor        = plan["anchor"]
    actual_alt_km = float(anchor["alt_km"])   # anchor altitude: labels the S3 prefix
    gen = plan["generation"]
    print(
        f"\nSeed families (N_D={N_D}, altitude {plan['alt_bounds_km'][0]:.0f}-{plan['alt_bounds_km'][1]:.0f} km "
        f"= shell + R_max, target latitudes {plan['lat_min_deg']:.2f}-{plan['lat_max_deg']:.2f} deg, "
        f"span {gen['target_lat_span_deg']:.2f}):",
        flush=True,
    )
    for f in families:
        red = f" [reducible -> {f['reduced_ratio']}]" if f["reducible"] else ""
        print(f"  {'anchor' if f['is_anchor'] else '      '} {f['N_P']:3d}:{f['N_D']}  i={f['inc_deg']:7.3f} deg  "
              f"alt={f['alt_km']:7.2f} km  reach={f['reach_deg']:.2f} deg  floor={f['inc_floor_deg']:.2f} deg{red}", flush=True)
    print(f"  unique altitude/inclination combos: generated {gen['n_unique_alt_inc_combos_all_candidates']} "
          f"(chosen ladder {gen['n_unique_alt_inc_combos_chosen_ladder']}), kept {gen['n_unique_alt_inc_combos_kept']}", flush=True)
    print(f"Shared repeat period T_r={plan['T_r_s']:.3f} s  L={L} steps  dt={_dt_s_actual:.6f} s  "
          f"horizon={horizon_s:.1f} s ({horizon_s / 3600:.3f} h)", flush=True)

    sim = SimConfig(
        horizon_s=horizon_s,
        dt_s=_dt_s_actual,
        analysis_matrix_dtype="uint8",
        use_j2=True,
        use_drag=False,
    )

    prefix = flat_prefix

    print(f"\nTarget mode: {target_mode}  n_targets={n_targets}", flush=True)
    print(f"[batch_worker] S3 prefix: {prefix}", flush=True)

    # ------------------------------------------------------------------
    # Seed pool (family x RAAN), simulation, filter (see Lee/seed_pipeline.py)
    # ------------------------------------------------------------------
    def _log(msg: str) -> None:
        print(msg, flush=True)

    r_required = int(np.ceil(cfg["salvo_size"] / cfg["n_interceptors_per_sat"]))
    seed_meta, sub_layers = build_seed_pool(plan, cfg["raan_step_deg"], earth)
    print(f"\nSeed grid: {len(seed_meta)} seeds  (RAAN step {cfg['raan_step_deg']:.2f} deg within each family's window)", flush=True)
    for f in families:
        print(f"  family {f['N_P']}:{f['N_D']}: {f['n_raan_offsets']} RAAN offsets in [0, {f['raan_interval_deg']:.3f}) deg", flush=True)

    v0_list = simulate_seed_pool(
        sub_layers, sim=sim, cov=cov, earth=earth, shell=shell,
        propagator=_PROPAGATOR, workers=cfg["seed_workers"], log=_log,
    )
    try:
        pool = filter_seed_pool(
            v0_list, sub_layers, seed_meta, plan, r_required,
            margin_multiplier=cfg["margin_multiplier"],
            trim_margin_multiplier=cfg["trim_margin_multiplier"],
            max_seeds=cfg["max_seeds_safety_cap"],
            log=_log,
        )
    except ValueError as exc:
        print(f"[batch_worker] ERROR: seed filtering failed: {exc}", flush=True)
        sys.exit(1)
    param_V    = pool["param_V"]
    sub_layers = pool["sub_layers"]
    seed_meta  = pool["seed_meta"]
    ncols      = pool["ncols"]

    param_r = r_required * np.ones((L, n_targets), dtype=np.float64)
    print(
        f"\nCoverage requirement: salvo_size={cfg['salvo_size']}, "
        f"interceptors_per_sat={cfg['n_interceptors_per_sat']}, "
        f"r_required={r_required} satellite(s) in view per time step",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Solve SCLP
    # ------------------------------------------------------------------
    log_local = Path(f"/tmp/{run_id}_gurobi.log")

    print("\n" + "=" * 60, flush=True)
    print("Solving SCLP",           flush=True)
    print("=" * 60,                 flush=True)

    result = solve_sclp(
        param_V=param_V,
        param_r=param_r,
        n_targets=n_targets,
        apply_rgt_symmetry_break=False,
        mip_gap=cfg["mip_gap"],
        time_limit_s=cfg["time_limit_s"],
        log_file=log_local,
        verbose=True,
        mip_focus=cfg["mip_focus"],
        heuristics_frac=cfg["heuristics_frac"],
        solver_params=cfg["solver_params"],
        stop_at_gap=cfg["stop_at_gap"],
    )

    # ------------------------------------------------------------------
    # Result summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60,                                                       flush=True)
    print(f"Status  : {result.status_str}",                                      flush=True)
    print(f"Selected: {result.n_satellites} satellite(s)",                       flush=True)
    print(f"Cost    : {result.obj_val:.1f}",                                     flush=True)
    _gap_pct = result.mip_gap * 100
    print(
        f"MIP gap : {_gap_pct:.4f}%" if math.isfinite(_gap_pct)
        else "MIP gap : N/A (no lower bound established)",
        flush=True,
    )
    print(
        f"Modeling: {result.time_modeling_s:.2f} s"
        f"   Solve: {result.time_optimization_s:.2f} s",
        flush=True,
    )
    print("=" * 60, flush=True)

    orb_elems = build_orb_elems_table(sub_layers, L, ncols)
    print_orb_table(orb_elems, result.selected_slots)

    # ------------------------------------------------------------------
    # Build run_config metadata block (mirrors lee_test2.py)
    # ------------------------------------------------------------------
    per_family_counts = family_satellite_counts(result.selected_slots, ncols, seed_meta, families)
    print("Satellites per family: " + ", ".join(f"{k}: {v}" for k, v in per_family_counts.items()), flush=True)
    seed_cfg = seed_run_config(
        plan, pool, per_family_counts,
        raan_step_deg=cfg["raan_step_deg"],
        margin_multiplier=cfg["margin_multiplier"],
        trim_margin_multiplier=cfg["trim_margin_multiplier"],
        max_seeds=cfg["max_seeds_safety_cap"],
    )

    run_config = {
        # --- Run metadata ---
        "run_name":                       cfg["run_name"],
        "run_id":                         run_id,
        "run_settings_key":               settings_key(run_settings),
        "is_principal_set":               cfg["is_principal_set"],
        "is_test_run":                    cfg["is_test_run"],
        # --- Target ---
        "target_mode":                    target_mode,
        "target_country":                 country if target_mode == "country" else None,
        "target_points":                  [list(p) for p in target_points] if target_mode == "prespecified" else None,
        "target_shell_n_points":          cfg["target_shell_n_points"] if target_mode == "country" else None,
        "n_targets":                      int(n_targets),
        # --- Engagement physics ---
        "dt_s":                           cfg["dt_s"],
        "detection_time_s":               cfg["detection_time_s"],
        "decision_time_s":                cfg["decision_time_s"],
        "target_missile_burnout_time_s":  cfg["target_missile_burnout_time_s"],
        "interceptor_engagement_time_s":  float(cov.interceptor_engagement_time_s),
        "v_bo_km_s":                      cfg["v_bo_km_s"],
        "a_g":                            cfg["a_g"],
        "intercept_alt_km":               cfg["intercept_alt_km"],
        "min_elev_deg":                   cfg["min_elev_deg"],
        "r_max_km":                       float(cov.max_range_km),
        # --- Orbit model, seed generation, seed filter (shared block) ---
        **seed_cfg,
        # --- Intercept salvo ---
        "salvo_size":                     cfg["salvo_size"],
        "n_interceptors_per_sat":         cfg["n_interceptors_per_sat"],
        "r_required":                     int(r_required),
        # --- Solver ---
        "mip_gap_target":                 cfg["mip_gap"],
        "time_limit_s":                   cfg["time_limit_s"],
        "mip_focus":                      cfg["mip_focus"],
        "heuristics_frac":                cfg["heuristics_frac"],
        "solver_params":                  cfg["solver_params"],
        "stop_at_gap":                    cfg["stop_at_gap"],
        # --- Result summary ---
        "timestamp_utc":                  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result_status":                  result.status_str,
        "mip_gap_achieved":               None if not math.isfinite(result.mip_gap) else float(result.mip_gap),
        "n_satellites_selected":          int(result.n_satellites),
        "obj_val":                        None if not math.isfinite(result.obj_val)   else float(result.obj_val),
        "obj_bound":                      None if not math.isfinite(result.obj_bound) else float(result.obj_bound),
        "time_modeling_s":                float(result.time_modeling_s),
        "time_optimization_s":            float(result.time_optimization_s),
    }

    # ------------------------------------------------------------------
    # Save results to /tmp then upload everything to S3
    # ------------------------------------------------------------------
    csv_local  = Path(f"/tmp/{run_id}_result.csv")
    json_local = csv_local.with_suffix(".json")

    targets_payload = [
        {"name": labels[i], "lat_deg": float(lats_deg[i]), "lon_deg": float(lons_deg[i])}
        for i in range(n_targets)
    ]
    bc_kg_m2 = float(np.asarray(sub_layers[0].phys.bc_kg_m2).ravel()[0])

    _countries_arg = [country] if target_mode == "country" else []
    save_sclp_result(
        result,
        orb_elems,
        csv_local,
        cov=cov,
        sim=sim,
        targets=targets_payload,
        countries=_countries_arg,
        propagator=_PROPAGATOR,
        bc_kg_m2=bc_kg_m2,
        run_config=run_config,
    )

    print(f"\n[batch_worker] Uploading results to s3://{bucket}/{prefix}/", flush=True)

    _s3_upload(csv_local,  bucket, f"{prefix}/{run_id}_result.csv")
    _s3_upload(json_local, bucket, f"{prefix}/{run_id}_result.json")

    if log_local.exists():
        _s3_upload(log_local, bucket, f"{prefix}/{run_id}_gurobi.log")

    # Echo the input config alongside outputs for full traceability
    config_echo = json.dumps(cfg, indent=2, default=str).encode("utf-8")
    _s3_upload_bytes(config_echo, bucket, f"{prefix}/{run_id}_config.json")

    print(f"\n[batch_worker] All outputs at s3://{bucket}/{prefix}/", flush=True)
    print("[batch_worker] Done.",                                       flush=True)


if __name__ == "__main__":
    main()
