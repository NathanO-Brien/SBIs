"""
batch_worker.py

Cloud-native SCLP optimization script for AWS Batch.

Mirrors lee_test2.py exactly — country-mode target shell or prespecified
target points, full inclination × RAAN seed grid, greedy seed selection — but
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
        "target_altitude_km":           450.0,           // required
        "altitude_max_repeat_days":     2,               // optional, default 2
        "inclination_sweep_bounds_deg": [20.0, 90.0],   // optional, default [20, 90]
        "v_bo_km_s":                    6.0,             // required
        "salvo_size":                   1,               // required
        "n_interceptors_per_sat":       1,               // required
        "mip_gap":                      0.01,            // optional, default 0.01
        "time_limit_s":                 3600             // required
    }

Outputs uploaded to s3://<S3_BUCKET>/<S3_PREFIX>/:
    <job_name>.csv              Orbital elements of selected satellites.
    <job_name>.json             Full simulation config sidecar
                                (compatible with coverage_from_sclp.py).
    <job_name>_gurobi.log       Gurobi solver log.
    <job_name>_config.json      Copy of the input config for traceability.
"""
from __future__ import annotations

import datetime
import io
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import numpy as np

from sbi_coverage.core.config import CoverageConfig, EarthConstants, SimConfig
from sbi_coverage.core.shell import Shell
from sbi_coverage.core.simulate import run_simulation
from sbi_coverage.Lee import (
    build_orb_elems_table,
    build_param_V,
    candidate_rgt_ratios,
    param_V_summary,
    print_orb_table,
    rgt_common_ground_track_layer,
    save_sclp_result,
    seed_layer,
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
    # Target — exactly one of target_country or target_points must be set
    "target_mode":                  "country",   # "country" or "prespecified"
    "target_country":               None,
    "target_points":                None,        # list of [name, lat, lon] for prespecified mode
    "target_shell_n_points":        15000,
    "target_altitude_km":           None,    # required
    "inclination_sweep_bounds_deg": [20.0, 90.0],
    "v_bo_km_s":                    None,    # required
    "salvo_size":                   None,    # required
    "n_interceptors_per_sat":       None,    # required
    "mip_gap":                      0.01,
    "time_limit_s":                 None,    # required
    "altitude_max_repeat_days":     2,
    # Seed selection: rarity-weighted, margin-driven build + trim (mirrors
    # lee_test2.py) -- replaces the old fixed n_seeds_for_milp / per-
    # r_required lookup table, which didn't scale with target-set size and
    # could, via its unconditional "one per every live inclination"
    # guarantee, blow far past any reasonable margin on wide-inclination
    # scenarios before the fill phase ever ran.
    "margin_multiplier":            3.0,
    "trim_margin_multiplier":       1.25,
    "max_seeds_safety_cap":         200,
    "polar_band_deg":               [70.0, 90.0],
    "polar_band_divisor":           5.0,
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
    "inc_screen_step_deg":          1.0,
    "n_raan_offsets":               10,
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
    "target_altitude_km",
    "v_bo_km_s",
    "salvo_size",
    "n_interceptors_per_sat",
    "time_limit_s",
}


# ---------------------------------------------------------------------------
# Environment variable helpers
# ---------------------------------------------------------------------------

def _build_s3_prefix(cfg: dict, actual_alt_km: float, n_targets: int) -> str:
    """Build a hierarchical S3 prefix from run parameters.

    Structure:
        <target-mode>/<target-name>/target-points-<N>/
        intercept-alt-<X>km/orbit-alt-<X>km/burnout-vel-<X>km-s/
        intercept-window-<X>s/detection-time-<X>s/decision-time-<X>s/
        max-accel-<X>g/interceptors-per-sat-<X>/doctrine-<X>x/salvo-size-<X>
    """
    # --- target mode and name ---
    mode = str(cfg.get("target_mode", "country")).strip().lower()

    if mode == "prespecified":
        pts = cfg.get("target_points") or []
        if pts:
            first_name = str(pts[0][0]).strip().lower().replace(" ", "-").replace(":", "").replace("/", "-")
            target_name = first_name[:40]
        else:
            target_name = "prespecified"
    else:
        mode = "country"
        country_raw = cfg["target_country"]
        if isinstance(country_raw, str):
            countries = [country_raw.strip()]
        else:
            countries = [str(c).strip() for c in country_raw]
        target_name = "+".join(
            sorted(c.lower().replace(" ", "-") for c in countries if c)
        )

    # --- numeric parameters ---
    intercept_alt = round(float(cfg["intercept_alt_km"]))
    orbit_alt     = round(float(actual_alt_km))
    vbo           = cfg["v_bo_km_s"]
    # "t_window" name and the "intercept-window" label below are kept (not
    # renamed to burnout_time/burnout-time) so existing path-parsing regexes
    # (cloud_plot_best_results.py, plot_best_results_by_country.py) keep
    # matching -- only the source of the number changed, from t_window_s to
    # target_missile_burnout_time_s (numerically identical at defaults).
    t_window      = round(float(cfg["target_missile_burnout_time_s"]))
    detect_time   = round(float(cfg["detection_time_s"]))
    decide_time   = round(float(cfg["decision_time_s"]))
    a_g           = float(cfg["a_g"])
    ips           = int(cfg["n_interceptors_per_sat"])
    doctrine      = 1   # multiplier on r_required; always 1 for now
    salvo         = int(cfg["salvo_size"])

    return "/".join([
        mode,
        target_name,
        f"target-points-{n_targets}",
        f"intercept-alt-{intercept_alt}km",
        f"orbit-alt-{orbit_alt}km",
        f"burnout-vel-{vbo:.1f}km-s",
        f"intercept-window-{t_window}s",
        f"detection-time-{detect_time}s",
        f"decision-time-{decide_time}s",
        f"max-accel-{a_g:.1f}g",
        f"interceptors-per-sat-{ips}",
        f"doctrine-{doctrine}x",
        f"salvo-size-{salvo}",
    ])


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
    cfg["target_shell_n_points"]        = int(cfg["target_shell_n_points"])
    cfg["target_country"]               = (
        str(cfg["target_country"]).strip() if cfg.get("target_country") is not None else None
    )
    cfg["target_altitude_km"]           = float(cfg["target_altitude_km"])
    cfg["inclination_sweep_bounds_deg"] = [
        float(x) for x in cfg["inclination_sweep_bounds_deg"]
    ]
    cfg["altitude_max_repeat_days"]     = int(cfg["altitude_max_repeat_days"])
    cfg["margin_multiplier"]            = float(cfg["margin_multiplier"])
    cfg["trim_margin_multiplier"]       = float(cfg["trim_margin_multiplier"])
    cfg["max_seeds_safety_cap"]         = int(cfg["max_seeds_safety_cap"])
    cfg["polar_band_deg"]               = [float(x) for x in cfg["polar_band_deg"]]
    cfg["polar_band_divisor"]           = float(cfg["polar_band_divisor"])
    cfg["dt_s"]                         = float(cfg["dt_s"])
    cfg["detection_time_s"]             = float(cfg["detection_time_s"])
    cfg["decision_time_s"]              = float(cfg["decision_time_s"])
    cfg["target_missile_burnout_time_s"] = float(cfg["target_missile_burnout_time_s"])
    cfg["a_g"]                          = float(cfg["a_g"])
    cfg["intercept_alt_km"]             = float(cfg["intercept_alt_km"])
    cfg["min_elev_deg"]                 = float(cfg["min_elev_deg"])
    cfg["inc_screen_step_deg"]          = float(cfg["inc_screen_step_deg"])
    cfg["n_raan_offsets"]               = int(cfg["n_raan_offsets"])
    cfg["seed_workers"]                 = int(cfg["seed_workers"])
    cfg["mip_focus"]                    = int(cfg["mip_focus"])
    cfg["heuristics_frac"]              = float(cfg["heuristics_frac"])
    cfg["solver_params"]                = (
        dict(cfg["solver_params"]) if cfg.get("solver_params") is not None else None
    )
    cfg["run_name"]                     = str(cfg["run_name"]) if cfg.get("run_name") is not None else None
    cfg["is_principal_set"]             = bool(cfg.get("is_principal_set", False))
    cfg["stop_at_gap"]                  = (
        float(cfg["stop_at_gap"]) if cfg.get("stop_at_gap") is not None else None
    )

    return cfg


# ---------------------------------------------------------------------------
# Seed selection: rarity-weighted, margin-driven build + trim (mirrors
# lee_test2.py -- see that file for the full design rationale).
# ---------------------------------------------------------------------------

def _rarity_weights(access: np.ndarray) -> np.ndarray:
    """Static, pool-level scarcity weight per target, computed once from the
    full candidate pool before any selection happens.

    Rank-based rather than a raw 1/total_access inverse: a target reachable
    by only one or two candidates in the whole pool would otherwise get a
    weight that dwarfs everything else and causes the greedy to fixate on it
    exclusively. Scarcest target -> 1.0, most common -> a 0.1 floor (kept
    nonzero so ties among candidates that only help common targets don't
    degenerate to all-zero scores).
    """
    total_access = access.sum(axis=0)
    n_targets = len(total_access)
    if n_targets <= 1:
        return np.ones(n_targets)
    order = np.argsort(total_access)          # ascending: scarcest target first
    rank = np.empty(n_targets, dtype=np.float64)
    rank[order] = np.arange(n_targets, dtype=np.float64)
    return 0.1 + 0.9 * (1.0 - rank / (n_targets - 1))


def _greedy_select_seeds(
    v0_list: list[np.ndarray],
    r_required: int,
    max_seeds: int,
    *,
    margin_multiplier: float,
    polar_band_deg: tuple[float, float],
    polar_band_divisor: float,
    inc_deg_list: list[float] | None = None,
) -> tuple[list[int], set[int], np.ndarray, np.ndarray]:
    """Rarity-weighted, margin-driven greedy seed selection.

    Phase 1 -- Polar-band reservation (capped, not unconditional): reserves
    up to ceil(r_required / polar_band_divisor) seeds, eligible inclinations
    restricted to polar_band_deg, chosen by the rarity-weighted, margin-capped
    score. Not "one per every live inclination" -- on a scenario with many
    live inclinations, guaranteeing all of them can by itself blow past the
    margin target before Phase 2 ever runs.

    Phase 2 -- Greedy fill: keeps adding seeds (rarity-weighted, margin-capped
    score, unbounded in inclination) until EITHER the worst-covered target's
    accumulated access reaches margin_multiplier * r_required, OR max_seeds
    is hit (a safety ceiling, not a target), OR no live candidates remain.

    Returns (selected indices in selection order, the set of indices picked
    during Phase 1 for reporting/tagging purposes only -- NOT protected from
    trimming, see _trim_seeds -- the final per-target coverage array, and the
    access matrix).
    """
    n_seeds       = len(v0_list)
    access        = np.stack([v0.sum(axis=0) for v0 in v0_list]).astype(np.float64)
    rarity        = _rarity_weights(access)
    n_targets     = access.shape[1]
    margin_target = margin_multiplier * r_required
    coverage      = np.zeros(n_targets, dtype=np.float64)
    selected: list[int] = []
    polar_reserved: set[int] = set()
    remaining = list(range(n_seeds))

    def _scores(cands_idx: list[int]) -> np.ndarray:
        deficit = np.maximum(0.0, margin_target - coverage)
        return (np.minimum(access[cands_idx], deficit) * rarity).sum(axis=1)

    # ------------------------------------------------------------------
    # Phase 1: Polar-band reservation
    # ------------------------------------------------------------------
    if inc_deg_list is not None:
        polar_lo, polar_hi = polar_band_deg
        polar_budget = int(np.ceil(r_required / polar_band_divisor))
        pending_incs = sorted(
            inc for inc in set(inc_deg_list)
            if polar_lo <= inc <= polar_hi
        )

        while (
            pending_incs
            and len(selected) < max_seeds
            and len(polar_reserved) < polar_budget
        ):
            best_score   = -np.inf
            best_global  = -1
            best_inc_idx = -1

            for ii, inc in enumerate(pending_incs):
                cands = [i for i in remaining if abs(inc_deg_list[i] - inc) < 0.1]
                if not cands:
                    continue
                scores     = _scores(cands)
                local_best = int(np.argmax(scores))
                score      = float(scores[local_best])
                if score > best_score:
                    best_score   = score
                    best_global  = cands[local_best]
                    best_inc_idx = ii

            if best_global == -1:
                break

            selected.append(best_global)
            polar_reserved.add(best_global)
            coverage = coverage + access[best_global]
            remaining.remove(best_global)
            pending_incs.pop(best_inc_idx)

    # ------------------------------------------------------------------
    # Phase 2: Greedy fill — stop once every target clears the margin target
    # ------------------------------------------------------------------
    while remaining and len(selected) < max_seeds:
        if float(coverage.min()) >= margin_target:
            break
        scores      = _scores(remaining)
        best_local  = int(np.argmax(scores))
        best_global = remaining[best_local]
        selected.append(best_global)
        coverage = coverage + access[best_global]
        remaining.pop(best_local)

    return selected, polar_reserved, coverage, access


def _trim_seeds(
    selected: list[int],
    access: np.ndarray,
    coverage: np.ndarray,
    r_required: int,
    trim_margin_multiplier: float,
) -> list[int]:
    """Reverse marginal-contribution trim -- single pass, no seed is exempt.

    Repeatedly removes whichever currently-selected seed does the LEAST
    damage to the worst-covered target, stopping once no further removal
    would leave every target at or above trim_margin_multiplier * r_required.
    No seed is exempt regardless of which phase selected it -- Phase 1
    reserves only a small, requirement-scaled number of seeds, so the risk of
    trim removing a large structural investment is bounded and acceptable.
    """
    floor    = trim_margin_multiplier * r_required
    current  = list(selected)
    coverage = coverage.copy()

    while True:
        remaining_margins = {
            i: float((coverage - access[i]).min()) for i in current
        }
        if not remaining_margins:
            break
        best_i = max(remaining_margins, key=remaining_margins.get)
        if remaining_margins[best_i] < floor:
            break
        current.remove(best_i)
        coverage = coverage - access[best_i]

    return current


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
    # prefix is resolved after actual_alt_km is known (see below)

    target_mode     = cfg["target_mode"]
    country         = cfg.get("target_country")
    target_points   = cfg.get("target_points")   # list of [name, lat, lon] or None
    inc_lo, inc_hi  = cfg["inclination_sweep_bounds_deg"]

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
    print(f"  Config source  : {s3_config_uri}",                                 flush=True)
    print(f"  Target         : {_target_desc}",                                  flush=True)
    print(f"  Inc bounds     : [{inc_lo:.1f}°, {inc_hi:.1f}°]",                 flush=True)
    print(f"  Altitude       : {cfg['target_altitude_km']} km",                   flush=True)
    print(f"  v_bo           : {cfg['v_bo_km_s']} km/s",                        flush=True)
    print(f"  Salvo size     : {cfg['salvo_size']}",                             flush=True)
    print(f"  Interceptors   : {cfg['n_interceptors_per_sat']} per sat",         flush=True)
    print(f"  MIP gap        : {cfg['mip_gap'] * 100:.2f}%",                     flush=True)
    print(f"  Time limit     : {cfg['time_limit_s']:.0f} s"
          f"  ({cfg['time_limit_s'] / 3600:.2f} h)",                             flush=True)
    print(f"  S3 bucket      : {bucket}  (prefix resolved after shell masking)",  flush=True)
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
    # RGT orbit selection
    # ------------------------------------------------------------------
    candidates = candidate_rgt_ratios(
        h_star_km=cfg["target_altitude_km"],
        earth=earth,
        max_repeat_days=cfg["altitude_max_repeat_days"],
        dt_s=cfg["dt_s"],
    )
    if not candidates:
        print(
            f"[batch_worker] ERROR: No RGT orbit found near {cfg['target_altitude_km']:.0f} km "
            f"with N_D ≤ {cfg['altitude_max_repeat_days']}.",
            flush=True,
        )
        sys.exit(1)

    best          = candidates[0]
    N_P           = int(best["N_P"])
    N_D           = int(best["N_D"])
    L             = int(best["L"])
    actual_alt_km = float(best["alt_km"])

    _dt_s_actual = float(best["T_r_s"]) / L
    horizon_s    = (L - 1) * _dt_s_actual

    print(
        f"\nTarget altitude requested: {cfg['target_altitude_km']:.1f} km  →  "
        f"nearest RGT orbit: {N_P}:{N_D}  "
        f"actual_alt={actual_alt_km:.2f} km  "
        f"(snap error: {float(best['alt_error_km']):.2f} km)",
        flush=True,
    )

    sim = SimConfig(
        horizon_s=horizon_s,
        dt_s=_dt_s_actual,
        analysis_matrix_dtype="uint8",
        use_j2=False,
        use_drag=False,
    )
    print(
        f"Shared repeat family: N_D={N_D}  L={L} steps",
        flush=True,
    )
    print(
        f"Simulation horizon: {horizon_s:.1f} s  ({horizon_s / 3600:.3f} h)  -> L={L}",
        flush=True,
    )

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

    _auto_prefix = _build_s3_prefix(cfg, actual_alt_km, n_targets)
    prefix = f"{s3_base}/{_auto_prefix}" if s3_base else _auto_prefix

    print(f"\nTarget mode: {target_mode}  n_targets={n_targets}", flush=True)
    print(f"[batch_worker] S3 prefix: {prefix}", flush=True)

    # ------------------------------------------------------------------
    # Inclination × RAAN seed grid
    # ------------------------------------------------------------------
    inc_grid = np.arange(
        float(inc_lo),
        float(inc_hi) + float(cfg["inc_screen_step_deg"]) * 0.5,
        float(cfg["inc_screen_step_deg"]),
    )
    raan_interval = 360.0 / N_P
    raan_grid     = np.linspace(0.0, raan_interval, cfg["n_raan_offsets"], endpoint=False)
    seed_pairs    = [(float(inc), float(raan)) for inc in inc_grid for raan in raan_grid]
    n_seeds_total = len(seed_pairs)

    print(
        f"\nSeed grid: {len(inc_grid)} inclinations × {cfg['n_raan_offsets']} RAANs"
        f" = {n_seeds_total} seeds",
        flush=True,
    )
    print(
        f"  inclinations : {float(inc_lo):.1f}°–{float(inc_hi):.1f}°"
        f"  step {cfg['inc_screen_step_deg']:.1f}°",
        flush=True,
    )
    print(
        f"  RAANs        : {cfg['n_raan_offsets']} offsets in [0°, {raan_interval:.3f}°)"
        f"  (360° / N_P={N_P})  step={raan_interval / cfg['n_raan_offsets']:.3f}°",
        flush=True,
    )

    # ------------------------------------------------------------------
    # Build sub-layers
    # ------------------------------------------------------------------
    sub_layers = [
        rgt_common_ground_track_layer(
            inc_deg=inc,
            N_P=N_P,
            N_D=N_D,
            L=L,
            earth=earth,
            raan0_deg=raan,
            use_j2=False,
        )
        for inc, raan in seed_pairs
    ]

    # ------------------------------------------------------------------
    # Simulate all seeds in parallel
    # ------------------------------------------------------------------
    def _simulate_seed(idx: int) -> tuple[int, np.ndarray]:
        s_layer  = seed_layer(sub_layers[idx])
        analysis = run_simulation(
            elems0=s_layer.elems,
            phys=s_layer.phys,
            sim=sim,
            cov=cov,
            earth=earth,
            shell=shell,
            layer_specs=s_layer.spec,
            propagator=_PROPAGATOR,
        )
        return idx, (analysis.counts > 0).astype(np.float64)

    n_seeds = n_seeds_total
    print(f"\nSimulating {n_seeds} seeds  (workers={cfg['seed_workers']}) ...", flush=True)
    t_sim0 = time.perf_counter()
    v0_list: list[np.ndarray] = [None] * n_seeds  # type: ignore[list-item]
    progress_every = max(1, n_seeds // 20)

    with ThreadPoolExecutor(max_workers=cfg["seed_workers"]) as pool:
        futs = {pool.submit(_simulate_seed, i): i for i in range(n_seeds)}
        done_count = 0
        for fut in as_completed(futs):
            idx, v0 = fut.result()
            v0_list[idx] = v0
            done_count += 1
            if done_count % progress_every == 0 or done_count == n_seeds:
                print(f"  {done_count}/{n_seeds} seeds simulated ...", flush=True)

    print(f"Total seed simulation time: {time.perf_counter() - t_sim0:.2f} s", flush=True)

    # ------------------------------------------------------------------
    # Seed selection: dead-filter → greedy rank → top N to MILP
    # ------------------------------------------------------------------
    r_required = int(np.ceil(cfg["salvo_size"] / cfg["n_interceptors_per_sat"]))

    live_mask = [bool(v0.any()) for v0 in v0_list]
    n_dead    = live_mask.count(False)
    if n_dead:
        print(f"\nPre-filter: removed {n_dead}/{n_seeds} seeds with zero target access.", flush=True)
        v0_list    = [v for v, keep in zip(v0_list,    live_mask) if keep]
        sub_layers = [s for s, keep in zip(sub_layers, live_mask) if keep]
        seed_pairs = [p for p, keep in zip(seed_pairs, live_mask) if keep]
        n_seeds    = len(v0_list)
    n_seeds_live = n_seeds

    inc_deg_list  = [float(p[0]) for p in seed_pairs]
    margin_target = cfg["margin_multiplier"] * r_required
    build_idx, polar_reserved_set, build_coverage, access_full = _greedy_select_seeds(
        v0_list, r_required, cfg["max_seeds_safety_cap"],
        margin_multiplier=cfg["margin_multiplier"],
        polar_band_deg=tuple(cfg["polar_band_deg"]),
        polar_band_divisor=cfg["polar_band_divisor"],
        inc_deg_list=inc_deg_list,
    )
    build_worst_margin = float(build_coverage.min())
    n_built = len(build_idx)
    print(
        f"\nBuild phase: {n_built} of {n_seeds} live seeds selected  "
        f"(target: worst-slot margin >= {cfg['margin_multiplier']:.2f}x requirement = {margin_target:.1f})",
        flush=True,
    )
    print(
        f"  Worst-slot margin achieved: {build_worst_margin:.1f}  "
        f"({build_worst_margin / r_required:.2f}x requirement)"
        + ("  [safety cap max_seeds_safety_cap reached before target margin]"
           if n_built >= cfg["max_seeds_safety_cap"] and build_worst_margin < margin_target - 1e-9
           else ""),
        flush=True,
    )

    trim_floor = cfg["trim_margin_multiplier"] * r_required
    greedy_idx = _trim_seeds(
        build_idx, access_full, build_coverage, r_required,
        cfg["trim_margin_multiplier"],
    )
    n_selected    = len(greedy_idx)
    trim_coverage = access_full[greedy_idx].sum(axis=0)
    trim_worst_margin = float(trim_coverage.min())
    n_trimmed = n_built - n_selected
    print(
        f"Trim phase : removed {n_trimmed} seed(s), {n_built} → {n_selected}  "
        f"(floor: worst-slot margin >= {cfg['trim_margin_multiplier']:.2f}x requirement = {trim_floor:.1f})",
        flush=True,
    )
    print(
        f"  Worst-slot margin after trim: {trim_worst_margin:.1f}  "
        f"({trim_worst_margin / r_required:.2f}x requirement)",
        flush=True,
    )

    feasible_proxy = trim_worst_margin >= r_required
    print(
        f"  Coverage proxy (all slots of selected seeds): "
        f"min={trim_worst_margin:.1f}  mean={float(trim_coverage.mean()):.1f}  "
        f"requirement={r_required}  "
        + ("✓ feasible" if feasible_proxy
           else "⚠ infeasible even after build+trim — try raising margin_multiplier or max_seeds_safety_cap"),
        flush=True,
    )

    # Tag seeds by selection phase: polar-band reservation (Phase 1) vs.
    # quantity-driven greedy fill (Phase 2). Not a protection from trimming.
    for rank, idx in enumerate(greedy_idx):
        inc, raan = seed_pairs[idx]
        tag = "  [phase 1: polar-band reservation]" if idx in polar_reserved_set else ""
        print(
            f"  {rank + 1:2d}: i={inc:5.1f}°  RAAN={raan:6.1f}°  "
            f"access/target={v0_list[idx].sum(axis=0).mean():.1f}{tag}",
            flush=True,
        )

    # Reindex to MILP subset (seed_pairs NOT reindexed — greedy_idx still valid)
    v0_list_milp    = [v0_list[i]    for i in greedy_idx]
    sub_layers_milp = [sub_layers[i] for i in greedy_idx]

    # ------------------------------------------------------------------
    # Build constraint matrix
    # ------------------------------------------------------------------
    param_V = build_param_V(v0_list_milp)
    param_V_summary(param_V, n_seeds=n_selected)

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
    log_local = Path(f"/tmp/{job}_gurobi.log")

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

    orb_elems = build_orb_elems_table(sub_layers_milp, L)
    print_orb_table(orb_elems, result.selected_slots)

    # ------------------------------------------------------------------
    # Build run_config metadata block (mirrors lee_test2.py)
    # ------------------------------------------------------------------
    run_config = {
        # --- Run metadata ---
        "run_name":                       cfg["run_name"],
        "is_principal_set":               cfg["is_principal_set"],
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
        # --- Orbit ---
        "target_altitude_km":             cfg["target_altitude_km"],
        "altitude_max_repeat_days":       cfg["altitude_max_repeat_days"],
        "actual_alt_km":                  float(actual_alt_km),
        "rgt_ratio":                      f"{N_P}:{N_D}",
        "L":                              int(L),
        "horizon_s":                      float(horizon_s),
        # --- Seed grid ---
        "inclination_sweep_bounds_deg":   [float(inc_lo), float(inc_hi)],
        "inclination_screen_step_deg":    cfg["inc_screen_step_deg"],
        "n_raan_offsets":                 cfg["n_raan_offsets"],
        "raan_interval_deg":              float(raan_interval),
        "raan_step_deg":                  float(raan_interval / cfg["n_raan_offsets"]),
        "n_seeds_simulated":              int(n_seeds_total),
        "n_seeds_live":                   int(n_seeds_live),
        "margin_multiplier":              cfg["margin_multiplier"],
        "trim_margin_multiplier":         cfg["trim_margin_multiplier"],
        "max_seeds_safety_cap":           cfg["max_seeds_safety_cap"],
        "polar_band_deg":                 list(cfg["polar_band_deg"]),
        "polar_band_divisor":             cfg["polar_band_divisor"],
        "n_seeds_built":                  int(n_built),
        "n_seeds_trimmed":                int(n_trimmed),
        "n_seeds_after_trim":             int(n_selected),
        "build_worst_margin":             build_worst_margin,
        "trim_worst_margin":              trim_worst_margin,
        "bilp_n_vars":                    int(n_selected * L),
        "seeds_selected": [
            {
                "rank":     rank + 1,
                "inc_deg":  float(seed_pairs[i][0]),
                "raan_deg": float(seed_pairs[i][1]),
            }
            for rank, i in enumerate(greedy_idx)
        ],
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
    csv_local  = Path(f"/tmp/{job}.csv")
    json_local = csv_local.with_suffix(".json")

    targets_payload = [
        {"name": labels[i], "lat_deg": float(lats_deg[i]), "lon_deg": float(lons_deg[i])}
        for i in range(n_targets)
    ]
    bc_kg_m2 = float(np.asarray(sub_layers_milp[0].phys.bc_kg_m2).ravel()[0])

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

    _s3_upload(csv_local,  bucket, f"{prefix}/{job}.csv")
    _s3_upload(json_local, bucket, f"{prefix}/{job}.json")

    if log_local.exists():
        _s3_upload(log_local, bucket, f"{prefix}/{job}_gurobi.log")

    # Echo the input config alongside outputs for full traceability
    config_echo = json.dumps(cfg, indent=2, default=str).encode("utf-8")
    _s3_upload_bytes(config_echo, bucket, f"{prefix}/{job}_config.json")

    print(f"\n[batch_worker] All outputs at s3://{bucket}/{prefix}/", flush=True)
    print("[batch_worker] Done.",                                       flush=True)


if __name__ == "__main__":
    main()
