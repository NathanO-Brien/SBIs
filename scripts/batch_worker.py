"""
batch_worker.py

Cloud-native SCLP optimization script for AWS Batch.

Mirrors lee_test2.py exactly — country-mode target shell, full inclination ×
RAAN seed grid, greedy seed selection, polar orbit reservation — but reads all
problem inputs from a JSON config file stored in S3 and writes all outputs back
to S3.  Designed to run inside a Docker container on AWS Batch.
Submitted via cloud_run.py (single job) or cloud_sweep.py (parameter sweep).

Environment variables (the only three needed):
    S3_CONFIG   Full S3 URI of the JSON config file to load.
                Format: s3://<bucket>/<key>
    S3_BUCKET   S3 bucket where all outputs are written.
    S3_PREFIX   S3 key prefix for this job's outputs, e.g. "results/run-001"
                (no trailing slash).

Config file schema (JSON):
    {
        "job_name":                     "china_450km",   // optional; auto-timestamped if omitted
        "target_country":               "China",         // required
        "target_shell_n_points":        10000,           // optional, default 10000
        "altitude_override_km":         450.0,           // optional; null uses geometric h*
        "inclination_sweep_bounds_deg": [20.0, 90.0],   // optional, default [20, 90]
        "use_j2":                       false,           // optional, default false (Keplerian)
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
    footprint_half_angles_deg,
    optimal_sat_altitude_km,
    param_V_summary,
    print_orb_table,
    rgt_common_ground_track_layer,
    save_sclp_result,
    seed_layer,
    solve_sclp,
)


# ---------------------------------------------------------------------------
# Fixed algorithm parameters — mirror lee_test2.py exactly.
# These are not exposed in the config file.
# ---------------------------------------------------------------------------
_DT_S: float                    = 120.0
_T_WINDOW_S: float              = 170.0
_A_G: float                     = 10.0
_INTERCEPT_ALT_KM: float        = 200.0
_MIN_ELEV_DEG: float            = 0.0
_INC_SCREEN_STEP_DEG: float     = 1.0
_ALTITUDE_MAX_REPEAT_DAYS: int  = 1
_N_RAAN_OFFSETS: int            = 10
_SEED_WORKERS: int              = 8
# Keyed on r_required = ceil(salvo_size / n_interceptors_per_sat), not on
# salvo_size directly — the greedy pool must be large enough for the
# solver to find r_required satellites simultaneously in view of every
# target, and that's the quantity that actually drives feasibility/
# difficulty, not salvo_size alone. See the "Two-Stage Hybrid" discussion
# in docs/paper/sbi_optimization_paper.tex for the reasoning.
_SEEDS_FOR_MILP_BY_R_REQUIRED: dict[int, int] = {
    1:   25,
    5:   25,
    20:  50,
    50: 100,
    100: 150,
}
_RESERVE_POLAR_ORBITS: bool     = False
_USE_J2: bool                   = False
_MIP_FOCUS: int                 = 1
_HEURISTICS_FRAC: float         = 0.2
_SOLVER_PARAMS: dict = {
    # Method 1 = dual simplex. Previously Method 2 (Barrier) — barrier took 300s+ on root LP
    # and crossover added another 70s. For pure 0/1 set-cover with an all-ones matrix,
    # dual simplex is faster because the matrix is sparse and simplex exploits that.
    "Method":                1,
    # Write B&B nodes to disk at 4 GB — prevents OOM during branching.
    "NodefileStart":         4,
    # RINS heuristic every 50 nodes — this is what actually finds good solutions.
    # B&B alone cannot explore enough nodes on problems of this size.
    "RINS":                  50,
    # Presolve removed 0 rows and 0 columns in testing (pure 0/1 set-cover gives it
    # nothing to exploit) while consuming 268s. Disabled.
    # Previously: "Presolve": 2
    # No-improvement termination: stop if no incumbent improvement for 1 hour,
    # but don't start that clock until 3 hours in — gives RINS time to find its
    # first solution on hard instances before the termination check activates.
    # Previously used a fixed TimeLimit only, which wasted hours on a stalled solver.
}
_PROPAGATOR: str                = "Nominal_Propagator"

_POLAR_RESERVATION_INCLINATIONS: tuple[float, ...] = (80.0, 85.0, 90.0)


# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

_CONFIG_DEFAULTS: dict = {
    "job_name":                     None,    # auto-generated if omitted
    # Target — exactly one of target_country or target_points must be set
    "target_mode":                  "country",   # "country" or "prespecified"
    "target_country":               None,
    "target_points":                None,        # list of [name, lat, lon] for prespecified mode
    "target_shell_n_points":        15000,
    "altitude_override_km":         None,    # null → use geometric h*
    "inclination_sweep_bounds_deg": [20.0, 90.0],
    "use_j2":                       False,
    "v_bo_km_s":                    None,    # required
    "salvo_size":                   None,    # required
    "n_interceptors_per_sat":       None,    # required
    "mip_gap":                      0.01,
    "time_limit_s":                 None,    # required
    "altitude_max_repeat_days":     1,
    # Seed count — set n_seeds_for_milp (flat int) OR seeds_for_milp_by_r_required
    # (dict, keyed on r_required = ceil(salvo_size / n_interceptors_per_sat), not
    # on salvo_size). If both are present, n_seeds_for_milp takes precedence.
    "n_seeds_for_milp":             None,    # flat int; if None, falls back to dict below
    "seeds_for_milp_by_r_required": {1: 25, 5: 25, 20: 50, 50: 100, 100: 150},
    # Simulation parameters
    "dt_s":                         120.0,
    "t_window_s":                   170.0,
    "a_g":                          10.0,
    "intercept_alt_km":             200.0,
    "min_elev_deg":                 0.0,
    "inc_screen_step_deg":          1.0,
    "n_raan_offsets":               10,
    "seed_workers":                 8,
    "reserve_polar_orbits":         False,
    # Solver tuning
    "mip_focus":                    1,
    "heuristics_frac":              0.2,
    "solver_params":                {
        "Method":                1,
        "NodefileStart":         4,
        "RINS":                  50,
        "ImproveStartTime":      10800,
        "NoImproveLimit":        3600,
    },
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

def _build_s3_prefix(cfg: dict, actual_alt_km: float, n_targets: int) -> str:
    """Build a hierarchical S3 prefix from run parameters.

    Structure:
        <target-mode>/<target-name>/target-points-<N>/
        intercept-alt-<X>km/orbit-alt-<X>km/burnout-vel-<X>km-s/
        intercept-window-<X>s/max-accel-<X>g/
        interceptors-per-sat-<X>/doctrine-<X>x/salvo-size-<X>
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
    t_window      = round(float(cfg["t_window_s"]))
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
    cfg["target_country"]               = str(cfg["target_country"]).strip()
    cfg["altitude_override_km"]         = (
        float(cfg["altitude_override_km"]) if cfg["altitude_override_km"] is not None else None
    )
    cfg["inclination_sweep_bounds_deg"] = [
        float(x) for x in cfg["inclination_sweep_bounds_deg"]
    ]
    cfg["altitude_max_repeat_days"]     = int(cfg["altitude_max_repeat_days"])
    cfg["seeds_for_milp_by_r_required"] = {int(k): int(v) for k, v in cfg["seeds_for_milp_by_r_required"].items()}
    cfg["n_seeds_for_milp"]             = (
        int(cfg["n_seeds_for_milp"]) if cfg.get("n_seeds_for_milp") is not None else None
    )
    cfg["dt_s"]                         = float(cfg["dt_s"])
    cfg["t_window_s"]                   = float(cfg["t_window_s"])
    cfg["a_g"]                          = float(cfg["a_g"])
    cfg["intercept_alt_km"]             = float(cfg["intercept_alt_km"])
    cfg["min_elev_deg"]                 = float(cfg["min_elev_deg"])
    cfg["inc_screen_step_deg"]          = float(cfg["inc_screen_step_deg"])
    cfg["n_raan_offsets"]               = int(cfg["n_raan_offsets"])
    cfg["seed_workers"]                 = int(cfg["seed_workers"])
    cfg["reserve_polar_orbits"]         = bool(cfg["reserve_polar_orbits"])
    cfg["mip_focus"]                    = int(cfg["mip_focus"])
    cfg["heuristics_frac"]              = float(cfg["heuristics_frac"])
    cfg["solver_params"]                = dict(cfg["solver_params"])

    return cfg


# ---------------------------------------------------------------------------
# Greedy seed selection — verbatim from lee_test2.py
# ---------------------------------------------------------------------------

def _greedy_select_seeds(
    v0_list: list[np.ndarray],
    r_required: int,
    n_max: int,
    *,
    inc_deg_list: list[float] | None = None,
    reserve_polar_orbits: bool = False,
) -> list[int]:
    """Greedy marginal-coverage seed selection (mirrors lee_test2.py exactly).

    Three-phase selection:

    Phase 0 — Polar reservation (if reserve_polar_orbits is True):
        Pre-select the best-RAAN seed for each inclination in
        _POLAR_RESERVATION_INCLINATIONS before any other selection runs.

    Phase 1 — Inclination guarantee:
        Every discrete inclination value present in inc_deg_list that was not
        already covered by Phase 0 is guaranteed exactly one representative.
        Among all RAAN candidates for a given inclination the one with the
        highest marginal-coverage score is chosen.  Inclinations are processed
        in greedy order — whichever missing inclination's best candidate
        contributes the most marginal coverage is selected first.

    Phase 2 — Greedy fill:
        Remaining n_max slots are filled by standard greedy marginal-coverage,
        allowing inclinations to repeat at different RAANs.
    """
    n_seeds   = len(v0_list)
    access    = np.stack([v0.sum(axis=0) for v0 in v0_list]).astype(np.float64)
    deficit   = np.full(access.shape[1], float(r_required))
    selected: list[int] = []
    remaining = list(range(n_seeds))

    # ------------------------------------------------------------------
    # Phase 0: Polar reservation
    # ------------------------------------------------------------------
    if reserve_polar_orbits and inc_deg_list is not None and n_max >= 1:
        for target_inc in _POLAR_RESERVATION_INCLINATIONS:
            if len(selected) >= n_max:
                break
            candidates = [
                i for i in remaining
                if abs(inc_deg_list[i] - target_inc) < 0.1
            ]
            if not candidates:
                continue
            best = max(candidates, key=lambda i: access[i].sum())
            selected.append(best)
            deficit = np.maximum(0.0, deficit - access[best])
            remaining.remove(best)

    # ------------------------------------------------------------------
    # Phase 1: Inclination guarantee — every unique inclination gets one rep
    # ------------------------------------------------------------------
    if inc_deg_list is not None:
        covered_incs = {inc_deg_list[i] for i in selected}
        pending_incs = [
            inc for inc in sorted(set(inc_deg_list))
            if inc not in covered_incs
        ]

        while pending_incs and len(selected) < n_max:
            best_score   = -np.inf
            best_global  = -1
            best_inc_idx = -1

            for ii, inc in enumerate(pending_incs):
                cands = [i for i in remaining if abs(inc_deg_list[i] - inc) < 0.1]
                if not cands:
                    continue
                cands_access = access[cands]
                if deficit.max() > 0.0:
                    scores = np.minimum(cands_access, deficit).sum(axis=1)
                else:
                    scores = cands_access.sum(axis=1)
                local_best = int(np.argmax(scores))
                score      = float(scores[local_best])
                if score > best_score:
                    best_score   = score
                    best_global  = cands[local_best]
                    best_inc_idx = ii

            if best_global == -1:
                break

            selected.append(best_global)
            deficit = np.maximum(0.0, deficit - access[best_global])
            remaining.remove(best_global)
            pending_incs.pop(best_inc_idx)

    # ------------------------------------------------------------------
    # Phase 2: Greedy fill — inclinations may repeat at different RAANs
    # ------------------------------------------------------------------
    for _ in range(min(n_max - len(selected), len(remaining))):
        if deficit.max() > 0.0:
            scores = np.minimum(access[remaining], deficit).sum(axis=1)
        else:
            scores = access[remaining].sum(axis=1)
        best_local  = int(np.argmax(scores))
        best_global = remaining[best_local]
        selected.append(best_global)
        deficit = np.maximum(0.0, deficit - access[best_global])
        remaining.pop(best_local)

    return selected


def _seeds_for_r_required(r_required: int, lookup: dict[int, int]) -> int:
    """Return MILP seed count for r_required from the provided lookup dict.

    r_required = ceil(salvo_size / n_interceptors_per_sat) is the number of
    satellites that must be simultaneously in view of a target — the
    quantity that actually governs how large the greedy-selected seed pool
    needs to be, not salvo_size on its own. Finds the largest key ≤
    r_required; falls back to the smallest key if r_required is below all
    defined keys.
    """
    keys   = sorted(lookup)
    result = lookup[keys[0]]
    for k in keys:
        if k <= r_required:
            result = lookup[k]
        else:
            break
    return result


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
    print(f"  Altitude       : {cfg['altitude_override_km']} km override"
          f"  (None → geometric h*)",                                            flush=True)
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
        T_window_s=cfg["t_window_s"],
        v_bo_km_s=cfg["v_bo_km_s"],
        a_g=cfg["a_g"],
        intercept_alt_km=cfg["intercept_alt_km"],
        min_elev_deg=cfg["min_elev_deg"],
    )
    print(f"\nComputed R_max: {cov.max_range_km:.2f} km", flush=True)

    h_star = optimal_sat_altitude_km(
        max_range_km=cov.max_range_km,
        min_elev_deg=cov.min_elev_deg,
        intercept_alt_km=cov.intercept_alt_km,
        earth=earth,
    )
    angles = footprint_half_angles_deg(
        h_sat_km=h_star,
        max_range_km=cov.max_range_km,
        min_elev_deg=cov.min_elev_deg,
        intercept_alt_km=cov.intercept_alt_km,
        earth=earth,
    )
    print(f"Optimal satellite altitude (h*): {h_star:.2f} km",                  flush=True)
    print(f"  rho_eff: {angles['rho_eff_deg']:.4f} deg"
          f"  (binding: {angles['binding']})",                                   flush=True)

    # ------------------------------------------------------------------
    # RGT orbit selection
    # ------------------------------------------------------------------
    altitude_ref = cfg["altitude_override_km"] if cfg["altitude_override_km"] is not None else h_star

    candidates = candidate_rgt_ratios(
        h_star_km=altitude_ref,
        earth=earth,
        max_repeat_days=cfg["altitude_max_repeat_days"],
        dt_s=cfg["dt_s"],
    )
    if not candidates:
        print(
            f"[batch_worker] ERROR: No RGT orbit found near {altitude_ref:.0f} km "
            f"with N_D ≤ {cfg['altitude_max_repeat_days']}.",
            flush=True,
        )
        sys.exit(1)

    best          = candidates[0]
    N_P           = int(best["N_P"])
    N_D           = int(best["N_D"])
    L             = int(best["L"])
    actual_alt_km = float(best["alt_km"])

    use_j2 = bool(cfg.get("use_j2", False))

    if use_j2:
        from sbi_coverage.Lee.rgt_slots import a_km_j2_rgt as _a_km_j2_rgt
        _inc_ref     = 0.5 * (float(inc_lo) + float(inc_hi))
        _a_j2        = _a_km_j2_rgt(N_P, N_D, _inc_ref, earth)
        _n_j2        = float(np.sqrt(earth.mu_km3_s2 / _a_j2**3))
        _fac_j2      = 1.5 * earth.j2 * (earth.r_eq_km / _a_j2)**2 * _n_j2
        _od_j2       = 0.5 * _fac_j2 * (5.0 * np.cos(np.deg2rad(_inc_ref))**2 - 1.0)
        _T_r_j2      = N_P * 2.0 * np.pi / (_n_j2 + _od_j2)
        L            = round(_T_r_j2 / cfg["dt_s"])
        _dt_s_actual = _T_r_j2 / L
        _a_j2_ref    = _a_j2
    else:
        _dt_s_actual = float(best["T_r_s"]) / L
        _a_j2_ref    = None

    horizon_s     = (L - 1) * _dt_s_actual

    if cfg["altitude_override_km"] is not None:
        print(
            f"\nAltitude OVERRIDE: {cfg['altitude_override_km']:.1f} km  →  "
            f"nearest RGT orbit: {N_P}:{N_D}  "
            f"actual_alt={actual_alt_km:.2f} km  "
            f"(snap error: {float(best['alt_error_km']):.2f} km)",
            flush=True,
        )
    else:
        print(
            f"\nGeometric h*: {h_star:.2f} km  →  "
            f"nearest RGT orbit: {N_P}:{N_D}  actual_alt={actual_alt_km:.2f} km",
            flush=True,
        )

    sim = SimConfig(
        horizon_s=horizon_s,
        dt_s=_dt_s_actual,
        analysis_matrix_dtype="uint8",
        use_j2=use_j2,
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
    if cfg["reserve_polar_orbits"]:
        polar_extra = np.array(list(_POLAR_RESERVATION_INCLINATIONS), dtype=np.float64)
        inc_grid    = np.unique(np.concatenate([inc_grid, polar_extra]))

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
    _a_km_ref  = _a_j2_ref      if sim.use_j2 else None
    _dt_s_slot = _dt_s_actual   if sim.use_j2 else None
    sub_layers = [
        rgt_common_ground_track_layer(
            inc_deg=inc,
            N_P=N_P,
            N_D=N_D,
            L=L,
            earth=earth,
            raan0_deg=raan,
            use_j2=sim.use_j2,
            a_km_override=_a_km_ref,
            dt_s=_dt_s_slot,
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

    if cfg.get("n_seeds_for_milp") is not None:
        n_seeds_for_milp = int(cfg["n_seeds_for_milp"])
    else:
        n_seeds_for_milp = _seeds_for_r_required(r_required, cfg["seeds_for_milp_by_r_required"])
    inc_deg_list = [float(p[0]) for p in seed_pairs]
    greedy_idx   = _greedy_select_seeds(
        v0_list, r_required, n_seeds_for_milp,
        inc_deg_list=inc_deg_list,
        reserve_polar_orbits=cfg["reserve_polar_orbits"],
    )
    n_selected = len(greedy_idx)
    print(f"\nGreedy seed selection: {n_selected} of {n_seeds} live seeds → MILP", flush=True)

    access_sum     = sum(v0_list[i].sum(axis=0) for i in greedy_idx)
    proxy_min      = float(access_sum.min())
    proxy_mean     = float(access_sum.mean())
    feasible_proxy = proxy_min >= r_required
    print(
        f"  Coverage proxy (all slots of selected seeds): "
        f"min={proxy_min:.1f}  mean={proxy_mean:.1f}  "
        f"requirement={r_required}  "
        + ("✓ feasible" if feasible_proxy else "⚠ may be infeasible — consider raising N_SEEDS_FOR_MILP"),
        flush=True,
    )

    # Tag seeds by selection phase: polar-reserved, inc-guaranteed, or greedy fill.
    polar_reserved_set: dict[int, float] = {}
    if cfg["reserve_polar_orbits"] and inc_deg_list:
        for idx in greedy_idx:
            inc_val = inc_deg_list[idx]
            for target_inc in _POLAR_RESERVATION_INCLINATIONS:
                if abs(inc_val - target_inc) < 0.1 and target_inc not in polar_reserved_set.values():
                    polar_reserved_set[idx] = target_inc
                    break
            if len(polar_reserved_set) >= len(_POLAR_RESERVATION_INCLINATIONS):
                break
    inc_guaranteed_set: set[int] = set()
    if inc_deg_list:
        _seen_incs: set[float] = {inc_deg_list[i] for i in polar_reserved_set}
        for idx in greedy_idx:
            if idx in polar_reserved_set:
                continue
            _inc_val = inc_deg_list[idx]
            if _inc_val not in _seen_incs:
                inc_guaranteed_set.add(idx)
                _seen_incs.add(_inc_val)
    for rank, idx in enumerate(greedy_idx):
        inc, raan = seed_pairs[idx]
        if idx in polar_reserved_set:
            tag = f"  [polar reserved {polar_reserved_set[idx]:.0f}°]"
        elif idx in inc_guaranteed_set:
            tag = "  [inc. guaranteed]"
        else:
            tag = ""
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
        # --- Target ---
        "target_mode":                    target_mode,
        "target_country":                 country,
        "target_points":                  [list(p) for p in target_points] if target_mode == "prespecified" else None,
        "target_shell_n_points":          cfg["target_shell_n_points"] if target_mode == "country" else None,
        "n_targets":                      int(n_targets),
        # --- Engagement physics ---
        "dt_s":                           cfg["dt_s"],
        "t_window_s":                     cfg["t_window_s"],
        "v_bo_km_s":                      cfg["v_bo_km_s"],
        "a_g":                            cfg["a_g"],
        "intercept_alt_km":               cfg["intercept_alt_km"],
        "min_elev_deg":                   cfg["min_elev_deg"],
        "r_max_km":                       float(cov.max_range_km),
        # --- Orbit ---
        "altitude_override_km":           cfg["altitude_override_km"],
        "altitude_max_repeat_days":       cfg["altitude_max_repeat_days"],
        "h_star_km":                      float(h_star),
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
        "n_seeds_for_milp":               n_seeds_for_milp,
        "reserve_polar_orbits":           cfg["reserve_polar_orbits"],
        "polar_reservation_inclinations": list(_POLAR_RESERVATION_INCLINATIONS) if cfg["reserve_polar_orbits"] else [],
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
