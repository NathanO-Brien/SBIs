"""Single source of truth for run configuration: key names, defaults, type coercion, and the
mapping from a configuration to the run-identity settings used for naming (``names.py``).

Every launcher (cloud_run, cloud_sweep, lee_test2) builds a plain dict with these keys; the batch
worker and the local pipeline (``Lee.run_pipeline.run_sclp``) call ``normalize_config`` on it, so
defaults and coercions can never drift between entry points. Configs written before the vocabulary
rename (``a_g``, ``v_bo_km_s``, ``dt_s``, ``max_seeds_safety_cap``, ``run_name``) are still accepted.
"""
from __future__ import annotations

from typing import Any, Mapping

# ---- AWS resources shared by every launcher / index tool -------------------
S3_BUCKET = "sbi-optimization-runs"
S3_RUNS_PREFIX = "runs/"
AWS_REGION = "us-east-1"
JOB_QUEUE = "sbi-queue"
JOB_DEFINITION = "sbi-optimizer-job"

#: Keys with no default: a config without them is rejected.
REQUIRED_KEYS = ("salvo_size", "n_interceptors_per_sat", "burnout_velocity_km_s", "time_limit_s")

CONFIG_DEFAULTS: dict[str, Any] = {
    # --- bookkeeping (not part of the result) ---
    "job_name": None,                 # AWS Batch job name / local label; auto-generated if None
    "label": None,                    # free-text campaign label stored in the JSON (was "run_name")
    "scenario": None,                 # 2-5 capital run-name code, e.g. "NK"; inferred for countries
    "is_principal_set": False,        # marks principal research sets (lowercase p in the run name)
    "is_test_run": False,             # prefixes the run name with TEST-
    # --- targets ---
    "target_mode": "prespecified",    # "prespecified" (list of [name, lat, lon]) or "country" (deprecated)
    "target_country": None,
    "target_points": None,
    "target_shell_n_points": 15000,   # country mode only: Fibonacci shell resolution
    # --- engagement physics (CoverageConfig) ---
    "salvo_size": None,
    "n_interceptors_per_sat": None,
    "burnout_velocity_km_s": None,    # interceptor burnout velocity
    "max_accel_g": 10.0,              # interceptor maximum acceleration, multiples of g0
    "detection_time_s": 0.0,
    "decision_time_s": 0.0,
    "target_missile_burnout_time_s": 170.0,
    "intercept_alt_km": 200.0,        # altitude of the target-point shell
    "min_elev_deg": 0.0,
    # --- seed families (Lee exact J2) ---
    "nd_repeat": 3,
    "altitude_min_km": 200.0,         # upper bound is intercept_alt_km + R_max (not configurable)
    "anchor_lat_margin_deg": 0.0,
    "allow_reducible": True,          # allow reducible N_P:N_D rungs (e.g. 45:3)
    "useful_reach_deg": 5.0,          # reach assumed when no CoverageConfig is supplied to the planner
    "raan_step_deg": 0.25,
    "dt_target_s": 120.0,             # nominal time step; the actual dt = T_r / L
    # --- seed selection ---
    "margin_multiplier": 5.0,         # build until worst-target access >= this x r_required
    "trim_margin_multiplier": 2.0,    # trim down to this x r_required
    "max_seeds": 200,                 # safety cap on the number of seeds passed to the MILP
    "seed_workers": 8,
    # --- solver (NOT part of run identity) ---
    "mip_gap": 0.01,
    "time_limit_s": None,
    "stop_at_gap": None,
    "mip_focus": 0,
    "heuristics_frac": 0.05,
    "solver_params": None,
}

#: pre-rename key -> current key. The old key is used only if the new one is absent.
LEGACY_KEYS = {
    "a_g": "max_accel_g",
    "v_bo_km_s": "burnout_velocity_km_s",
    "dt_s": "dt_target_s",
    "max_seeds_safety_cap": "max_seeds",
    "run_name": "label",
}

#: keys dropped silently when reading older configs (inputs of the retired altitude/inclination-sweep pipeline)
IGNORED_LEGACY_KEYS = frozenset({
    "target_altitude_km", "altitude_max_repeat_days", "inclination_sweep_bounds_deg", "inc_screen_step_deg",
    "n_raan_offsets", "polar_band_deg", "polar_inclinations_deg", "anchor_mode",
})


def normalize_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return a complete, type-coerced configuration dict (defaults filled, legacy keys mapped).

    Raises ValueError for missing required keys or inconsistent target settings.
    """
    c: dict[str, Any] = {k: v for k, v in cfg.items() if v is not None or k in CONFIG_DEFAULTS}
    for old, new in LEGACY_KEYS.items():
        if old in c:
            old_val = c.pop(old)
            if c.get(new) is None and old_val is not None:
                c[new] = old_val
    for k in IGNORED_LEGACY_KEYS:
        c.pop(k, None)
    out = {**CONFIG_DEFAULTS, **{k: v for k, v in c.items() if v is not None}}

    missing = [k for k in REQUIRED_KEYS if out.get(k) is None]
    if missing:
        raise ValueError(f"config is missing required key(s): {missing}")

    out["target_mode"] = str(out["target_mode"]).strip().lower()
    if out["target_mode"] == "country":
        if not out.get("target_country"):
            raise ValueError("target_mode='country' requires 'target_country'")
        tc = out["target_country"]
        out["target_country"] = tc.strip() if isinstance(tc, str) else [str(x).strip() for x in tc]
    elif out["target_mode"] == "prespecified":
        pts = out.get("target_points")
        if not pts:
            raise ValueError("target_mode='prespecified' requires a non-empty 'target_points' list of [name, lat, lon]")
        out["target_points"] = [[str(p[0]), float(p[1]), float(p[2])] for p in pts]
    else:
        raise ValueError(f"target_mode must be 'prespecified' or 'country', got {out['target_mode']!r}")

    for k in ("salvo_size", "n_interceptors_per_sat", "nd_repeat", "max_seeds", "seed_workers", "mip_focus",
              "target_shell_n_points"):
        out[k] = int(out[k])
    for k in ("burnout_velocity_km_s", "max_accel_g", "detection_time_s", "decision_time_s",
              "target_missile_burnout_time_s", "intercept_alt_km", "min_elev_deg", "altitude_min_km",
              "anchor_lat_margin_deg", "useful_reach_deg", "raan_step_deg", "dt_target_s", "margin_multiplier",
              "trim_margin_multiplier", "mip_gap", "time_limit_s", "heuristics_frac"):
        out[k] = float(out[k])
    for k in ("is_principal_set", "is_test_run", "allow_reducible"):
        out[k] = bool(out[k])
    if out["stop_at_gap"] is not None:
        out["stop_at_gap"] = float(out["stop_at_gap"])
    if out["solver_params"] is not None:
        out["solver_params"] = dict(out["solver_params"])
    if out["label"] is not None:
        out["label"] = str(out["label"])
    if out["raan_step_deg"] <= 0.0:
        raise ValueError("raan_step_deg must be > 0")
    return out


def settings_from_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Run-identity settings (see names.py) from a configuration. Accepts raw or normalized configs."""
    c = normalize_config(cfg)
    st: dict[str, Any] = {k: c[k] for k in (
        "salvo_size", "detection_time_s", "burnout_velocity_km_s", "max_accel_g",
        "target_missile_burnout_time_s", "decision_time_s", "n_interceptors_per_sat", "intercept_alt_km",
        "min_elev_deg", "dt_target_s", "nd_repeat", "altitude_min_km", "anchor_lat_margin_deg",
        "raan_step_deg", "margin_multiplier", "trim_margin_multiplier", "max_seeds", "allow_reducible",
        "useful_reach_deg")}
    st["scenario"] = c["scenario"]
    st["principal"] = c["is_principal_set"]
    st["test"] = c["is_test_run"]
    if c["target_mode"] == "prespecified":
        st["target_points"] = c["target_points"]
    else:
        st["target_country"] = c["target_country"]
        st["target_shell_n_points"] = c["target_shell_n_points"]
    return st
