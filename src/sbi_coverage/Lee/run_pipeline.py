"""One end-to-end optimization run: configuration in, result CSV + JSON out.

``run_sclp`` is the single implementation behind lee_test2.py (local), cloud_sweep.py --local and
batch_worker.py (AWS Batch); the entry points differ only in where the configuration comes from and
where the files go afterwards, so their results and JSON can never drift apart.

Flow: normalize config -> CoverageConfig -> target shell -> plan seed families (altitude is an output)
-> seed pool (family x RAAN) -> simulate seeds -> filter -> MILP -> result CSV + JSON sidecar.
"""
from __future__ import annotations

import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..core.config import CoverageConfig, EarthConstants, SimConfig
from ..core.shell import Shell
from ..core.target_points import load_target_points_csv  # noqa: F401  (re-exported for launchers)
from ..runs.config import normalize_config, settings_from_config
from ..runs.names import local_run_name, scenario_code, settings_hash, settings_key
from ..runs.provenance import collect_provenance
from .milp_solver import (GUROBI_FIXED_PARAMS, build_orb_elems_table, print_orb_table, save_sclp_result,
                          solve_sclp)
from .seed_families import plan_seed_families
from .seed_pipeline import (build_seed_pool, family_satellite_counts, filter_seed_pool, seed_run_config,
                            simulate_seed_pool, slot_columns)

PROPAGATOR = "Nominal_Propagator"


def build_target_shell(c: dict, earth: EarthConstants, cov: CoverageConfig):
    """Target shell for a normalized config: (shell, labels, lats_deg, lons_deg, countries).

    Prespecified mode uses the given [name, lat, lon] points; country mode (deprecated) masks a
    Fibonacci shell to the country polygon(s) and reports the countries for downstream rendering.
    """
    if c["target_mode"] == "prespecified":
        labels = [p[0] for p in c["target_points"]]
        lats = np.array([p[1] for p in c["target_points"]], dtype=np.float64)
        lons = np.array([p[2] for p in c["target_points"]], dtype=np.float64)
        shell = Shell.from_points(earth, lat_deg=lats, lon_deg=lons, shell_alt_km=cov.intercept_alt_km,
                                  meta_extra={"labels": labels, "roi": "prespecified target points"})
        return shell, labels, lats, lons, []

    countries = [c["target_country"]] if isinstance(c["target_country"], str) else list(c["target_country"])
    base = Shell(earth, n_points=c["target_shell_n_points"], shell_alt_km=cov.intercept_alt_km,
                 analysis_shell_mode="full")
    shell = base.mask_country(countries)
    if shell.n_points_used <= 0:
        raise ValueError(f"No shell points found inside {countries!r}; raise target_shell_n_points or check the name.")
    lats = np.asarray(shell.lat_deg, dtype=np.float64)
    lons = np.asarray(shell.lon_deg, dtype=np.float64)
    labels = [f"{', '.join(countries)} target {i}" for i in range(shell.n_points_used)]
    shell.meta["labels"] = labels
    return shell, labels, lats, lons, countries


def write_failure_record(out_dir: str | Path, run_name: str, cfg: dict, exc: BaseException) -> Path:
    """Write ``<run_name>_failed.json`` so a failed run is visible instead of silently absent."""
    path = Path(out_dir) / f"{run_name}_failed.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 2, "run_name": run_name, "status": "failed",
        "error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc(),
        "config": {k: v for k, v in cfg.items() if k != "target_points"},
        "provenance": collect_provenance(),
    }
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def run_sclp(
    cfg: dict[str, Any],
    *,
    out_dir: str | Path,
    run_name: str | None = None,
    log: Callable[[str], None] = print,
    solver_verbose: bool = True,
) -> dict[str, Any]:
    """Run one full optimization and write ``<run_name>_result.csv/.json`` (+ ``_gurobi.log``) to ``out_dir``.

    ``cfg`` is a launcher configuration (see runs/config.py; legacy keys accepted). ``run_name`` is
    normally reserved by the caller (batch_worker reserves it atomically in S3); if omitted a local
    name with the next free same-day sequence letter is generated from ``out_dir``.

    Raises ValueError / RuntimeError on infeasible or failed solves; callers decide how to record that
    (see write_failure_record). Returns a summary dict with the run name, file paths and headline numbers.
    """
    t_start = time.perf_counter()
    c = normalize_config(cfg)
    settings = settings_from_config(c)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_name = run_name or local_run_name(settings, out_dir)
    log(f"Run name: {run_name}")

    # ---- engagement geometry and targets ------------------------------------------------------
    earth = EarthConstants()
    cov = CoverageConfig(
        earth=earth,
        detection_time_s=c["detection_time_s"], decision_time_s=c["decision_time_s"],
        target_missile_burnout_time_s=c["target_missile_burnout_time_s"],
        burnout_velocity_km_s=c["burnout_velocity_km_s"], max_accel_g=c["max_accel_g"],
        intercept_alt_km=c["intercept_alt_km"], min_elev_deg=c["min_elev_deg"],
    )
    log(f"Interceptor engagement time: {cov.interceptor_engagement_time_s:.2f} s "
        f"(burnout {c['target_missile_burnout_time_s']:.0f} s - detection {c['detection_time_s']:.0f} s "
        f"- decision {c['decision_time_s']:.0f} s);  R_max = {cov.max_range_km:.2f} km")
    shell, labels, lats_deg, lons_deg, countries = build_target_shell(c, earth, cov)
    n_targets = len(labels)
    log(f"Targets: mode={c['target_mode']}  n_targets={n_targets}"
        + (f"  countries={', '.join(countries)}" if countries else ""))

    # ---- seed families: anchored top-down from the target latitudes; altitude is an OUTPUT -----
    plan = plan_seed_families(
        [[labels[i], float(lats_deg[i]), float(lons_deg[i])] for i in range(n_targets)],
        earth=earth, N_D=c["nd_repeat"], alt_min_km=c["altitude_min_km"],
        anchor_lat_margin_deg=c["anchor_lat_margin_deg"], useful_reach_deg=c["useful_reach_deg"], cov=cov,
        allow_reducible=c["allow_reducible"], dt_target_s=c["dt_target_s"],
    )
    families, L, dt_s = plan["families"], int(plan["L"]), float(plan["dt_s"])
    horizon_s = (L - 1) * dt_s
    gen = plan["generation"]
    log(f"\nSeed families (N_D={plan['N_D']}, altitude {plan['alt_bounds_km'][0]:.0f}-{plan['alt_bounds_km'][1]:.0f} km, "
        f"target latitude span {gen['target_lat_span_deg']:.2f} deg; unique alt/inc combos generated "
        f"{gen['n_unique_alt_inc_combos_all_candidates']}, kept {gen['n_unique_alt_inc_combos_kept']}):")
    for f in families:
        red = f" [reducible -> {f['reduced_ratio']}]" if f["reducible"] else ""
        log(f"  {'anchor' if f['is_anchor'] else '      '} {f['N_P']:3d}:{f['N_D']}  i={f['inc_deg']:7.3f} deg  "
            f"alt={f['alt_km']:7.2f} km  reach={f['reach_deg']:.2f} deg  floor={f['inc_floor_deg']:.2f} deg{red}")
    log(f"Shared repeat period T_r={plan['T_r_s']:.3f} s  L={L} steps  dt={dt_s:.6f} s "
        f"(nominal {c['dt_target_s']:.1f} s)  horizon={horizon_s / 3600:.3f} h")

    sim = SimConfig(horizon_s=horizon_s, dt_s=dt_s, analysis_matrix_dtype="uint8", use_j2=True)

    # ---- seed pool -> simulate -> filter ---------------------------------------------------------
    r_required = int(math.ceil(c["salvo_size"] / c["n_interceptors_per_sat"]))
    seed_meta, seed_layers, n_raan_offsets = build_seed_pool(plan, c["raan_step_deg"], earth)
    log(f"\nSeed grid: {len(seed_meta)} seeds (RAAN step {c['raan_step_deg']:.2f} deg within each family's window)")
    for f, n in zip(families, n_raan_offsets):
        log(f"  family {f['N_P']}:{f['N_D']}: {n} RAAN offsets in [0, {f['raan_interval_deg']:.3f}) deg")
    t_seed = time.perf_counter()
    profiles = simulate_seed_pool(seed_layers, sim=sim, cov=cov, earth=earth, shell=shell, propagator=PROPAGATOR,
                                  workers=c["seed_workers"], expected_steps=L, log=log)
    seed_sim_time_s = time.perf_counter() - t_seed
    pool = filter_seed_pool(profiles, seed_layers, seed_meta, plan, r_required,
                            margin_multiplier=c["margin_multiplier"],
                            trim_margin_multiplier=c["trim_margin_multiplier"], max_seeds=c["max_seeds"], log=log)
    del profiles

    # ---- MILP -------------------------------------------------------------------------------------
    param_r = r_required * np.ones((L, n_targets), dtype=np.float64)
    log(f"\nCoverage requirement: salvo_size={c['salvo_size']}, interceptors_per_sat={c['n_interceptors_per_sat']}, "
        f"r_required={r_required} satellite(s) in view per time step")
    gurobi_log = out_dir / f"{run_name}_gurobi.log"
    log("\n" + "=" * 60 + "\nSolving SCLP  (Ctrl+C stops the solver and returns the best incumbent)\n" + "=" * 60)
    result = solve_sclp(
        param_V=pool["param_V"], param_r=param_r, apply_rgt_symmetry_break=False,
        mip_gap=c["mip_gap"], time_limit_s=c["time_limit_s"], log_file=gurobi_log, verbose=solver_verbose,
        mip_focus=c["mip_focus"], heuristics_frac=c["heuristics_frac"], solver_params=c["solver_params"],
        stop_at_gap=c["stop_at_gap"],
    )
    gap_pct = result.mip_gap * 100
    log("\n" + "=" * 60 + f"\nStatus  : {result.status_str}\nSelected: {result.n_satellites} satellite(s)\n"
        + (f"MIP gap : {gap_pct:.4f}%" if math.isfinite(gap_pct) else "MIP gap : N/A (no lower bound established)")
        + f"\nModeling: {result.time_modeling_s:.2f} s   Solve: {result.time_optimization_s:.2f} s\n" + "=" * 60)

    # ---- outputs ----------------------------------------------------------------------------------
    orb_elems = build_orb_elems_table(pool["seed_layers"], pool["cols_per_seed"])
    print_orb_table(orb_elems, result.selected_slots)
    per_family = family_satellite_counts(result.selected_slots, pool["cols_per_seed"], pool["seed_meta"], families)
    log("Satellites per family: " + ", ".join(f"{k}: {v}" for k, v in per_family.items()))

    seed_cfg = seed_run_config(
        plan, pool, n_raan_offsets, per_family, raan_step_deg=c["raan_step_deg"],
        margin_multiplier=c["margin_multiplier"], trim_margin_multiplier=c["trim_margin_multiplier"],
        max_seeds=c["max_seeds"], seed_sim_time_s=seed_sim_time_s,
    )
    key_settings = {k: v for k, v in settings.items() if k != "target_points"}
    run_config = {
        # --- identity ---
        "run_name": run_name, "label": c["label"], "scenario": scenario_code(settings),
        "is_principal_set": c["is_principal_set"], "is_test_run": c["is_test_run"],
        "settings_key": settings_key(settings), "settings_hash": settings_hash(settings), "settings": key_settings,
        # --- targets ---
        "target_mode": c["target_mode"], "target_country": c["target_country"] if c["target_mode"] == "country" else None,
        "target_shell_n_points": c["target_shell_n_points"] if c["target_mode"] == "country" else None,
        "n_targets": int(n_targets),
        # --- engagement ---
        "detection_time_s": c["detection_time_s"], "decision_time_s": c["decision_time_s"],
        "target_missile_burnout_time_s": c["target_missile_burnout_time_s"],
        "interceptor_engagement_time_s": float(cov.interceptor_engagement_time_s),
        "burnout_velocity_km_s": c["burnout_velocity_km_s"], "max_accel_g": c["max_accel_g"],
        "intercept_alt_km": c["intercept_alt_km"], "min_elev_deg": c["min_elev_deg"], "r_max_km": float(cov.max_range_km),
        # --- salvo ---
        "salvo_size": c["salvo_size"], "n_interceptors_per_sat": c["n_interceptors_per_sat"], "r_required": r_required,
        # --- seed generation inputs not stored elsewhere ---
        "dt_target_s": c["dt_target_s"], "allow_reducible": c["allow_reducible"], "useful_reach_deg": c["useful_reach_deg"],
        **seed_cfg,
        # --- solver ---
        "solver": {
            "mip_gap_target": c["mip_gap"], "time_limit_s": c["time_limit_s"], "stop_at_gap": c["stop_at_gap"],
            "mip_focus": c["mip_focus"], "heuristics_frac": c["heuristics_frac"], "solver_params": c["solver_params"],
            "fixed_params": dict(GUROBI_FIXED_PARAMS), "symmetry_break": False,
        },
        "wall_time_s": float(time.perf_counter() - t_start),
    }
    targets_payload = [{"name": labels[i], "lat_deg": float(lats_deg[i]), "lon_deg": float(lons_deg[i])}
                       for i in range(n_targets)]
    csv_path = out_dir / f"{run_name}_result.csv"
    json_path = save_sclp_result(
        result, orb_elems, csv_path, cov=cov, sim=sim, targets=targets_payload, countries=countries,
        run_name=run_name, run_config=run_config, provenance=collect_provenance(),
        extra_columns=slot_columns(pool["seed_meta"], pool["cols_per_seed"], families),
    )
    return {
        "run_name": run_name, "csv_path": csv_path, "json_path": json_path, "gurobi_log": gurobi_log,
        "n_satellites": int(result.n_satellites), "status": result.status_str,
        "mip_gap_achieved": None if not math.isfinite(result.mip_gap) else float(result.mip_gap),
        "anchor_alt_km": float(plan["anchor"]["alt_km"]), "families": run_config["families"],
        "n_targets": int(n_targets),
    }
