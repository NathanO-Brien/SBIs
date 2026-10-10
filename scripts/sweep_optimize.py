"""
sweep_optimize.py

Sequential parameter sweep for the SBI SCLP constellation optimizer.

Iterates over all combinations of:
  - Country target    : China → North Korea (China runs first)
  - Orbit family      : derived from the target latitudes (Lee exact-J2 seed families; no altitude input)
  - Salvo size        : 1, 10, 20, 50, 100
  - Interceptors/sat  : 1, 2, 4, 6
  - Burnout velocity  : 4, 5, 6, 10 km/s

Total: 2 × 5 × 4 × 4 = 160 combinations (before the optional timing/a_g dimensions).

Each run stops at 20 minutes OR a 1% MIP gap, whichever comes first.
Already-completed and infeasible runs are skipped automatically on restart.
Errors are caught, logged, and the sweep continues with the next combination.

Progress is written line-by-line to:
    results/sweep_progress.jsonl

To re-run a specific combination, delete its line from that file and restart
the script.  Runs logged as "error" are automatically retried on the next
invocation.

Usage:
    python scripts/sweep_optimize.py
"""
from __future__ import annotations

import datetime
import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np

from sbi_coverage.core.config import CoverageConfig, EarthConstants, SimConfig
from sbi_coverage.core.shell import Shell
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
# Fixed parameters — must match lee_test2.py exactly.
# Edit lee_test2.py first, then mirror changes here.
# ---------------------------------------------------------------------------
TARGET_MODE: str                                  = "country"
USE_DRAG: bool                                     = False
TARGET_SHELL_N_POINTS: int                        = 15000
# Target temporal resolution — determines L = round(T_r / DT_S) slots per repeat
# period.  The actual simulation timestep is snapped to T_r/L after the RGT orbit
# is selected, so DT_S itself is never used directly as a timestep.
DT_S: float                                       = 120.0
# Old T_WINDOW_S (time from target launch to intercept, assumed == interceptor
# flyout time) split into three inputs -- see CoverageConfig's docstring.
# Defaults (0, 0) leave TARGET_MISSILE_BURNOUT_TIME_S == the old T_WINDOW_S
# value, unchanged.
DETECTION_TIME_S: float                           = 0.0
DECISION_TIME_S: float                            = 0.0
TARGET_MISSILE_BURNOUT_TIME_S: float              = 170.0
A_G: float                                        = 10.0
INTERCEPT_ALT_KM: float                           = 200.0
MIN_ELEV_DEG: float                               = 0.0
# Seed families (Lee exact J2, N_D-day synchronized repeat period).  Altitude and
# inclination are NOT inputs: plan_seed_families() anchors the highest-inclination
# family at the highest target latitude and solves every other family on the same
# Omega_dot.  Altitude spans ALTITUDE_MIN_KM up to intercept_alt + R_max (derived
# per scenario from CoverageConfig; no hard-coded upper bound).
ND_REPEAT: int                                    = 3
ALTITUDE_MIN_KM: float                            = 200.0
ANCHOR_LAT_MARGIN_DEG: float                      = 0.0
# RAAN spacing between seeds of one family, over that family's distinct-track window.
RAAN_STEP_DEG: float                              = 0.25
SEED_SCREENING_WORKERS: int                       = 8
# ---------------------------------------------------------------------------
# Seed selection strategy: worst-case-first, margin-driven build + trim.
# Shared with lee_test2.py / batch_worker.py via sbi_coverage.Lee.seed_pipeline
# (greedy_select_seeds / trim_seeds docstrings have the full rationale). Rather than
# handing the MILP a fixed seed count (the old SEEDS_FOR_MILP_BY_R_REQUIRED
# lookup, removed), seeds are added greedily -- each pick is the seed that most raises the
# worst-covered target's accumulated access -- until
# every target's accumulated access reaches MARGIN_MULTIPLIER x r_required at
# the worst-covered target. A reverse-greedy trim pass then removes whichever
# seeds are least damaging to keep, down to a leaner TRIM_MARGIN_MULTIPLIER x
# r_required floor.
# ---------------------------------------------------------------------------
MARGIN_MULTIPLIER: float = 5.0
TRIM_MARGIN_MULTIPLIER: float = 2.0
# Hard ceiling on total seeds passed to the MILP, regardless of margin
# reached -- a safety valve only, not itself a target to hit.
MAX_SEEDS_FOR_MILP: int = 200
# mip_focus/heuristics_frac/solver_params are intentionally not overridden
# here -- solve_sclp() now defaults to vanilla Gurobi tuning (MIPFocus=0,
# Heuristics=0.05, no solver_params overrides). The architecture to
# specialize these per-run is still in solve_sclp()'s signature; pass them
# explicitly again here if/when that's worth re-enabling.
LOG_DIR: Path | None                              = Path("logFiles")
RESULTS_DIR: Path                                 = Path("results")

PROPAGATOR: str = "Nominal_Propagator"

# ---------------------------------------------------------------------------
# Run metadata (optional; saved into the output JSON only -- never used for
# file/folder naming, which is unaffected by these). For a sweep, each run's
# saved run_name gets "_<n>" appended, n being its 1-based position in grid.
# ---------------------------------------------------------------------------
RUN_NAME: str | None = None
IS_PRINCIPAL_SET: bool = False

# ---------------------------------------------------------------------------
# Sweep grid — China runs execute before North Korea runs.
# ---------------------------------------------------------------------------
SWEEP_COUNTRIES: list[str]                 = ["China", "North Korea"]
SWEEP_SALVO_SIZES: list[int]               = [1, 10, 20, 50, 100]
SWEEP_INTERCEPTORS_PER_SAT: list[int]      = [1, 2, 4, 6]
SWEEP_BURNOUT_VELOCITIES_KM_S: list[float] = [4.0, 5.0, 6.0, 10.0]
# New sweepable dimensions -- default to a single element (the corresponding
# fixed value above) so an unmodified sweep's grid size is unaffected.
SWEEP_DETECTION_TIMES_S: list[float]              = [DETECTION_TIME_S]
SWEEP_DECISION_TIMES_S: list[float]               = [DECISION_TIME_S]
SWEEP_BURNOUT_TIMES_S: list[float]                = [TARGET_MISSILE_BURNOUT_TIME_S]
SWEEP_A_G_LIST: list[float]                       = [A_G]

# Per-run solver budget
SWEEP_TIME_LIMIT_S: float      = 7200.0   # wall time stop (normal runs)
RETRY_TIME_LIMIT_S: float      = 14400.0  # extended time limit for 100%-gap re-runs (4 h)
MIP_GAP_RERUN_THRESHOLD: float = 0.999    # treat MIP gap >= this as "effectively 100%"
SWEEP_MIP_GAP: float           = 0.01     # 1% — stop early if gap reaches this

# One-line-per-run JSON progress log
SWEEP_PROGRESS_FILE: Path = RESULTS_DIR / "sweep_progress.jsonl"


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------

def _run_key_from_entry(entry: dict) -> tuple:
    """Build the dedup key for a progress-log entry.

    The 4 new dims (detection/decision/burnout time, a_g) default to the
    current fixed constants when absent, so log lines written before these
    dims existed still match today's single-element-list sweeps exactly.
    """
    return (
        str(entry["country"]),
        int(entry["salvo_size"]),
        int(entry["n_interceptors_per_sat"]),
        float(entry["v_bo_km_s"]),
        float(entry.get("detection_time_s", DETECTION_TIME_S)),
        float(entry.get("decision_time_s", DECISION_TIME_S)),
        float(entry.get("target_missile_burnout_time_s", TARGET_MISSILE_BURNOUT_TIME_S)),
        float(entry.get("a_g", A_G)),
        # Results from before the Lee exact-J2 seed families (Keplerian RGT, J2 off,
        # requested-altitude sweep) lack this field and must not match new runs.
        str(entry.get("orbit_model", "keplerian_rgt")),
    )


def _load_completed_keys() -> set[tuple]:
    """
    Return the set of run keys already recorded as 'completed', 'infeasible',
    or 'no_incumbent' in the progress log.

    'error' entries are NOT included so they are retried on the next run.
    """
    done: set[tuple] = set()
    if not SWEEP_PROGRESS_FILE.exists():
        return done
    with open(SWEEP_PROGRESS_FILE, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if entry.get("outcome") in ("completed", "infeasible", "no_incumbent"):
                try:
                    done.add(_run_key_from_entry(entry))
                except KeyError:
                    pass
    return done


def _log_progress(entry: dict) -> None:
    """Append one JSON line to the sweep progress file."""
    SWEEP_PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(SWEEP_PROGRESS_FILE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")


def _find_100pct_gap_runs() -> list[dict]:
    """Scan the progress log for completed runs that have a 100% MIP gap.

    For each unique run key, only the *last* recorded entry is considered —
    so a run that was already successfully retried will not be queued again.
    A run is returned for retry only when:
      • outcome == "completed"
      • mip_gap_achieved >= MIP_GAP_RERUN_THRESHOLD  (i.e. effectively 100%)
      • time_limit_s_used < RETRY_TIME_LIMIT_S        (not already retried)

    Returns a list of raw progress-log dicts, one per qualifying run.
    """
    if not SWEEP_PROGRESS_FILE.exists():
        return []

    # Walk the file in order; last entry per key wins.
    latest: dict[tuple, dict] = {}
    with open(SWEEP_PROGRESS_FILE, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if entry.get("outcome") != "completed":
                continue
            try:
                key = _run_key_from_entry(entry)
            except KeyError:
                continue
            latest[key] = entry  # last completed entry for this key wins

    reruns: list[dict] = []
    for entry in latest.values():
        gap = entry.get("mip_gap_achieved")
        if gap is None:
            continue
        if float(gap) < MIP_GAP_RERUN_THRESHOLD:
            continue
        # time_limit_s_used was added in this version; default to SWEEP_TIME_LIMIT_S
        # for older log entries that pre-date this field.
        time_used = float(entry.get("time_limit_s_used", SWEEP_TIME_LIMIT_S))
        if time_used >= RETRY_TIME_LIMIT_S - 1.0:
            continue  # already ran with the extended budget — do not retry again
        reruns.append(entry)

    return reruns


# ---------------------------------------------------------------------------
# Single-combination optimization run
# ---------------------------------------------------------------------------

def run_single(
    *,
    country: str,
    salvo_size: int,
    n_interceptors_per_sat: int,
    v_bo_km_s: float,
    detection_time_s: float = DETECTION_TIME_S,
    decision_time_s: float = DECISION_TIME_S,
    target_missile_burnout_time_s: float = TARGET_MISSILE_BURNOUT_TIME_S,
    a_g: float = A_G,
    time_limit_s: float = SWEEP_TIME_LIMIT_S,
    run_name: str | None = None,
    is_principal_set: bool = False,
) -> dict:
    """
    Execute one full SCLP optimization for the given parameter combination.

    Returns a summary dict:
        outcome           : "completed" | "infeasible" | "no_incumbent"
        actual_alt_km     : float
        n_satellites      : int
        mip_gap_achieved  : float | None
        result_status     : str
    """
    earth = EarthConstants()

    # ------------------------------------------------------------------
    # Coverage geometry
    # ------------------------------------------------------------------
    cov = CoverageConfig(
        earth=earth,
        detection_time_s=detection_time_s,
        decision_time_s=decision_time_s,
        target_missile_burnout_time_s=target_missile_burnout_time_s,
        v_bo_km_s=v_bo_km_s,
        a_g=a_g,
        intercept_alt_km=INTERCEPT_ALT_KM,
        min_elev_deg=MIN_ELEV_DEG,
    )
    print(f"Computed R_max: {cov.max_range_km:.2f} km")

    # ------------------------------------------------------------------
    # Build country target shell
    # ------------------------------------------------------------------
    base_shell = Shell(
        earth,
        n_points=TARGET_SHELL_N_POINTS,
        shell_alt_km=cov.intercept_alt_km,
        analysis_shell_mode="full",
    )
    shell = base_shell.mask_country([country])
    if shell.n_points_used <= 0:
        raise ValueError(
            f"No shell points found inside country={country!r}. "
            "Check the country name against the geodata database."
        )
    lats_deg  = np.asarray(shell.lat_deg, dtype=np.float64)
    lons_deg  = np.asarray(shell.lon_deg, dtype=np.float64)
    n_targets = shell.n_points_used
    labels    = [f"{country} target {i}" for i in range(n_targets)]
    shell.meta["labels"] = labels
    print(f"\nTarget: {country}  n_targets={n_targets}")

    # ------------------------------------------------------------------
    # Seed families: anchored top-down from the target latitudes, synchronized
    # repeat period (Lee Eq. 33), exact secular J2.  Altitude is an OUTPUT.
    # ------------------------------------------------------------------
    plan = plan_seed_families(
        [[labels[i], float(lats_deg[i]), float(lons_deg[i])] for i in range(n_targets)],
        earth=earth,
        N_D=ND_REPEAT,
        alt_min_km=ALTITUDE_MIN_KM,
        anchor_lat_margin_deg=ANCHOR_LAT_MARGIN_DEG,
        cov=cov,
        dt_target_s=DT_S,
    )
    families      = plan["families"]
    N_D           = int(plan["N_D"])
    L             = int(plan["L"])
    dt_s_actual   = float(plan["dt_s"])
    horizon_s     = (L - 1) * dt_s_actual
    anchor        = plan["anchor"]
    actual_alt_km = float(anchor["alt_km"])   # anchor altitude: labels the results folder
    gen = plan["generation"]
    print(
        f"\nSeed families (N_D={N_D}, altitude {plan['alt_bounds_km'][0]:.0f}-{plan['alt_bounds_km'][1]:.0f} km "
        f"= shell + R_max, target latitudes {plan['lat_min_deg']:.2f}-{plan['lat_max_deg']:.2f} deg, "
        f"span {gen['target_lat_span_deg']:.2f}):"
    )
    for f in families:
        red = f" [reducible -> {f['reduced_ratio']}]" if f["reducible"] else ""
        print(f"  {'anchor' if f['is_anchor'] else '      '} {f['N_P']:3d}:{f['N_D']}  i={f['inc_deg']:7.3f} deg  "
              f"alt={f['alt_km']:7.2f} km  reach={f['reach_deg']:.2f} deg  floor={f['inc_floor_deg']:.2f} deg{red}")
    print(f"  unique altitude/inclination combos: generated {gen['n_unique_alt_inc_combos_all_candidates']} "
          f"(chosen ladder {gen['n_unique_alt_inc_combos_chosen_ladder']}), kept {gen['n_unique_alt_inc_combos_kept']}")
    print(f"Shared repeat period T_r={plan['T_r_s']:.3f} s  L={L} steps  dt={dt_s_actual:.6f} s  "
          f"horizon={horizon_s:.1f} s ({horizon_s / 3600:.3f} h)")

    sim = SimConfig(
        horizon_s=horizon_s,
        dt_s=dt_s_actual,
        analysis_matrix_dtype="uint8",
        use_j2=True,
        use_drag=USE_DRAG,
    )

    # ------------------------------------------------------------------
    # Seed pool (family x RAAN), simulation, filter (see Lee/seed_pipeline.py)
    # ------------------------------------------------------------------
    r_required = int(np.ceil(salvo_size / n_interceptors_per_sat))
    seed_meta, sub_layers = build_seed_pool(plan, RAAN_STEP_DEG, earth)
    print(f"\nSeed grid: {len(seed_meta)} seeds  (RAAN step {RAAN_STEP_DEG:.2f} deg within each family's window)")
    v0_list = simulate_seed_pool(
        sub_layers, sim=sim, cov=cov, earth=earth, shell=shell,
        propagator=PROPAGATOR, workers=SEED_SCREENING_WORKERS,
    )
    pool = filter_seed_pool(
        v0_list, sub_layers, seed_meta, plan, r_required,
        margin_multiplier=MARGIN_MULTIPLIER,
        trim_margin_multiplier=TRIM_MARGIN_MULTIPLIER,
        max_seeds=MAX_SEEDS_FOR_MILP,
    )
    param_V    = pool["param_V"]
    sub_layers = pool["sub_layers"]
    seed_meta  = pool["seed_meta"]
    ncols      = pool["ncols"]

    param_r = r_required * np.ones((L, n_targets), dtype=np.float64)
    print(
        f"\nCoverage requirement: salvo_size={salvo_size}, "
        f"interceptors_per_sat={n_interceptors_per_sat}, "
        f"r_required={r_required} satellite(s) in view per time step"
    )

    # ------------------------------------------------------------------
    # Solve SCLP
    # ------------------------------------------------------------------
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_settings = settings_from_config({
        "is_principal_set": is_principal_set, "target_mode": "country", "target_country": country,
        "target_shell_n_points": TARGET_SHELL_N_POINTS, "salvo_size": salvo_size,
        "n_interceptors_per_sat": n_interceptors_per_sat, "v_bo_km_s": v_bo_km_s, "a_g": a_g,
        "detection_time_s": detection_time_s, "decision_time_s": decision_time_s,
        "target_missile_burnout_time_s": target_missile_burnout_time_s, "intercept_alt_km": INTERCEPT_ALT_KM,
        "nd_repeat": ND_REPEAT, "altitude_min_km": ALTITUDE_MIN_KM, "anchor_lat_margin_deg": ANCHOR_LAT_MARGIN_DEG,
        "raan_step_deg": RAAN_STEP_DEG, "margin_multiplier": MARGIN_MULTIPLIER,
        "trim_margin_multiplier": TRIM_MARGIN_MULTIPLIER, "max_seeds_safety_cap": MAX_SEEDS_FOR_MILP,
    })
    run_id = local_run_id(run_settings, RESULTS_DIR)
    print(f"Run name: {run_id}")
    log_file = None
    if LOG_DIR is not None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"{run_id}_gurobi.log"

    print("\n" + "=" * 60)
    print("Solving SCLP")
    print("=" * 60)

    result = solve_sclp(
        param_V=param_V,
        param_r=param_r,
        n_targets=n_targets,
        apply_rgt_symmetry_break=False,
        mip_gap=SWEEP_MIP_GAP,
        time_limit_s=time_limit_s,
        log_file=log_file,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print(f"Status  : {result.status_str}")
    print(f"Selected: {result.n_satellites} satellite(s)")
    print(f"Cost    : {result.obj_val:.1f}")
    _gap_pct = result.mip_gap * 100
    print(
        f"MIP gap : {_gap_pct:.4f}%"
        if np.isfinite(_gap_pct)
        else "MIP gap : N/A (no lower bound established)"
    )
    print(f"Modeling: {result.time_modeling_s:.2f} s   Solve: {result.time_optimization_s:.2f} s")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Classify outcome — handle infeasible and no-incumbent gracefully
    # ------------------------------------------------------------------
    status_upper = result.status_str.upper()

    if "INFEASIBLE" in status_upper or "INF_OR_UNBD" in status_upper:
        print(
            "\n[sweep_optimize] Problem is INFEASIBLE for this parameter combination. "
            "Skipping save."
        )
        return {
            "outcome":          "infeasible",
            "actual_alt_km":    actual_alt_km,
            "n_satellites":     0,
            "mip_gap_achieved": None,
            "result_status":    result.status_str,
        }

    if result.n_satellites == 0:
        print(
            "\n[sweep_optimize] No incumbent found within time/gap budget. "
            "Skipping save."
        )
        return {
            "outcome":          "no_incumbent",
            "actual_alt_km":    actual_alt_km,
            "n_satellites":     0,
            "mip_gap_achieved": None,
            "result_status":    result.status_str,
        }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    orb_elems = build_orb_elems_table(sub_layers, L, ncols)
    print_orb_table(orb_elems, result.selected_slots)

    csv_path = RESULTS_DIR / f"{run_id}_result.csv"

    per_family_counts = family_satellite_counts(result.selected_slots, ncols, seed_meta, families)
    print("Satellites per family: " + ", ".join(f"{k}: {v}" for k, v in per_family_counts.items()))
    seed_cfg = seed_run_config(
        plan, pool, per_family_counts,
        raan_step_deg=RAAN_STEP_DEG,
        margin_multiplier=MARGIN_MULTIPLIER,
        trim_margin_multiplier=TRIM_MARGIN_MULTIPLIER,
        max_seeds=MAX_SEEDS_FOR_MILP,
    )

    run_config = {
        # --- Run metadata ---
        "run_name":                       run_name,
        "run_id":                         run_id,
        "run_settings_key":               settings_key(run_settings),
        "is_principal_set":               is_principal_set,
        # --- Target ---
        "target_mode":                    TARGET_MODE,
        "target_country":                 country,
        "target_shell_n_points":          TARGET_SHELL_N_POINTS,
        "n_targets":                      int(n_targets),
        # --- Engagement physics ---
        "dt_s":                           dt_s_actual,
        "dt_s_nominal":                   DT_S,
        "detection_time_s":               detection_time_s,
        "decision_time_s":                decision_time_s,
        "target_missile_burnout_time_s":  target_missile_burnout_time_s,
        "interceptor_engagement_time_s":  float(cov.interceptor_engagement_time_s),
        "v_bo_km_s":                      v_bo_km_s,
        "a_g":                            a_g,
        "intercept_alt_km":               INTERCEPT_ALT_KM,
        "min_elev_deg":                   MIN_ELEV_DEG,
        "r_max_km":                       float(cov.max_range_km),
        # --- Orbit model, seed generation, seed filter (shared block) ---
        **seed_cfg,
        # --- Intercept salvo ---
        "salvo_size":                     salvo_size,
        "n_interceptors_per_sat":         n_interceptors_per_sat,
        "r_required":                     int(r_required),
        # --- Solver ---
        "mip_gap_target":                 SWEEP_MIP_GAP,
        "time_limit_s":                   time_limit_s,
        # --- Result summary ---
        "timestamp_utc":                  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result_status":                  result.status_str,
        "mip_gap_achieved":               None if not np.isfinite(result.mip_gap) else float(result.mip_gap),
        "n_satellites_selected":          int(result.n_satellites),
        "obj_bound":                      None if not np.isfinite(result.obj_bound) else float(result.obj_bound),
        "time_modeling_s":                float(result.time_modeling_s),
        "time_optimization_s":            float(result.time_optimization_s),
    }

    targets_payload = [
        {"name": labels[i], "lat_deg": float(lats_deg[i]), "lon_deg": float(lons_deg[i])}
        for i in range(n_targets)
    ]
    bc_kg_m2 = float(np.asarray(sub_layers[0].phys.bc_kg_m2).ravel()[0])

    save_sclp_result(
        result,
        orb_elems,
        csv_path,
        cov=cov,
        sim=sim,
        targets=targets_payload,
        countries=[country],
        propagator=PROPAGATOR,
        bc_kg_m2=bc_kg_m2,
        run_config=run_config,
    )

    return {
        "outcome":          "completed",
        "actual_alt_km":    actual_alt_km,
        "n_satellites":     int(result.n_satellites),
        "obj_bound":        None if not np.isfinite(result.obj_bound) else float(result.obj_bound),
        "mip_gap_achieved": None if not np.isfinite(result.mip_gap) else float(result.mip_gap),
        "result_status":    result.status_str,
    }


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if LOG_DIR is not None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)

    grid = list(itertools.product(
        SWEEP_COUNTRIES,
        SWEEP_SALVO_SIZES,
        SWEEP_INTERCEPTORS_PER_SAT,
        SWEEP_BURNOUT_VELOCITIES_KM_S,
        SWEEP_DETECTION_TIMES_S,
        SWEEP_DECISION_TIMES_S,
        SWEEP_BURNOUT_TIMES_S,
        SWEEP_A_G_LIST,
    ))
    n_total = len(grid)

    # -------------------------------------------------------------------
    # Phase 0 — Re-run any 100%-gap entries with an extended time budget
    # -------------------------------------------------------------------
    reruns = _find_100pct_gap_runs()
    if reruns:
        print("=" * 70)
        print(f"  Phase 0 — Re-running {len(reruns)} run(s) with 100% MIP gap")
        print(f"  Extended time limit: {RETRY_TIME_LIMIT_S / 3600:.0f} h  "
              f"(was {SWEEP_TIME_LIMIT_S / 3600:.0f} h)")
        print("=" * 70)
        for rerun_entry in reruns:
            r_country  = str(rerun_entry["country"])
            r_salvo    = int(rerun_entry["salvo_size"])
            r_n_int    = int(rerun_entry["n_interceptors_per_sat"])
            r_vbo      = float(rerun_entry["v_bo_km_s"])
            # New dims default to the current fixed constants for log entries
            # written before these dims existed (see _run_key_from_entry).
            r_detect   = float(rerun_entry.get("detection_time_s", DETECTION_TIME_S))
            r_decide   = float(rerun_entry.get("decision_time_s", DECISION_TIME_S))
            r_burnout  = float(rerun_entry.get("target_missile_burnout_time_s", TARGET_MISSILE_BURNOUT_TIME_S))
            r_a_g      = float(rerun_entry.get("a_g", A_G))
            r_combo    = (r_country, r_salvo, r_n_int, r_vbo, r_detect, r_decide, r_burnout, r_a_g)
            r_run_name = f"{RUN_NAME}_{grid.index(r_combo) + 1}" if RUN_NAME and r_combo in grid else RUN_NAME
            print(
                f"\n[Phase 0]  {r_country}  "
                f"salvo={r_salvo}  int/sat={r_n_int}  v_bo={r_vbo:.1f} km/s  "
                f"detect={r_detect:.0f}s  decide={r_decide:.0f}s  burnout={r_burnout:.0f}s  a_g={r_a_g:.1f}"
            )
            t_run0 = time.perf_counter()
            try:
                summary  = run_single(
                    country=r_country,
                    salvo_size=r_salvo,
                    n_interceptors_per_sat=r_n_int,
                    v_bo_km_s=r_vbo,
                    detection_time_s=r_detect,
                    decision_time_s=r_decide,
                    target_missile_burnout_time_s=r_burnout,
                    a_g=r_a_g,
                    time_limit_s=RETRY_TIME_LIMIT_S,
                    run_name=r_run_name,
                    is_principal_set=IS_PRINCIPAL_SET,
                )
                wall_s       = time.perf_counter() - t_run0
                outcome      = summary["outcome"]
                gap_achieved = summary.get("mip_gap_achieved")
                gap_str      = f"{gap_achieved * 100:.3f}%" if gap_achieved is not None else "N/A"

                _log_progress({
                    "outcome":                outcome,
                    "country":                r_country,
                    "orbit_model":            "lee_j2",
                    "salvo_size":             r_salvo,
                    "n_interceptors_per_sat": r_n_int,
                    "v_bo_km_s":              r_vbo,
                    "detection_time_s":       r_detect,
                    "decision_time_s":        r_decide,
                    "target_missile_burnout_time_s": r_burnout,
                    "a_g":                    r_a_g,
                    "actual_alt_km":          summary.get("actual_alt_km"),
                    "n_satellites":           summary.get("n_satellites"),
                    "obj_bound":              summary.get("obj_bound"),
                    "mip_gap_achieved":       gap_achieved,
                    "result_status":          summary.get("result_status"),
                    "time_limit_s_used":      RETRY_TIME_LIMIT_S,
                    "wall_time_s":            round(wall_s, 1),
                    "timestamp_utc":          datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })

                _bound    = summary.get("obj_bound")
                bound_str = f"{_bound:.1f}" if _bound is not None else "—"
                print(
                    f"\n→ [Phase 0] {outcome.upper():<14}  "
                    f"sats={summary.get('n_satellites', '—'):<6}  "
                    f"bound={bound_str:<8}  "
                    f"gap={gap_str:<10}  "
                    f"wall={wall_s:.0f}s"
                )

            except Exception as exc:
                wall_s = time.perf_counter() - t_run0
                print(f"\n[Phase 0] ERROR  {exc}")
                traceback.print_exc()
                _log_progress({
                    "outcome":                "error",
                    "country":                r_country,
                    "orbit_model":            "lee_j2",
                    "salvo_size":             r_salvo,
                    "n_interceptors_per_sat": r_n_int,
                    "v_bo_km_s":              r_vbo,
                    "detection_time_s":       r_detect,
                    "decision_time_s":        r_decide,
                    "target_missile_burnout_time_s": r_burnout,
                    "a_g":                    r_a_g,
                    "error":                  str(exc),
                    "time_limit_s_used":      RETRY_TIME_LIMIT_S,
                    "wall_time_s":            round(wall_s, 1),
                    "timestamp_utc":          datetime.datetime.now(datetime.timezone.utc).isoformat(),
                })

        print(f"\n  Phase 0 complete — {len(reruns)} re-run(s) finished.")

    # Reload after Phase 0 so the main sweep sees retry outcomes as completed.
    completed_keys = _load_completed_keys()
    n_already_done = sum(1 for combo in grid if combo in completed_keys)
    n_remaining = n_total - n_already_done

    print("=" * 70)
    print("sweep_optimize.py  —  SBI SCLP Parameter Sweep")
    print("=" * 70)
    print(f"  Combinations       : {n_total}  "
          f"({' × '.join(str(len(x)) for x in [SWEEP_COUNTRIES, SWEEP_SALVO_SIZES, SWEEP_INTERCEPTORS_PER_SAT, SWEEP_BURNOUT_VELOCITIES_KM_S, SWEEP_DETECTION_TIMES_S, SWEEP_DECISION_TIMES_S, SWEEP_BURNOUT_TIMES_S, SWEEP_A_G_LIST])})")
    print(f"  Already completed  : {n_already_done}")
    print(f"  Remaining          : {n_remaining}")
    print(f"  Time budget / run  : {SWEEP_TIME_LIMIT_S / 60:.0f} min  OR  MIP gap ≤ {SWEEP_MIP_GAP * 100:.1f}%")
    est_h = n_remaining * SWEEP_TIME_LIMIT_S / 3600.0
    print(f"  Est. max runtime   : {est_h:.1f} h  ({est_h / 24:.1f} days)")
    print(f"  Progress log       : {SWEEP_PROGRESS_FILE}")
    print("=" * 70)

    for combo_idx, (country, salvo, n_int, v_bo, detect_s, decide_s, burnout_s, a_g_val) in enumerate(grid, 1):
        key = (country, int(salvo), int(n_int), float(v_bo),
               float(detect_s), float(decide_s), float(burnout_s), float(a_g_val), "lee_j2")
        combo_run_name = f"{RUN_NAME}_{combo_idx}" if RUN_NAME else None

        if key in completed_keys:
            print(
                f"\n[{combo_idx:3d}/{n_total}] SKIP  {country}  "
                f"salvo={salvo}  int/sat={n_int}  v_bo={v_bo:.0f}km/s  "
                f"detect={detect_s:.0f}s  decide={decide_s:.0f}s  burnout={burnout_s:.0f}s  a_g={a_g_val:.1f}"
            )
            continue

        print(f"\n{'=' * 70}")
        print(
            f"[{combo_idx:3d}/{n_total}]  {country}  "
            f"salvo={salvo}  int/sat={n_int}  v_bo={v_bo:.0f}km/s  "
            f"detect={detect_s:.0f}s  decide={decide_s:.0f}s  burnout={burnout_s:.0f}s  a_g={a_g_val:.1f}"
        )
        print(f"{'=' * 70}")

        t_run0 = time.perf_counter()
        try:
            summary = run_single(
                country=country,
                salvo_size=int(salvo),
                n_interceptors_per_sat=int(n_int),
                v_bo_km_s=float(v_bo),
                detection_time_s=float(detect_s),
                decision_time_s=float(decide_s),
                target_missile_burnout_time_s=float(burnout_s),
                a_g=float(a_g_val),
                time_limit_s=SWEEP_TIME_LIMIT_S,
                run_name=combo_run_name,
                is_principal_set=IS_PRINCIPAL_SET,
            )
            wall_s  = time.perf_counter() - t_run0
            outcome = summary["outcome"]

            gap_achieved = summary.get("mip_gap_achieved")
            gap_str = f"{gap_achieved * 100:.3f}%" if gap_achieved is not None else "N/A"

            _log_progress({
                "outcome":                outcome,
                "country":                country,
                "orbit_model":            "lee_j2",
                "salvo_size":             salvo,
                "n_interceptors_per_sat": n_int,
                "v_bo_km_s":              v_bo,
                "detection_time_s":       detect_s,
                "decision_time_s":        decide_s,
                "target_missile_burnout_time_s": burnout_s,
                "a_g":                    a_g_val,
                "actual_alt_km":          summary.get("actual_alt_km"),
                "n_satellites":           summary.get("n_satellites"),
                "obj_bound":              summary.get("obj_bound"),
                "mip_gap_achieved":       gap_achieved,
                "result_status":          summary.get("result_status"),
                "time_limit_s_used":      SWEEP_TIME_LIMIT_S,
                "wall_time_s":            round(wall_s, 1),
                "timestamp_utc":          datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            completed_keys.add(key)

            _bound = summary.get("obj_bound")
            bound_str = f"{_bound:.1f}" if _bound is not None else "—"
            print(
                f"\n→ {outcome.upper():<14}  "
                f"sats={summary.get('n_satellites', '—'):<6}  "
                f"bound={bound_str:<8}  "
                f"gap={gap_str:<10}  "
                f"wall={wall_s:.0f}s"
            )

        except Exception as exc:
            wall_s = time.perf_counter() - t_run0
            print(f"\n[sweep_optimize] ERROR  {exc}")
            traceback.print_exc()
            _log_progress({
                "outcome":                "error",
                "country":                country,
                "orbit_model":            "lee_j2",
                "salvo_size":             salvo,
                "n_interceptors_per_sat": n_int,
                "v_bo_km_s":              v_bo,
                "detection_time_s":       detect_s,
                "decision_time_s":        decide_s,
                "target_missile_burnout_time_s": burnout_s,
                "a_g":                    a_g_val,
                "error":                  str(exc),
                "wall_time_s":            round(wall_s, 1),
                "timestamp_utc":          datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            # Error runs are NOT added to completed_keys — they will be retried
            # on the next invocation.  Delete the error line from sweep_progress.jsonl
            # to suppress the retry.

    print("\n" + "=" * 70)
    print("Sweep complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
