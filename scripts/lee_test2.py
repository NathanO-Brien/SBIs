"""
lee_test2.py

Minimal SCLP constellation design workflow.

Builds the RGT visibility matrix, solves the SCLP directly in Python using
Gurobi, and saves the selected constellation's orbital characteristics to CSV.

Orbit model: Lee et al. (2020) exact secular J2 (RAAN, argument of perigee and
mean-anomaly rates).  Seed families are N_D = ND_REPEAT synchronized
(N_P, inclination, altitude) tuples that share one repeat period T_r; they are
anchored top-down from the target latitudes (see Lee/seed_families.py), so
altitude is an output, not an input.

Usage:
    python scripts/lee_test2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sbi_coverage.core.config import CoverageConfig, EarthConstants, SimConfig
from sbi_coverage.core.shell import Shell
from sbi_coverage.core.target_points import load_target_points_csv
from sbi_coverage.runnames import local_run_id, settings_from_config, settings_key
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
# Problem definition
# ---------------------------------------------------------------------------

TARGET_MODE: str = "country"  # "prespecified" or "country"

# For use in prespecified mode
TARGET_POINTS: list[tuple[str, float, float]] = [
    ("NK: Yongbyon (nuclear)", 39.79, 125.66),
]
# If set, overrides TARGET_POINTS above by loading a CSV with a header row
# containing some case-insensitive variant of Location/Lat/Lon (see
# sbi_coverage.core.target_points.load_target_points_csv for the exact
# accepted column names and formatting tolerance).
TARGET_POINTS_CSV: str | Path | None = None

# For use in country mode
TARGET_COUNTRY: str | list[str] | tuple[str, ...] = "Brazil"
TARGET_SHELL_N_POINTS: int = 15000

# Time discretization
DT_S: float = 120.0

# Interceptor / engagement geometry inputs.
# Old T_WINDOW_S (time from target launch to intercept, assumed == interceptor
# flyout time) is now split into three inputs: DETECTION_TIME_S (target launch
# to detection) and DECISION_TIME_S (detection to engage decision) both eat
# into TARGET_MISSILE_BURNOUT_TIME_S (target launch to target burnout) before
# any of it is available to the interceptor's own flight -- see
# CoverageConfig's docstring for the exact derivation. Defaults (0, 0) leave
# TARGET_MISSILE_BURNOUT_TIME_S == the old T_WINDOW_S value, unchanged.
DETECTION_TIME_S: float = 0.0
DECISION_TIME_S: float = 0.0
TARGET_MISSILE_BURNOUT_TIME_S: float = 170.0
V_BO_KM_S: float = 4.0
A_G: float = 10.0
INTERCEPT_ALT_KM: float = 200.0
MIN_ELEV_DEG: float = 0.0

# Seed family configuration (Lee exact-J2, synchronized repeat period).
# Altitude and inclination are NOT inputs: plan_seed_families() anchors the
# highest-inclination family at the highest target latitude (+ANCHOR margin)
# and solves every other family (other N_P) on the same Omega_dot, so all of
# them share one repeat period T_r.  Families whose altitude is below
# ALTITUDE_MIN_KM or above the scenario's maximum useful altitude, or whose
# inclination is too low to reach any target (lowest target latitude minus the
# footprint reach at that altitude), are dropped.
#
# There is no hard-coded upper altitude bound: it is the intercept-shell altitude
# plus the interceptor's maximum range R_max (CoverageConfig.max_range_km),
# beyond which no seed has any of the shell within interceptor range.  Seeds
# near that limit see almost none of the shell, so the seed filter prunes them.
ND_REPEAT: int = 3
ALTITUDE_MIN_KM: float = 200.0
# Anchor inclination = highest |target latitude| - ANCHOR_LAT_MARGIN_DEG.
ANCHOR_LAT_MARGIN_DEG: float = 0.0

# Seed simulation configuration
# RAAN spacing between seeds of the same family.  Each family is sampled every
# RAAN_STEP_DEG over its own distinct-track window (360 / reduced N_P degrees);
# offsets beyond that window repeat the same ground tracks.
RAAN_STEP_DEG: float = 0.25
SEED_SCREENING_WORKERS: int = 8

# ---------------------------------------------------------------------------
# Seed selection strategy: worst-case-first, margin-driven build + trim.
#
# Rather than handing the MILP a fixed seed count (the old N_SEEDS_FOR_MILP
# knob), seeds are added greedily -- each pick is the seed that most raises
# the worst-covered target's accumulated access (the quantity the satellite
# count actually depends on), with rarity-weighted capped access only as a
# tie-break -- until every target's accumulated access reaches
# MARGIN_MULTIPLIER x r_required at the worst-covered target. That number
# doesn't need to be hand-tuned per r_required/n_targets combination the way
# a fixed count did. A reverse-greedy trim pass then removes whichever
# seeds are least damaging to keep, down to a leaner TRIM_MARGIN_MULTIPLIER x
# r_required floor, to recover most of the extra seed count spent building a
# safety margin during construction. See _greedy_select_seeds/_trim_seeds.
# ---------------------------------------------------------------------------
MARGIN_MULTIPLIER: float = 5.0
TRIM_MARGIN_MULTIPLIER: float = 2.0
# Hard ceiling on total seeds passed to the MILP, regardless of margin
# reached -- a safety valve only (e.g. a target that's geometrically
# unreachable at the target margin within the candidate grid), not itself a
# target to hit. The build stops on margin, not on this count, unless this
# cap binds first.
MAX_SEEDS_FOR_MILP: int = 200

# ---------------------------------------------------------------------------
# Intercept salvo parameters
# ---------------------------------------------------------------------------
SALVO_SIZE: int = 2
N_INTERCEPTORS_PER_SAT: int = 1

# ---------------------------------------------------------------------------
# Run metadata (optional; saved into the output JSON only -- never used for
# file/folder naming, which is unaffected by these).
# ---------------------------------------------------------------------------
RUN_NAME: str | None = None
IS_PRINCIPAL_SET: bool = False
# Short scenario code used in the run name (e.g. "NK", "CN"). None -> inferred from the country.
SCENARIO: str | None = None

# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------
MIP_GAP: float = 0.001
TIME_LIMIT_S: float = 45
# If set, stop as soon as a live incumbent's gap drops below this value,
# regardless of TIME_LIMIT_S -- e.g. 0.999999 stops as soon as any
# non-trivial lower bound is established (gap under 100%). None: no early
# stop beyond MIP_GAP/TIME_LIMIT_S. Intended for quickly comparing seed
# pools/problem sizes across test runs without waiting out a full time
# budget on each one.
STOP_AT_GAP: float | None = None
# mip_focus/heuristics_frac/solver_params are intentionally not overridden
# here -- solve_sclp() now defaults to vanilla Gurobi tuning (MIPFocus=0,
# Heuristics=0.05, no solver_params overrides). The architecture to
# specialize these per-run is still in solve_sclp()'s signature; pass them
# explicitly again here if/when that's worth re-enabling.
LOG_DIR: Path | None = Path("logFiles")
RESULTS_DIR: Path = Path("results")


def _normalize_country_input(value: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(value, str):
        countries = [value]
    else:
        countries = [str(item) for item in value]
    return [country.strip() for country in countries if country.strip()]


def _dedupe_altitudes_km(values_km: list[float], *, tol_km: float = 1e-6) -> list[float]:
    out: list[float] = []
    for value in values_km:
        x = float(value)
        if not np.isfinite(x) or x <= 0.0:
            raise ValueError(f"All altitude inputs must be finite and > 0 km. Got {value!r}.")
        if not any(abs(x - existing) <= tol_km for existing in out):
            out.append(x)
    return out


def _build_target_shell(earth: EarthConstants, cov: CoverageConfig) -> tuple[Shell, list[str], np.ndarray, np.ndarray]:
    mode = str(TARGET_MODE).strip().lower()

    if mode == "prespecified":
        labels = [point[0] for point in TARGET_POINTS]
        lats_deg = np.array([point[1] for point in TARGET_POINTS], dtype=np.float64)
        lons_deg = np.array([point[2] for point in TARGET_POINTS], dtype=np.float64)
        shell = Shell.from_points(
            earth,
            lat_deg=lats_deg,
            lon_deg=lons_deg,
            shell_alt_km=cov.intercept_alt_km,
            meta_extra={"labels": labels, "roi": "prespecified target points"},
        )
        return shell, labels, lats_deg, lons_deg

    if mode == "country":
        countries = _normalize_country_input(TARGET_COUNTRY)
        if not countries:
            raise ValueError("TARGET_COUNTRY must contain at least one country.")

        base_shell = Shell(
            earth,
            n_points=TARGET_SHELL_N_POINTS,
            shell_alt_km=cov.intercept_alt_km,
            analysis_shell_mode="full",
        )
        shell = base_shell.mask_country(countries)
        if shell.n_points_used <= 0:
            raise ValueError(
                f"No shell points found inside TARGET_COUNTRY={TARGET_COUNTRY!r}. "
                "Increase TARGET_SHELL_N_POINTS or check the country name."
            )

        lats_deg = np.asarray(shell.lat_deg, dtype=np.float64)
        lons_deg = np.asarray(shell.lon_deg, dtype=np.float64)
        label_prefix = ", ".join(countries)
        labels = [f"{label_prefix} target {i}" for i in range(shell.n_points_used)]
        shell.meta["labels"] = labels
        return shell, labels, lats_deg, lons_deg

    raise ValueError("TARGET_MODE must be either 'prespecified' or 'country'")


def main() -> None:
    global TARGET_POINTS
    if str(TARGET_MODE).strip().lower() == "prespecified" and TARGET_POINTS_CSV:
        TARGET_POINTS = load_target_points_csv(TARGET_POINTS_CSV)

    countries = _normalize_country_input(TARGET_COUNTRY)
    if not countries:
        raise ValueError("TARGET_COUNTRY must contain at least one country.")

    earth = EarthConstants()
    propagator = "Nominal_Propagator"

    cov = CoverageConfig(
        earth=earth,
        detection_time_s=DETECTION_TIME_S,
        decision_time_s=DECISION_TIME_S,
        target_missile_burnout_time_s=TARGET_MISSILE_BURNOUT_TIME_S,
        v_bo_km_s=V_BO_KM_S,
        a_g=A_G,
        intercept_alt_km=INTERCEPT_ALT_KM,
        min_elev_deg=MIN_ELEV_DEG,
    )

    print(
        f"Interceptor engagement time: {cov.interceptor_engagement_time_s:.2f} s "
        f"(burnout {TARGET_MISSILE_BURNOUT_TIME_S:.0f}s - detection {DETECTION_TIME_S:.0f}s "
        f"- decision {DECISION_TIME_S:.0f}s)"
    )
    print(f"Computed R_max: {cov.max_range_km:.2f} km")

    shell, labels, lats_deg, lons_deg = _build_target_shell(earth, cov)
    n_targets = len(labels)

    if n_targets == 1:
        print(f"\nTarget: {labels[0]}  lat={lats_deg[0]:.2f} deg  lon={lons_deg[0]:.2f} deg")
    else:
        print(f"\nTarget source summary: mode={TARGET_MODE}  n_targets={n_targets}")
        if str(TARGET_MODE).strip().lower() == "country":
            print(f"Country shell target source: {', '.join(countries)}  base shell N={TARGET_SHELL_N_POINTS}")
    print(f"Countries for downstream reporting: {', '.join(countries)}")

    # ------------------------------------------------------------------
    # Seed families: anchored top-down from the target latitudes, all
    # sharing one repeat period (Lee 2020 Eq. 33) under exact secular J2.
    # Altitude is an OUTPUT of this step, not an input.
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
    families = plan["families"]
    N_D         = int(plan["N_D"])
    L           = int(plan["L"])
    dt_s_actual = float(plan["dt_s"])
    T_r_s       = float(plan["T_r_s"])
    horizon_s   = (L - 1) * dt_s_actual
    anchor      = plan["anchor"]
    actual_alt_km = float(anchor["alt_km"])   # anchor altitude: labels the results folder

    print(
        f"\nSeed families (N_D={N_D}, altitude bounds {plan['alt_bounds_km'][0]:.0f}-{plan['alt_bounds_km'][1]:.0f} km "
        f"= shell + R_max, target latitudes {plan['lat_min_deg']:.2f}-{plan['lat_max_deg']:.2f} deg):"
    )
    gen = plan["generation"]
    print(
        f"  target latitude span: {gen['target_lat_span_deg']:.2f} deg "
        f"({gen['target_lat_min_deg']:.2f} to {gen['target_lat_max_deg']:.2f})   "
        f"anchor candidates: {gen['n_anchor_candidates']}   "
        f"unique altitude/inclination combos: generated {gen['n_unique_alt_inc_combos_all_candidates']} "
        f"(chosen ladder {gen['n_unique_alt_inc_combos_chosen_ladder']}), kept {gen['n_unique_alt_inc_combos_kept']}, "
        f"dropped from chosen ladder {gen['n_combos_dropped']}"
    )
    for f in families:
        red = f" [reducible -> {f['reduced_ratio']}]" if f["reducible"] else ""
        print(
            f"  {'anchor' if f['is_anchor'] else '      '} {f['N_P']:3d}:{f['N_D']}  i={f['inc_deg']:7.3f} deg  "
            f"alt={f['alt_km']:7.2f} km  reach={f['reach_deg']:.2f} deg  floor={f['inc_floor_deg']:.2f} deg{red}"
        )
    print(
        f"Shared repeat period T_r={T_r_s:.3f} s  L={L} steps  dt={dt_s_actual:.6f} s "
        f"(nominal {DT_S:.1f} s)  horizon={horizon_s:.1f} s ({horizon_s / 3600:.3f} h)"
    )

    sim = SimConfig(
        horizon_s=horizon_s,
        dt_s=dt_s_actual,
        analysis_matrix_dtype="uint8",
        use_j2=True,
        use_drag=False,
    )

    # ------------------------------------------------------------------
    # Seed pool (family x RAAN), simulation, filter (see Lee/seed_pipeline.py)
    # ------------------------------------------------------------------
    r_required = int(np.ceil(SALVO_SIZE / N_INTERCEPTORS_PER_SAT))
    seed_meta, sub_layers = build_seed_pool(plan, RAAN_STEP_DEG, earth)
    print(f"\nSeed grid: {len(seed_meta)} seeds  (RAAN step {RAAN_STEP_DEG:.2f} deg within each family's window)")
    for f in families:
        print(f"  family {f['N_P']}:{f['N_D']}: {f['n_raan_offsets']} RAAN offsets in [0, {f['raan_interval_deg']:.3f}) deg")

    v0_list = simulate_seed_pool(
        sub_layers, sim=sim, cov=cov, earth=earth, shell=shell,
        propagator=propagator, workers=SEED_SCREENING_WORKERS,
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
        f"\nCoverage requirement: salvo_size={SALVO_SIZE}, "
        f"interceptors_per_sat={N_INTERCEPTORS_PER_SAT}, "
        f"r_required={r_required} satellite(s) in view per time step"
    )

    import datetime
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_settings = settings_from_config({
        "scenario": SCENARIO, "is_principal_set": IS_PRINCIPAL_SET, "target_mode": TARGET_MODE,
        "target_country": TARGET_COUNTRY if TARGET_MODE == "country" else None,
        "target_points": [list(p) for p in TARGET_POINTS] if TARGET_MODE != "country" else None,
        "target_shell_n_points": TARGET_SHELL_N_POINTS, "salvo_size": SALVO_SIZE,
        "n_interceptors_per_sat": N_INTERCEPTORS_PER_SAT, "v_bo_km_s": V_BO_KM_S, "a_g": A_G,
        "detection_time_s": DETECTION_TIME_S, "decision_time_s": DECISION_TIME_S,
        "target_missile_burnout_time_s": TARGET_MISSILE_BURNOUT_TIME_S, "intercept_alt_km": INTERCEPT_ALT_KM,
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
    print("Solving SCLP  (Ctrl+C to interrupt and return best incumbent)")
    print("=" * 60)

    result = solve_sclp(
        param_V=param_V,
        param_r=param_r,
        n_targets=n_targets,
        apply_rgt_symmetry_break=False,
        mip_gap=MIP_GAP,
        time_limit_s=TIME_LIMIT_S,
        log_file=log_file,
        verbose=True,
        stop_at_gap=STOP_AT_GAP,
    )

    print("\n" + "=" * 60)
    print(f"Status  : {result.status_str}")
    print(f"Selected: {result.n_satellites} satellite(s)")
    print(f"Cost    : {result.obj_val:.1f}")
    _gap_pct = result.mip_gap * 100
    print(f"MIP gap : {_gap_pct:.4f}%" if np.isfinite(_gap_pct) else "MIP gap : N/A (no lower bound established)")
    print(f"Modeling: {result.time_modeling_s:.2f} s   Solve: {result.time_optimization_s:.2f} s")
    print("=" * 60)

    orb_elems = build_orb_elems_table(sub_layers, L, ncols)

    print("\nOrbital characteristics of selected constellation:")
    print_orb_table(orb_elems, result.selected_slots)

    # Satellites chosen per family
    per_family_counts = family_satellite_counts(result.selected_slots, ncols, seed_meta, families)
    print("Satellites per family: " + ", ".join(f"{k}: {v}" for k, v in per_family_counts.items()))

    csv_path = RESULTS_DIR / f"{run_id}_result.csv"

    seed_cfg = seed_run_config(
        plan, pool, per_family_counts,
        raan_step_deg=RAAN_STEP_DEG,
        margin_multiplier=MARGIN_MULTIPLIER,
        trim_margin_multiplier=TRIM_MARGIN_MULTIPLIER,
        max_seeds=MAX_SEEDS_FOR_MILP,
    )

    run_config = {
        # --- Run metadata ---
        "run_name":                       RUN_NAME,
        "run_id":                         run_id,
        "run_settings_key":               settings_key(run_settings),
        "is_principal_set":               IS_PRINCIPAL_SET,
        # --- Target ---
        "target_mode":                    TARGET_MODE,
        "target_country":                 TARGET_COUNTRY if TARGET_MODE == "country" else None,
        "target_points":                  [list(p) for p in TARGET_POINTS] if TARGET_MODE != "country" else None,
        "target_shell_n_points":          TARGET_SHELL_N_POINTS if TARGET_MODE == "country" else None,
        "n_targets":                      int(n_targets),
        # --- Engagement physics ---
        "dt_s":                           dt_s_actual,
        "dt_s_nominal":                   DT_S,
        "detection_time_s":               DETECTION_TIME_S,
        "decision_time_s":                DECISION_TIME_S,
        "target_missile_burnout_time_s":  TARGET_MISSILE_BURNOUT_TIME_S,
        "interceptor_engagement_time_s":  float(cov.interceptor_engagement_time_s),
        "v_bo_km_s":                      V_BO_KM_S,
        "a_g":                            A_G,
        "intercept_alt_km":               INTERCEPT_ALT_KM,
        "min_elev_deg":                   MIN_ELEV_DEG,
        "r_max_km":                       float(cov.max_range_km),
        # --- Orbit model, seed generation, seed filter (shared block) ---
        **seed_cfg,
        # --- Intercept salvo ---
        "salvo_size":                     SALVO_SIZE,
        "n_interceptors_per_sat":         N_INTERCEPTORS_PER_SAT,
        "r_required":                     int(r_required),
        # --- Solver ---
        "mip_gap_target":                 MIP_GAP,
        "time_limit_s":                   None if TIME_LIMIT_S == float("inf") else TIME_LIMIT_S,
        "stop_at_gap":                    STOP_AT_GAP,
        # --- Result summary ---
        "timestamp_utc":                  datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "result_status":                  result.status_str,
        "mip_gap_achieved":               None if not np.isfinite(result.mip_gap) else float(result.mip_gap),
        "n_satellites_selected":          int(result.n_satellites),
        "time_modeling_s":                float(result.time_modeling_s),
        "time_optimization_s":            float(result.time_optimization_s),
    }

    targets = [
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
        targets=targets,
        countries=countries,
        propagator=propagator,
        bc_kg_m2=bc_kg_m2,
        run_config=run_config,
    )


if __name__ == "__main__":
    main()
