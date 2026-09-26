"""
lee_test2.py

Minimal SCLP constellation design workflow.

Builds the RGT visibility matrix, solves the SCLP directly in Python using
Gurobi, and saves the selected constellation's orbital characteristics to CSV.

Usage:
    python scripts/lee_test2.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from sbi_coverage.core.config import CoverageConfig, EarthConstants, SimConfig
from sbi_coverage.core.shell import Shell
from sbi_coverage.core.simulate import run_simulation
from sbi_coverage.core.target_points import load_target_points_csv, smart_inclination_bounds
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

# Inclination sweep configuration.  Country mode uses this fixed tuple as-is;
# prespecified mode instead computes a geometry-aware floor from the actual
# target points -- see smart_inclination_bounds() call in main().
INCLINATION_SWEEP_BOUNDS_DEG: tuple[float, float] = (0, 90.0)
INCLINATION_SCREEN_STEP_DEG: float = 1

# Altitude family configuration
ALTITUDE_MAX_REPEAT_DAYS: int = 2

# Required target altitude (km); snapped to the nearest RGT orbit
# (N_D ≤ ALTITUDE_MAX_REPEAT_DAYS). Both this value and the actual snapped
# altitude are saved to the output JSON as target_altitude_km/actual_alt_km.
TARGET_ALTITUDE_KM: float = 400.0

# Seed simulation configuration
# Number of RAAN phase offsets sampled per inclination.  Values are evenly
# spaced in [0°, 360°/N_P) — the one truly independent plane-spacing interval
# for the chosen RGT orbit.  Offsets outside this range are periodic repeats
# of the same N_P orbital planes and add no new coverage geometry.
# Recommended: 3–5 for country targets; 1 is sufficient for point targets.
N_RAAN_OFFSETS: int = 5
SEED_SCREENING_WORKERS: int = 8
# Number of seed orbits passed to the MILP after greedy marginal-coverage
# ranking.  All seeds are simulated; only the best N are given to Gurobi.
N_SEEDS_FOR_MILP: int = 10

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

# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------
MIP_GAP: float = 0.001
TIME_LIMIT_S: float = 45
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


def _result_subdir(actual_alt_km: float) -> Path:
    """Build and create the hierarchical results subfolder for this run.

    Structure (country mode):
        results/<country>/<burnout-vel>/<intercept-window>/<intercept-alt>/
                <max-accel>/<orbit-alt>/<interceptors-per-sat>/<salvo-size>/

    Structure (prespecified mode):
        results/prespecified/<target-name>/<burnout-vel>/...
    """
    mode = str(TARGET_MODE).strip().lower()

    if mode == "country":
        countries = _normalize_country_input(TARGET_COUNTRY)
        target_slug = "-".join(c.lower().replace(" ", "-") for c in countries)
        top = RESULTS_DIR / target_slug
    else:
        name = TARGET_POINTS[0][0] if TARGET_POINTS else "target"
        name_slug = (
            name.strip().lower()
            .replace(" ", "-")
            .replace(":", "")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "-")
        )[:20]  # truncated -- the full results hierarchy underneath this name
        # (burnout-vel/.../salvo-size/<timestamp>_result.csv) adds ~180-200
        # more characters, so an untruncated real-world target name (this repo's
        # own absolute path already accounts for ~50 of Windows' 260-char
        # budget) can exceed the limit and fail with a plain FileNotFoundError.
        # batch_worker.py's equivalent S3-prefix slug uses [:40] since S3 keys
        # have no such limit; this is deliberately tighter for the local path.
        top = RESULTS_DIR / "prespecified" / name_slug

    path = (
        top
        / f"burnout-vel-{V_BO_KM_S:.1f}km-s"
        # Label kept as "intercept-window" (not renamed to "burnout-time") so
        # existing path-parsing regexes (cloud_plot_best_results.py,
        # plot_best_results_by_country.py) keep matching -- only the source
        # of the number changed, from T_WINDOW_S to
        # TARGET_MISSILE_BURNOUT_TIME_S (numerically identical at defaults).
        / f"intercept-window-{TARGET_MISSILE_BURNOUT_TIME_S:.0f}s"
        / f"intercept-alt-{INTERCEPT_ALT_KM:.0f}km"
        / f"max-accel-{A_G:.1f}g"
        / f"orbit-alt-{round(actual_alt_km)}km"
        / f"interceptors-per-sat-{N_INTERCEPTORS_PER_SAT}"
        / f"salvo-size-{SALVO_SIZE}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _greedy_select_seeds(
    v0_list: list[np.ndarray],
    r_required: int,
    n_max: int,
    *,
    inc_deg_list: list[float] | None = None,
) -> list[int]:
    """Greedy marginal-coverage seed selection.

    Two-phase selection:

    Phase 1 — Inclination guarantee:
        Every discrete inclination value present in inc_deg_list is
        guaranteed exactly one representative. Among all RAAN candidates for
        a given inclination the one with the highest marginal-coverage score
        is chosen (unique access against the current deficit; total access
        once deficit is met).  Inclinations are processed in greedy order —
        whichever missing inclination's best candidate contributes the most
        is selected first.

    Phase 2 — Greedy fill:
        Remaining n_max slots are filled by the standard greedy marginal-
        coverage criterion, allowing inclinations to repeat at different RAANs.

    Returns a list of indices (into v0_list) in selection order.
    """
    n_seeds   = len(v0_list)
    access    = np.stack([v0.sum(axis=0) for v0 in v0_list]).astype(np.float64)
    deficit   = np.full(access.shape[1], float(r_required))
    selected: list[int] = []
    remaining = list(range(n_seeds))

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
            # Among all pending inclinations, find the (inc, candidate) pair
            # that contributes the most marginal coverage right now.
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
                break  # no live candidates remain for any pending inclination

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

    # Snap to the nearest N_D ≤ ALTITUDE_MAX_REPEAT_DAYS RGT orbit at the
    # requested target altitude.
    candidates = candidate_rgt_ratios(
        h_star_km=TARGET_ALTITUDE_KM,
        earth=earth,
        max_repeat_days=ALTITUDE_MAX_REPEAT_DAYS,
        dt_s=DT_S,
    )
    if not candidates:
        raise ValueError(
            f"No RGT orbit found near TARGET_ALTITUDE_KM={TARGET_ALTITUDE_KM} km "
            f"with N_D ≤ {ALTITUDE_MAX_REPEAT_DAYS}.  Raise ALTITUDE_MAX_REPEAT_DAYS "
            "or choose a different altitude."
        )
    best = candidates[0]
    N_P             = int(best["N_P"])
    N_D             = int(best["N_D"])
    L               = int(best["L"])
    altitude_ratios = [best]
    altitude_ref_km = float(TARGET_ALTITUDE_KM)
    print(
        f"\nTarget altitude requested: {TARGET_ALTITUDE_KM:.1f} km  →  "
        f"nearest RGT orbit: {best['N_P']}:{best['N_D']}  "
        f"actual_alt={float(best['alt_km']):.2f} km  "
        f"(snap error: {float(best['alt_error_km']):.2f} km)"
    )

    shared_worst_error = max(abs(float(item["alt_km"]) - altitude_ref_km) for item in altitude_ratios)
    actual_alt_km      = float(altitude_ratios[0]["alt_km"])

    # Keplerian T_r/L: the APC circulant property is exact for all
    # inclinations under Keplerian dynamics -- coverage output is guaranteed
    # 100% by construction.
    dt_s_actual = float(altitude_ratios[0]["T_r_s"]) / L

    horizon_s = (L - 1) * dt_s_actual
    print(
        f"\nShared repeat family: N_D={N_D}  L={L} steps  "
        f"(worst altitude mapping error across families = {shared_worst_error:.3f} km)"
    )
    print(f"dt_s: {DT_S:.4f} s (nominal) → {dt_s_actual:.6f} s (Keplerian T_r/L)")

    sim = SimConfig(
        horizon_s=horizon_s,
        dt_s=dt_s_actual,
        analysis_matrix_dtype="uint8",
        use_j2=False,
        use_drag=False,
    )
    print(f"Simulation horizon: {horizon_s:.1f} s  ({horizon_s/3600:.3f} h)  -> L={L}")

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
    # Enumerate all (inclination, RAAN) seed combinations — flat grid
    # ------------------------------------------------------------------
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed

    fam  = altitude_ratios[0]
    N_P  = int(fam["N_P"])

    # Country mode uses the fixed INCLINATION_SWEEP_BOUNDS_DEG tuple as-is;
    # prespecified mode computes a geometry-aware floor from the actual
    # target points (narrows the search instead of always sweeping 0-90).
    if str(TARGET_MODE).strip().lower() == "prespecified":
        inc_bounds = smart_inclination_bounds(TARGET_POINTS)
    else:
        inc_bounds = INCLINATION_SWEEP_BOUNDS_DEG
    inc_lo, inc_hi = inc_bounds
    inc_grid  = np.arange(
        float(inc_lo),
        float(inc_hi) + float(INCLINATION_SCREEN_STEP_DEG) * 0.5,
        float(INCLINATION_SCREEN_STEP_DEG),
    )

    raan_interval = 360.0 / N_P
    raan_grid     = np.linspace(0.0, raan_interval, N_RAAN_OFFSETS, endpoint=False)

    seed_pairs    = [(float(inc), float(raan)) for inc in inc_grid for raan in raan_grid]
    n_seeds       = len(seed_pairs)
    n_seeds_total = n_seeds
    print(
        f"\nSeed grid: {len(inc_grid)} inclinations × {N_RAAN_OFFSETS} RAANs = {n_seeds} seeds"
    )
    print(f"  inclinations : {float(inc_lo):.1f}°–{float(inc_hi):.1f}° step {float(INCLINATION_SCREEN_STEP_DEG):.1f}°")
    print(f"  RAANs        : {N_RAAN_OFFSETS} offsets in [0°, {raan_interval:.3f}°)  "
          f"(360° / N_P={N_P})  step={raan_interval / N_RAAN_OFFSETS:.3f}°")

    # ------------------------------------------------------------------
    # Build sub-layers and simulate all seeds in parallel
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
            propagator=propagator,
        )
        return idx, (analysis.counts > 0).astype(np.float64)

    print(f"\nSimulating {n_seeds} seeds  (workers={SEED_SCREENING_WORKERS}) ...")
    t_sim0         = time.perf_counter()
    v0_list: list[np.ndarray] = [None] * n_seeds  # type: ignore[list-item]
    progress_every = max(1, n_seeds // 20)

    with ThreadPoolExecutor(max_workers=SEED_SCREENING_WORKERS) as pool:
        futs = {pool.submit(_simulate_seed, i): i for i in range(n_seeds)}
        done = 0
        for fut in as_completed(futs):
            idx, v0 = fut.result()
            v0_list[idx] = v0
            done += 1
            if done % progress_every == 0 or done == n_seeds:
                print(f"  {done}/{n_seeds} seeds simulated ...", flush=True)

    print(f"Total seed simulation time: {time.perf_counter() - t_sim0:.2f} s")

    # ------------------------------------------------------------------
    # Seed selection: filter → greedy rank → pass best N to MILP
    # ------------------------------------------------------------------
    r_required = int(np.ceil(SALVO_SIZE / N_INTERCEPTORS_PER_SAT))

    # Step 1 — drop seeds with zero target access (free variable reduction)
    live_mask = [bool(v0.any()) for v0 in v0_list]
    n_dead    = live_mask.count(False)
    if n_dead:
        print(f"\nPre-filter: removed {n_dead}/{n_seeds} seeds with zero target access.")
        v0_list    = [v for v, keep in zip(v0_list,    live_mask) if keep]
        sub_layers = [s for s, keep in zip(sub_layers, live_mask) if keep]
        seed_pairs = [p for p, keep in zip(seed_pairs, live_mask) if keep]
        n_seeds    = len(v0_list)
    n_seeds_live = n_seeds

    # Step 2 — greedy marginal-coverage ranking
    inc_deg_list = [float(p[0]) for p in seed_pairs]
    greedy_idx   = _greedy_select_seeds(
        v0_list, r_required, N_SEEDS_FOR_MILP,
        inc_deg_list=inc_deg_list,
    )
    n_selected = len(greedy_idx)
    print(f"\nGreedy seed selection: {n_selected} of {n_seeds} live seeds → MILP")

    # Coverage proxy: sum of per-target access over selected seeds.
    # Due to the circulant property this equals simultaneous views at any t.
    access_sum = sum(v0_list[i].sum(axis=0) for i in greedy_idx)
    proxy_min  = float(access_sum.min())
    proxy_mean = float(access_sum.mean())
    feasible_proxy = proxy_min >= r_required
    print(
        f"  Coverage proxy (all slots of selected seeds): "
        f"min={proxy_min:.1f}  mean={proxy_mean:.1f}  "
        f"requirement={r_required}  "
        + ("✓ feasible" if feasible_proxy else "⚠ may be infeasible — consider raising N_SEEDS_FOR_MILP")
    )
    # Tag seeds by selection phase: inc-guaranteed, or greedy fill.
    inc_guaranteed_set: set[int] = set()
    if inc_deg_list:
        _seen_incs: set[float] = set()
        for idx in greedy_idx:
            _inc_val = inc_deg_list[idx]
            if _inc_val not in _seen_incs:
                inc_guaranteed_set.add(idx)
                _seen_incs.add(_inc_val)
    for rank, idx in enumerate(greedy_idx):
        inc, raan = seed_pairs[idx]
        tag = "  [inc. guaranteed]" if idx in inc_guaranteed_set else ""
        print(f"  {rank + 1:2d}: i={inc:5.1f}°  RAAN={raan:6.1f}°  "
              f"access/target={v0_list[idx].sum(axis=0).mean():.1f}{tag}")

    v0_list    = [v0_list[i]    for i in greedy_idx]
    sub_layers = [sub_layers[i] for i in greedy_idx]
    n_seeds    = n_selected

    # ------------------------------------------------------------------
    # Build constraint matrix and solve SCLP
    # ------------------------------------------------------------------
    param_V = build_param_V(v0_list)
    param_V_summary(param_V, n_seeds=n_seeds)

    param_r = r_required * np.ones((L, n_targets), dtype=np.float64)
    print(
        f"\nCoverage requirement: salvo_size={SALVO_SIZE}, "
        f"interceptors_per_sat={N_INTERCEPTORS_PER_SAT}, "
        f"r_required={r_required} satellite(s) in view per time step"
    )

    import datetime
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    log_file = None
    if LOG_DIR is not None:
        log_file = LOG_DIR / f"SCLP_{ts}.txt"

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
    )

    print("\n" + "=" * 60)
    print(f"Status  : {result.status_str}")
    print(f"Selected: {result.n_satellites} satellite(s)")
    print(f"Cost    : {result.obj_val:.1f}")
    _gap_pct = result.mip_gap * 100
    print(f"MIP gap : {_gap_pct:.4f}%" if np.isfinite(_gap_pct) else "MIP gap : N/A (no lower bound established)")
    print(f"Modeling: {result.time_modeling_s:.2f} s   Solve: {result.time_optimization_s:.2f} s")
    print("=" * 60)

    orb_elems = build_orb_elems_table(sub_layers, L)

    print("\nOrbital characteristics of selected constellation:")
    print_orb_table(orb_elems, result.selected_slots)

    out_dir  = _result_subdir(actual_alt_km)
    csv_path = out_dir / f"{ts}_result.csv"

    run_config = {
        # --- Run metadata ---
        "run_name":                       RUN_NAME,
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
        # --- Orbit ---
        "target_altitude_km":             TARGET_ALTITUDE_KM,
        "altitude_max_repeat_days":       ALTITUDE_MAX_REPEAT_DAYS,
        "actual_alt_km":                  float(actual_alt_km),
        "rgt_ratio":                      f"{N_P}:{N_D}",
        "L":                              int(L),
        "horizon_s":                      float(horizon_s),
        # --- Seed grid ---
        "inclination_sweep_bounds_deg":   list(inc_bounds),
        "inclination_screen_step_deg":    INCLINATION_SCREEN_STEP_DEG,
        "n_raan_offsets":                  N_RAAN_OFFSETS,
        "raan_interval_deg":              float(raan_interval),
        "raan_step_deg":                  float(raan_interval / N_RAAN_OFFSETS),
        "n_seeds_simulated":              int(n_seeds_total),
        "n_seeds_live":                   int(n_seeds_live),
        "n_seeds_for_milp":               N_SEEDS_FOR_MILP,
        "seeds_selected": [
            {
                "rank":     rank + 1,
                "inc_deg":  float(seed_pairs[i][0]),
                "raan_deg": float(seed_pairs[i][1]),
            }
            for rank, i in enumerate(greedy_idx)
        ],
        # --- Intercept salvo ---
        "salvo_size":                     SALVO_SIZE,
        "n_interceptors_per_sat":         N_INTERCEPTORS_PER_SAT,
        "r_required":                     int(r_required),
        # --- Solver ---
        "mip_gap_target":                 MIP_GAP,
        "time_limit_s":                   None if TIME_LIMIT_S == float("inf") else TIME_LIMIT_S,
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
