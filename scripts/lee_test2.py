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
# Problem definition
# ---------------------------------------------------------------------------

TARGET_MODE: str = "country"  # "prespecified" or "country"

# For use in prespecified mode
TARGET_POINTS: list[tuple[str, float, float]] = [
    ("NK: Yongbyon (nuclear)", 39.79, 125.66),
]

# For use in country mode
TARGET_COUNTRY: str | list[str] | tuple[str, ...] = "Brazil"
TARGET_SHELL_N_POINTS: int = 15000

# Time discretization
DT_S: float = 120.0

# Interceptor / engagement geometry inputs
T_WINDOW_S: float = 170.0
V_BO_KM_S: float = 4.0
A_G: float = 10.0
INTERCEPT_ALT_KM: float = 200.0
MIN_ELEV_DEG: float = 0.0

# Inclination sweep configuration
INCLINATION_SWEEP_BOUNDS_DEG: tuple[float, float] = (0, 90.0)
INCLINATION_SCREEN_STEP_DEG: float = 1

# Altitude family configuration
# ALTITUDE_SEARCH_MIN_KM: float = 200.0
# ALTITUDE_SEARCH_MAX_KM: float = 700.0
# ALTITUDE_MAX_FAMILIES: int = 1
# ALTITUDE_MIN_SEPARATION_KM: float = 25.0
ALTITUDE_MAX_REPEAT_DAYS: int = 2

# Set to a specific altitude (km) to bypass the geometric h* search and snap to
# the nearest RGT orbit at that altitude (N_D ≤ ALTITUDE_MAX_REPEAT_DAYS).
# Leave as None to use the geometrically optimal h*.
ALTITUDE_OVERRIDE_KM: float | None = 400

# J2 perturbation model.  When True, the propagator applies secular J2 rates
# (RAAN drift, argp drift) and the RGT repeat period is computed at the J2-
# correct altitude for the reference inclination.  When False, Keplerian
# dynamics are used and the APC circulant property is exact for all
# inclinations — coverage output is guaranteed 100% by construction.
USE_J2: bool = False

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
# When True, three polar seeds are reserved before the greedy runs — one each
# at inc = 80°, 85°, and 90°.  For each inclination the seed with the highest
# total target access (best RAAN) is chosen from the simulated pool.  Seeds
# that are missing from the grid (e.g. because INCLINATION_SCREEN_STEP_DEG is
# coarser than 5°) are silently skipped.  Each reservation counts against the
# N_SEEDS_FOR_MILP cap.
RESERVE_POLAR_ORBITS: bool = False

# ---------------------------------------------------------------------------
# Intercept salvo parameters
# ---------------------------------------------------------------------------
SALVO_SIZE: int = 2
N_INTERCEPTORS_PER_SAT: int = 1

# ---------------------------------------------------------------------------
# Solver options
# ---------------------------------------------------------------------------
MIP_GAP: float = 0.001
TIME_LIMIT_S: float = 45
# MIPFocus=1: prioritise finding feasible incumbents fast (best for sweep / large
# grids where the first solution is hard to find).  Switch to 2 when you want
# Gurobi to focus on proving optimality once a good incumbent already exists.
MIP_FOCUS: int = 1
# Fraction of B&B time Gurobi spends on primal heuristics.  Default is 0.05;
# 0.2–0.3 helps significantly when the first incumbent is the bottleneck.
HEURISTICS_FRAC: float = 0.2
# Additional Gurobi parameter overrides.  Applied after MIP_FOCUS and
# HEURISTICS_FRAC, so any key here takes full precedence.  Set a key to
# None to let Gurobi use its own default for that parameter.
# Note: CoverCuts=2, Cuts=2, and LPWarmStart=1 are already hardcoded in
# the solver and do not need to be repeated here.
SOLVER_PARAMS: dict = {
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


def _enumerate_exact_rgt_altitudes_for_repeat_days(
    *,
    N_D: int,
    min_alt_km: float,
    max_alt_km: float,
    dt_s: float,
    earth: EarthConstants,
    max_n_p: int = 512,
) -> list[dict[str, float | int]]:
    out: list[dict[str, float | int]] = []
    for N_P in range(1, max_n_p + 1):
        if np.gcd(N_P, N_D) != 1:
            continue

        T_sid = (2.0 * np.pi) / earth.omega_earth_rad_s
        T_r_s = N_D * T_sid
        T_S_s = T_r_s / N_P
        a_km = (earth.mu_km3_s2 * (T_S_s / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0)
        alt_km = float(a_km - earth.r_eq_km)
        if alt_km < min_alt_km or alt_km > max_alt_km:
            continue
        out.append(
            {
                "N_P": int(N_P),
                "N_D": int(N_D),
                "alt_km": float(alt_km),
                "alt_error_km": 0.0,
                "T_r_s": float(T_r_s),
                "T_S_s": float(T_S_s),
                "L": int(round(T_r_s / float(dt_s))),
            }
        )
    out.sort(key=lambda item: float(item["alt_km"]))
    return out


def _select_band_search_altitude_families(
    *,
    h_star_km: float,
    cov: CoverageConfig,
    earth: EarthConstants,
    dt_s: float,
) -> tuple[int, int, list[dict[str, float | int]]]:
    best_choice: tuple[float, int, int, list[dict[str, float | int]]] | None = None

    for N_D in range(1, ALTITUDE_MAX_REPEAT_DAYS + 1):
        families = _enumerate_exact_rgt_altitudes_for_repeat_days(
            N_D=N_D,
            min_alt_km=max(float(cov.intercept_alt_km), float(ALTITUDE_SEARCH_MIN_KM)),
            max_alt_km=float(ALTITUDE_SEARCH_MAX_KM),
            dt_s=dt_s,
            earth=earth,
        )
        if not families:
            continue

        nearest_error = min(abs(float(item["alt_km"]) - float(h_star_km)) for item in families)
        candidate = (nearest_error, -len(families), N_D, families)
        if best_choice is None or candidate[:3] < best_choice[:3]:
            best_choice = candidate

    if best_choice is None:
        raise ValueError(
            "band_search found no exact RGT altitude families in the requested altitude band. "
            "Adjust ALTITUDE_SEARCH_MIN_KM / ALTITUDE_SEARCH_MAX_KM or ALTITUDE_MAX_REPEAT_DAYS."
        )

    N_D = int(best_choice[2])
    families = list(best_choice[3])
    families.sort(key=lambda item: abs(float(item["alt_km"]) - float(h_star_km)))

    selected: list[dict[str, float | int]] = []
    for family in families:
        alt_km = float(family["alt_km"])
        if any(abs(alt_km - float(existing["alt_km"])) < ALTITUDE_MIN_SEPARATION_KM for existing in selected):
            continue
        selected.append(family)
        if len(selected) >= ALTITUDE_MAX_FAMILIES:
            break

    if not selected:
        raise ValueError("band_search did not retain any altitude families after separation filtering.")

    L = int(selected[0]["L"])
    return N_D, L, selected


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
            name.lower()
            .replace(" ", "-")
            .replace(":", "")
            .replace("(", "")
            .replace(")", "")
        )
        top = RESULTS_DIR / "prespecified" / name_slug

    path = (
        top
        / f"burnout-vel-{V_BO_KM_S:.1f}km-s"
        / f"intercept-window-{T_WINDOW_S:.0f}s"
        / f"intercept-alt-{INTERCEPT_ALT_KM:.0f}km"
        / f"max-accel-{A_G:.1f}g"
        / f"orbit-alt-{round(actual_alt_km)}km"
        / f"interceptors-per-sat-{N_INTERCEPTORS_PER_SAT}"
        / f"salvo-size-{SALVO_SIZE}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


_POLAR_RESERVATION_INCLINATIONS: tuple[float, ...] = (80.0, 85.0, 90.0)


def _greedy_select_seeds(
    v0_list: list[np.ndarray],
    r_required: int,
    n_max: int,
    *,
    inc_deg_list: list[float] | None = None,
    reserve_polar_orbits: bool = False,
) -> list[int]:
    """Greedy marginal-coverage seed selection.

    Three-phase selection:

    Phase 0 — Polar reservation (if reserve_polar_orbits is True):
        Pre-select the best-RAAN seed for each inclination in
        _POLAR_RESERVATION_INCLINATIONS before any other selection runs.

    Phase 1 — Inclination guarantee:
        Every discrete inclination value present in inc_deg_list that was not
        already covered by Phase 0 is guaranteed exactly one representative.
        Among all RAAN candidates for a given inclination the one with the
        highest marginal-coverage score is chosen (unique access against the
        current deficit; total access once deficit is met).  Inclinations are
        processed in greedy order — whichever missing inclination's best
        candidate contributes the most is selected first.

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
    countries = _normalize_country_input(TARGET_COUNTRY)
    if not countries:
        raise ValueError("TARGET_COUNTRY must contain at least one country.")

    earth = EarthConstants()
    propagator = "Nominal_Propagator"

    cov = CoverageConfig(
        earth=earth,
        T_window_s=T_WINDOW_S,
        v_bo_km_s=V_BO_KM_S,
        a_g=A_G,
        intercept_alt_km=INTERCEPT_ALT_KM,
        min_elev_deg=MIN_ELEV_DEG,
    )

    print(f"Computed R_max: {cov.max_range_km:.2f} km")

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
    print(f"Optimal satellite altitude (h*): {h_star:.2f} km")
    print(f"  rho_eff: {angles['rho_eff_deg']:.4f} deg  (binding: {angles['binding']})")

    if ALTITUDE_OVERRIDE_KM is not None:
        # Altitude override: snap to the nearest N_D ≤ ALTITUDE_MAX_REPEAT_DAYS
        # RGT orbit at the requested altitude instead of the geometric h*.
        candidates = candidate_rgt_ratios(
            h_star_km=ALTITUDE_OVERRIDE_KM,
            earth=earth,
            max_repeat_days=ALTITUDE_MAX_REPEAT_DAYS,
            dt_s=DT_S,
        )
        if not candidates:
            raise ValueError(
                f"No RGT orbit found near ALTITUDE_OVERRIDE_KM={ALTITUDE_OVERRIDE_KM} km "
                f"with N_D ≤ {ALTITUDE_MAX_REPEAT_DAYS}.  Raise ALTITUDE_MAX_REPEAT_DAYS "
                "or choose a different altitude."
            )
        best = candidates[0]
        N_P             = int(best["N_P"])
        N_D             = int(best["N_D"])
        L               = int(best["L"])
        altitude_ratios = [best]
        altitude_ref_km = float(ALTITUDE_OVERRIDE_KM)
        print(
            f"\nAltitude OVERRIDE requested: {ALTITUDE_OVERRIDE_KM:.1f} km  →  "
            f"nearest RGT orbit: {best['N_P']}:{best['N_D']}  "
            f"actual_alt={float(best['alt_km']):.2f} km  "
            f"(snap error: {float(best['alt_error_km']):.2f} km)"
        )
    else:
        N_D, L, altitude_ratios = _select_band_search_altitude_families(
            h_star_km=h_star,
            cov=cov,
            earth=earth,
            dt_s=DT_S,
        )
        N_P             = int(altitude_ratios[0]["N_P"])
        altitude_ref_km = float(h_star)
        print("\nAltitude families:")
        for ratio in altitude_ratios:
            print(f"  exact_RGT={ratio['N_P']}:{ratio['N_D']}  actual_alt={float(ratio['alt_km']):.2f} km")

    shared_worst_error = max(abs(float(item["alt_km"]) - altitude_ref_km) for item in altitude_ratios)
    actual_alt_km      = float(altitude_ratios[0]["alt_km"])

    # J2-correct T_r and L: use the exact RGT repeat period at a reference
    # inclination (midpoint of the sweep).  With J2-corrected a_km (from
    # a_km_j2_rgt), the orbit repeats precisely after T_r_j2 seconds, making
    # the circulant V matrix exact for ALL entries including wrap-arounds.
    # This eliminates the residual ~0.04 % coverage gap caused by the
    # Keplerian T_r ≠ J2-correct T_r mismatch.
    if USE_J2:
        from sbi_coverage.Lee.rgt_slots import a_km_j2_rgt as _a_km_j2_rgt
        _inc_ref     = float(np.mean(INCLINATION_SWEEP_BOUNDS_DEG))
        _a_j2        = _a_km_j2_rgt(N_P, N_D, _inc_ref, earth)
        _n_j2        = float(np.sqrt(earth.mu_km3_s2 / _a_j2**3))
        _fac_j2      = 1.5 * earth.j2 * (earth.r_eq_km / _a_j2)**2 * _n_j2
        _od_j2       = 0.5 * _fac_j2 * (5.0 * np.cos(np.deg2rad(_inc_ref))**2 - 1.0)
        _T_r_j2      = N_P * 2.0 * np.pi / (_n_j2 + _od_j2)
        L            = round(_T_r_j2 / DT_S)
        dt_s_actual  = _T_r_j2 / L
    else:
        dt_s_actual  = float(altitude_ratios[0]["T_r_s"]) / L

    horizon_s = (L - 1) * dt_s_actual
    print(
        f"\nShared repeat family: N_D={N_D}  L={L} steps  "
        f"(worst altitude mapping error across families = {shared_worst_error:.3f} km)"
    )
    _tr_label = "J2-corrected T_r/L" if USE_J2 else "Keplerian T_r/L"
    print(f"dt_s: {DT_S:.4f} s (nominal) → {dt_s_actual:.6f} s ({_tr_label})")

    sim = SimConfig(
        horizon_s=horizon_s,
        dt_s=dt_s_actual,
        analysis_matrix_dtype="uint8",
        use_j2=USE_J2,
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

    inc_lo, inc_hi = INCLINATION_SWEEP_BOUNDS_DEG
    inc_grid  = np.arange(
        float(inc_lo),
        float(inc_hi) + float(INCLINATION_SCREEN_STEP_DEG) * 0.5,
        float(INCLINATION_SCREEN_STEP_DEG),
    )

    # When polar reservation is active, ensure every inclination in
    # _POLAR_RESERVATION_INCLINATIONS (80°, 85°, 90°) is simulated,
    # regardless of INCLINATION_SCREEN_STEP_DEG.  np.unique deduplicates any
    # values already present in the main grid and keeps everything sorted.
    if RESERVE_POLAR_ORBITS:
        polar_extra = np.array(list(_POLAR_RESERVATION_INCLINATIONS), dtype=np.float64)
        inc_grid    = np.unique(np.concatenate([inc_grid, polar_extra]))

    raan_interval = 360.0 / N_P
    raan_grid     = np.linspace(0.0, raan_interval, N_RAAN_OFFSETS, endpoint=False)

    seed_pairs    = [(float(inc), float(raan)) for inc in inc_grid for raan in raan_grid]
    n_seeds       = len(seed_pairs)
    n_seeds_total = n_seeds
    n_polar_extra = int(sum(
        1 for p in _POLAR_RESERVATION_INCLINATIONS
        if not any(abs(p - g) < 0.01 for g in np.arange(
            float(inc_lo),
            float(inc_hi) + float(INCLINATION_SCREEN_STEP_DEG) * 0.5,
            float(INCLINATION_SCREEN_STEP_DEG),
        ))
    )) if RESERVE_POLAR_ORBITS else 0
    print(
        f"\nSeed grid: {len(inc_grid)} inclinations × {N_RAAN_OFFSETS} RAANs = {n_seeds} seeds"
    )
    print(f"  inclinations : {float(inc_lo):.1f}°–{float(inc_hi):.1f}° step {float(INCLINATION_SCREEN_STEP_DEG):.1f}°"
          + (f"  (+{n_polar_extra} polar at 5° steps)" if n_polar_extra > 0 else ""))
    print(f"  RAANs        : {N_RAAN_OFFSETS} offsets in [0°, {raan_interval:.3f}°)  "
          f"(360° / N_P={N_P})  step={raan_interval / N_RAAN_OFFSETS:.3f}°")

    # ------------------------------------------------------------------
    # Build sub-layers and simulate all seeds in parallel
    # ------------------------------------------------------------------
    _a_km_ref  = _a_j2        if sim.use_j2 else None
    _dt_s_slot = dt_s_actual  if sim.use_j2 else None
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
        reserve_polar_orbits=RESERVE_POLAR_ORBITS,
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
    # Tag seeds by selection phase: polar-reserved, inc-guaranteed, or greedy fill.
    polar_reserved_set: dict[int, float] = {}   # idx → reserved inclination
    if RESERVE_POLAR_ORBITS and inc_deg_list:
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
        mip_focus=MIP_FOCUS,
        heuristics_frac=HEURISTICS_FRAC,
        solver_params=SOLVER_PARAMS,
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
        # --- Target ---
        "target_mode":                    TARGET_MODE,
        "target_country":                 TARGET_COUNTRY if TARGET_MODE == "country" else None,
        "target_points":                  [list(p) for p in TARGET_POINTS] if TARGET_MODE != "country" else None,
        "target_shell_n_points":          TARGET_SHELL_N_POINTS,
        "n_targets":                      int(n_targets),
        # --- Engagement physics ---
        "dt_s":                           dt_s_actual,
        "dt_s_nominal":                   DT_S,
        "t_window_s":                     T_WINDOW_S,
        "v_bo_km_s":                      V_BO_KM_S,
        "a_g":                            A_G,
        "intercept_alt_km":               INTERCEPT_ALT_KM,
        "min_elev_deg":                   MIN_ELEV_DEG,
        "r_max_km":                       float(cov.max_range_km),
        # --- Orbit ---
        "altitude_override_km":           ALTITUDE_OVERRIDE_KM,
        "altitude_max_repeat_days":       ALTITUDE_MAX_REPEAT_DAYS,
        "h_star_km":                      float(h_star),
        "actual_alt_km":                  float(actual_alt_km),
        "rgt_ratio":                      f"{N_P}:{N_D}",
        "L":                              int(L),
        "horizon_s":                      float(horizon_s),
        # --- Seed grid ---
        "inclination_sweep_bounds_deg":   list(INCLINATION_SWEEP_BOUNDS_DEG),
        "inclination_screen_step_deg":    INCLINATION_SCREEN_STEP_DEG,
        "n_raan_offsets":                  N_RAAN_OFFSETS,
        "raan_interval_deg":              float(raan_interval),
        "raan_step_deg":                  float(raan_interval / N_RAAN_OFFSETS),
        "n_seeds_simulated":              int(n_seeds_total),
        "n_seeds_live":                   int(n_seeds_live),
        "n_seeds_for_milp":               N_SEEDS_FOR_MILP,
        "reserve_polar_orbits":           RESERVE_POLAR_ORBITS,
        "polar_reservation_inclinations": list(_POLAR_RESERVATION_INCLINATIONS) if RESERVE_POLAR_ORBITS else [],
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
        "mip_focus":                      MIP_FOCUS,
        "heuristics_frac":                HEURISTICS_FRAC,
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
