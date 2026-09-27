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

# ---------------------------------------------------------------------------
# Seed selection strategy: rarity-weighted, margin-driven build + trim.
#
# Rather than handing the MILP a fixed seed count (the old N_SEEDS_FOR_MILP
# knob), seeds are added greedily -- each pick weighted by how globally
# scarce its access is across the whole candidate pool, not just current
# deficit -- until every target's accumulated access reaches
# MARGIN_MULTIPLIER x r_required at the worst-covered target. That number
# doesn't need to be hand-tuned per r_required/n_targets combination the way
# a fixed count did. A reverse-greedy trim pass then removes whichever
# seeds are least damaging to keep, down to a leaner TRIM_MARGIN_MULTIPLIER x
# r_required floor, to recover most of the extra seed count spent building a
# safety margin during construction. See _greedy_select_seeds/_trim_seeds.
# ---------------------------------------------------------------------------
MARGIN_MULTIPLIER: float = 3.0
TRIM_MARGIN_MULTIPLIER: float = 1.25
# Hard ceiling on total seeds passed to the MILP, regardless of margin
# reached -- a safety valve only (e.g. a target that's geometrically
# unreachable at the target margin within the candidate grid), not itself a
# target to hit. The build stops on margin, not on this count, unless this
# cap binds first.
MAX_SEEDS_FOR_MILP: int = 200
# Phase 1 does not guarantee a representative for EVERY live inclination --
# on a wide live-inclination-range scenario that can, by itself, blow far
# past MARGIN_MULTIPLIER before Phase 2 (the actual margin-driven logic) ever
# runs. Instead, Phase 1 reserves a SMALL, requirement-scaled number of seeds
# restricted to a narrow near-polar inclination band -- generalizing the
# older, fixed 3-seed polar-reservation feature (removed) into something
# requirement-scaled and rarity-weighted, rather than everything else being
# left entirely to Phase 2's unconstrained greedy fill.
POLAR_BAND_DEG: tuple[float, float] = (70.0, 90.0)
POLAR_BAND_DIVISOR: float = 5.0

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
    inc_deg_list: list[float] | None = None,
) -> tuple[list[int], set[int], np.ndarray, np.ndarray]:
    """Rarity-weighted, margin-driven greedy seed selection.

    Two changes relative to a plain deficit-capped greedy:

    1. Every score is weighted by a static, pool-level rarity weight per
       target (see _rarity_weights). A candidate's score would otherwise
       only reflect how much CURRENT deficit it satisfied, with no notion of
       how globally scarce a target's access is across the whole candidate
       pool -- two targets at equal current deficit would score identically
       even if one was reachable by 90% of the pool and the other by 2%.
    2. The per-target "deficit" driving both the capped score and the
       stopping rule is measured against margin_multiplier * r_required, not
       bare r_required. This replaces a fixed seed count with a measured
       stopping criterion -- keep adding seeds until the WORST-covered
       target's accumulated access reaches margin_multiplier x the
       requirement, not until an arbitrary count is hit. Capping the deficit
       against the actual build target (rather than capping at 1x and
       falling back to an unweighted raw-access tiebreaker once 1x is
       reached) means every pick, all the way from zero coverage up through
       the margin target, stays directed at whichever target most needs it.

    Phase 1 -- Polar-band reservation (capped, not unconditional):
        Reserves up to ceil(r_required / POLAR_BAND_DIVISOR) seeds, eligible
        inclinations restricted to POLAR_BAND_DEG, chosen by the same
        rarity-weighted, margin-capped score used everywhere else. This is
        NOT "one per every live inclination" -- on a scenario with many live
        inclinations, guaranteeing all of them can by itself blow past the
        margin target before Phase 2 ever runs. Restricting to a small,
        bounded, near-polar band instead targets the band Phase 2's pure
        marginal-value scoring is least likely to visit on its own (near-
        polar orbits typically have low instantaneous access to a specific
        mid-latitude target set, so an unconstrained greedy tends to
        under-sample them even though they may offer valuable, hard-to-
        replace coverage geometry).

    Phase 2 -- Greedy fill:
        Keeps adding seeds (rarity-weighted, margin-capped score, unbounded
        in inclination) until EITHER the worst-covered target's accumulated
        access reaches margin_multiplier * r_required, OR max_seeds is hit
        (a safety ceiling, not a target), OR no live candidates remain.

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
    # Phase 1: Polar-band reservation -- capped at ceil(r_required /
    # POLAR_BAND_DIVISOR) seeds, eligible inclinations restricted to
    # POLAR_BAND_DEG. Not "one per every live inclination" -- see docstring.
    # ------------------------------------------------------------------
    if inc_deg_list is not None:
        polar_lo, polar_hi = POLAR_BAND_DEG
        polar_budget = int(np.ceil(r_required / POLAR_BAND_DIVISOR))
        pending_incs = sorted(
            inc for inc in set(inc_deg_list)
            if polar_lo <= inc <= polar_hi
        )

        while (
            pending_incs
            and len(selected) < max_seeds
            and len(polar_reserved) < polar_budget
        ):
            # Among all pending near-polar inclinations, find the (inc,
            # candidate) pair that contributes the most rarity-weighted
            # marginal coverage.
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
                break  # no live candidates remain for any pending inclination

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
    This is the reverse of the construction greedy -- the same vectorized
    "score everyone, take the extremal one, update state" pattern, run
    backward with a floor instead of a target.

    No seed is exempt from removal regardless of which phase selected it --
    Phase 1 reserves only a small, requirement-scaled number of seeds (see
    POLAR_BAND_DEG / POLAR_BAND_DIVISOR), so the risk of trim removing a
    large structural investment is bounded and acceptable; if the polar-band
    reservation genuinely isn't earning its keep for a given scenario,
    letting trim remove it is correct behavior, not a bug.
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

    # Step 2 — rarity-weighted, margin-driven build (replaces the old fixed
    # N_SEEDS_FOR_MILP seed count -- see _greedy_select_seeds' docstring)
    inc_deg_list  = [float(p[0]) for p in seed_pairs]
    margin_target = MARGIN_MULTIPLIER * r_required
    build_idx, polar_reserved_set, build_coverage, access_full = _greedy_select_seeds(
        v0_list, r_required, MAX_SEEDS_FOR_MILP,
        margin_multiplier=MARGIN_MULTIPLIER,
        inc_deg_list=inc_deg_list,
    )
    build_worst_margin = float(build_coverage.min())
    n_built = len(build_idx)
    print(
        f"\nBuild phase: {n_built} of {n_seeds} live seeds selected  "
        f"(target: worst-slot margin >= {MARGIN_MULTIPLIER:.2f}x requirement = {margin_target:.1f})"
    )
    print(
        f"  Worst-slot margin achieved: {build_worst_margin:.1f}  "
        f"({build_worst_margin / r_required:.2f}x requirement)"
        + ("  [safety cap MAX_SEEDS_FOR_MILP reached before target margin]"
           if n_built >= MAX_SEEDS_FOR_MILP and build_worst_margin < margin_target - 1e-9
           else "")
    )

    # Step 3 — reverse marginal-contribution trim back down to a leaner floor
    trim_floor = TRIM_MARGIN_MULTIPLIER * r_required
    greedy_idx = _trim_seeds(
        build_idx, access_full, build_coverage, r_required,
        TRIM_MARGIN_MULTIPLIER,
    )
    n_selected    = len(greedy_idx)
    trim_coverage = access_full[greedy_idx].sum(axis=0)
    trim_worst_margin = float(trim_coverage.min())
    n_trimmed = n_built - n_selected
    print(
        f"Trim phase : removed {n_trimmed} seed(s), {n_built} → {n_selected}  "
        f"(floor: worst-slot margin >= {TRIM_MARGIN_MULTIPLIER:.2f}x requirement = {trim_floor:.1f})"
    )
    print(
        f"  Worst-slot margin after trim: {trim_worst_margin:.1f}  "
        f"({trim_worst_margin / r_required:.2f}x requirement)"
    )

    # Coverage proxy: sum of per-target access over selected seeds.
    # Due to the circulant property this equals simultaneous views at any t.
    feasible_proxy = trim_worst_margin >= r_required
    print(
        f"  Coverage proxy (all slots of selected seeds): "
        f"min={trim_worst_margin:.1f}  mean={float(trim_coverage.mean()):.1f}  "
        f"requirement={r_required}  "
        + ("✓ feasible" if feasible_proxy
           else "⚠ infeasible even after build+trim — try raising MARGIN_MULTIPLIER or MAX_SEEDS_FOR_MILP")
    )
    # Tag seeds by selection phase: polar-band reservation (Phase 1) vs.
    # quantity-driven greedy fill (Phase 2). Not a protection from trimming
    # -- purely informational (see _trim_seeds).
    for rank, idx in enumerate(greedy_idx):
        inc, raan = seed_pairs[idx]
        tag = "  [phase 1: polar-band reservation]" if idx in polar_reserved_set else ""
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
        "margin_multiplier":              MARGIN_MULTIPLIER,
        "trim_margin_multiplier":         TRIM_MARGIN_MULTIPLIER,
        "max_seeds_safety_cap":           MAX_SEEDS_FOR_MILP,
        "polar_band_deg":                 list(POLAR_BAND_DEG),
        "polar_band_divisor":             POLAR_BAND_DIVISOR,
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
