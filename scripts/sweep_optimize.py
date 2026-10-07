"""
sweep_optimize.py

Sequential parameter sweep for the SBI SCLP constellation optimizer.

Iterates over all combinations of:
  - Country target    : China → North Korea (China runs first)
  - Requested altitude: 350, 450, 600 km   (snapped to nearest RGT orbit)
  - Salvo size        : 1, 10, 20, 50, 100
  - Interceptors/sat  : 1, 2, 4, 6
  - Burnout velocity  : 4, 5, 6, 10 km/s

Total: 2 × 3 × 5 × 4 × 4 = 480 combinations.

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
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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
# Per-country inclination sweep bounds — lower bound set to approximately the
# minimum latitude of interest for each country to avoid wasting seeds on
# inclinations that can never reach the target.
INCLINATION_SWEEP_BOUNDS_DEG: dict[str, tuple[float, float]] = {
    "China":       (20.0, 90.0),
    "North Korea": (35.0, 90.0),
}
INCLINATION_SCREEN_STEP_DEG: float                = 1.0
ALTITUDE_MAX_REPEAT_DAYS: int                     = 2
N_RAAN_OFFSETS: int                               = 10
SEED_SCREENING_WORKERS: int                       = 8
# ---------------------------------------------------------------------------
# Seed selection strategy: rarity-weighted, margin-driven build + trim.
# Mirrors lee_test2.py exactly -- see that file's constants block and
# _greedy_select_seeds/_trim_seeds docstrings for full rationale. Rather than
# handing the MILP a fixed seed count (the old SEEDS_FOR_MILP_BY_R_REQUIRED
# lookup, removed), seeds are added greedily -- each pick weighted by how
# globally scarce its access is across the whole candidate pool -- until
# every target's accumulated access reaches MARGIN_MULTIPLIER x r_required at
# the worst-covered target. A reverse-greedy trim pass then removes whichever
# seeds are least damaging to keep, down to a leaner TRIM_MARGIN_MULTIPLIER x
# r_required floor.
# ---------------------------------------------------------------------------
MARGIN_MULTIPLIER: float = 3.0
TRIM_MARGIN_MULTIPLIER: float = 1.25
# Hard ceiling on total seeds passed to the MILP, regardless of margin
# reached -- a safety valve only, not itself a target to hit.
MAX_SEEDS_FOR_MILP: int = 200
# Phase 1 reserves a small, requirement-scaled number of seeds restricted to
# a narrow near-polar inclination band (see _greedy_select_seeds' docstring
# below, mirrored verbatim from lee_test2.py, for why this is capped rather
# than unconditional).
POLAR_BAND_DEG: tuple[float, float] = (70.0, 90.0)
POLAR_BAND_DIVISOR: float = 5.0
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
SWEEP_ALTITUDES_KM: list[float]            = [250.0, 550.0]
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
    """Build the 9-field dedup key for a progress-log entry.

    The 4 new dims (detection/decision/burnout time, a_g) default to the
    current fixed constants when absent, so log lines written before these
    dims existed still match today's single-element-list sweeps exactly.
    """
    return (
        str(entry["country"]),
        float(entry["altitude_km"]),
        int(entry["salvo_size"]),
        int(entry["n_interceptors_per_sat"]),
        float(entry["v_bo_km_s"]),
        float(entry.get("detection_time_s", DETECTION_TIME_S)),
        float(entry.get("decision_time_s", DECISION_TIME_S)),
        float(entry.get("target_missile_burnout_time_s", TARGET_MISSILE_BURNOUT_TIME_S)),
        float(entry.get("a_g", A_G)),
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
# Seed selection — verbatim copy of lee_test2.py's rarity-weighted,
# margin-driven build + trim (see lee_test2.py for full rationale).
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
       requirement, not until an arbitrary count is hit.

    Phase 1 -- Polar-band reservation (capped, not unconditional):
        Reserves up to ceil(r_required / POLAR_BAND_DIVISOR) seeds, eligible
        inclinations restricted to POLAR_BAND_DEG, chosen by the same
        rarity-weighted, margin-capped score used everywhere else.

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

    No seed is exempt from removal regardless of which phase selected it --
    Phase 1 reserves only a small, requirement-scaled number of seeds, so if
    the polar-band reservation genuinely isn't earning its keep for a given
    scenario, letting trim remove it is correct behavior, not a bug.
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
# Parameterised result subfolder builder
# ---------------------------------------------------------------------------

def _result_subdir_for(
    country: str,
    v_bo_km_s: float,
    actual_alt_km: float,
    n_interceptors_per_sat: int,
    salvo_size: int,
    *,
    detection_time_s: float,
    decision_time_s: float,
    target_missile_burnout_time_s: float,
    a_g: float,
) -> Path:
    """Build and create the hierarchical results subfolder for one sweep run.

    detection_time_s/decision_time_s/target_missile_burnout_time_s/a_g are
    taken as parameters (not read from the module constants) because they can
    now vary per combo (item 9's new sweep dimensions) -- reading the module
    constants directly here would label every combo's folder with the same
    stale value regardless of which one was actually used.
    """
    target_slug = country.lower().replace(" ", "-")
    path = (
        RESULTS_DIR / target_slug
        / f"burnout-vel-{v_bo_km_s:.1f}km-s"
        # Label kept as "intercept-window" (not renamed) so existing
        # path-parsing regexes (cloud_plot_best_results.py,
        # plot_best_results_by_country.py) keep matching -- only the source
        # of the number changed, from T_WINDOW_S to
        # target_missile_burnout_time_s (numerically identical at defaults).
        / f"intercept-window-{target_missile_burnout_time_s:.0f}s"
        # New segments for the other two timing dims. Named as short
        # "det"/"dec" (not the fuller "detection-time"/"decision-time" used
        # in batch_worker.py's/cloud_sweep.py's S3 keys, which have no length
        # limit) because this path is written to the local filesystem: with
        # the full labels, a realistic country-mode result path landed at
        # 257-271 chars, over Windows' 260-char MAX_PATH, and failed with a
        # bare FileNotFoundError on write. No downstream tooling parses these
        # two segments by name (only "intercept-window-" is regex-matched by
        # plot_best_results_by_country.py), so shortening them is safe.
        / f"det-{detection_time_s:.0f}s"
        / f"dec-{decision_time_s:.0f}s"
        / f"intercept-alt-{INTERCEPT_ALT_KM:.0f}km"
        / f"max-accel-{a_g:.1f}g"
        / f"orbit-alt-{round(actual_alt_km)}km"
        / f"interceptors-per-sat-{n_interceptors_per_sat}"
        / f"salvo-size-{salvo_size}"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


# ---------------------------------------------------------------------------
# Single-combination optimization run
# ---------------------------------------------------------------------------

def run_single(
    *,
    country: str,
    target_altitude_km: float,
    salvo_size: int,
    n_interceptors_per_sat: int,
    v_bo_km_s: float,
    inclination_sweep_bounds_deg: tuple[float, float],
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
    # Snap to nearest RGT orbit at the requested altitude
    # ------------------------------------------------------------------
    candidates = candidate_rgt_ratios(
        h_star_km=target_altitude_km,
        earth=earth,
        max_repeat_days=ALTITUDE_MAX_REPEAT_DAYS,
        dt_s=DT_S,
    )
    if not candidates:
        raise ValueError(
            f"No RGT orbit found near {target_altitude_km:.0f} km "
            f"with N_D ≤ {ALTITUDE_MAX_REPEAT_DAYS}.  "
            "Raise ALTITUDE_MAX_REPEAT_DAYS or choose a different altitude."
        )
    best          = candidates[0]
    N_D           = int(best["N_D"])
    N_P           = int(best["N_P"])
    L             = int(best["L"])
    actual_alt_km = float(best["alt_km"])
    dt_s_actual   = float(best["T_r_s"]) / L

    horizon_s     = (L - 1) * dt_s_actual
    print(
        f"\nTarget altitude requested: {target_altitude_km:.1f} km  →  "
        f"nearest RGT orbit: {N_P}:{N_D}  "
        f"actual_alt={actual_alt_km:.2f} km  "
        f"(snap error: {float(best['alt_error_km']):.2f} km)"
    )

    sim = SimConfig(
        horizon_s=horizon_s,
        dt_s=dt_s_actual,
        analysis_matrix_dtype="uint8",
        use_j2=False,
        use_drag=USE_DRAG,
    )
    print(f"Shared repeat family: N_D={N_D}  L={L} steps")
    print(f"Simulation horizon: {horizon_s:.1f} s  ({horizon_s / 3600:.3f} h)  -> L={L}")
    print(f"dt_s: {DT_S:.4f} s (nominal) → {dt_s_actual:.6f} s (Keplerian T_r/L)")

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
    # Inclination / RAAN seed grid
    # ------------------------------------------------------------------
    inc_lo, inc_hi = inclination_sweep_bounds_deg
    inc_grid = np.arange(
        float(inc_lo),
        float(inc_hi) + float(INCLINATION_SCREEN_STEP_DEG) * 0.5,
        float(INCLINATION_SCREEN_STEP_DEG),
    )

    raan_interval = 360.0 / N_P
    raan_grid     = np.linspace(0.0, raan_interval, N_RAAN_OFFSETS, endpoint=False)
    seed_pairs    = [(float(inc), float(raan)) for inc in inc_grid for raan in raan_grid]
    n_seeds_total = len(seed_pairs)
    print(f"\nSeed grid: {len(inc_grid)} inclinations × {N_RAAN_OFFSETS} RAANs = {n_seeds_total} seeds")

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
    # Simulate seeds in parallel
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
            propagator=PROPAGATOR,
        )
        return idx, (analysis.counts > 0).astype(np.float64)

    n_seeds = n_seeds_total
    print(f"\nSimulating {n_seeds} seeds  (workers={SEED_SCREENING_WORKERS}) ...")
    t_sim0 = time.perf_counter()
    v0_list: list[np.ndarray] = [None] * n_seeds  # type: ignore[list-item]
    progress_every = max(1, n_seeds // 20)

    with ThreadPoolExecutor(max_workers=SEED_SCREENING_WORKERS) as pool:
        futs = {pool.submit(_simulate_seed, i): i for i in range(n_seeds)}
        done_count = 0
        for fut in as_completed(futs):
            idx, v0 = fut.result()
            v0_list[idx] = v0
            done_count += 1
            if done_count % progress_every == 0 or done_count == n_seeds:
                print(f"  {done_count}/{n_seeds} seeds simulated ...", flush=True)

    print(f"Total seed simulation time: {time.perf_counter() - t_sim0:.2f} s")

    # ------------------------------------------------------------------
    # Seed selection: dead-filter → rarity-weighted margin build → trim
    # ------------------------------------------------------------------
    r_required = int(np.ceil(salvo_size / n_interceptors_per_sat))

    live_mask = [bool(v0.any()) for v0 in v0_list]
    n_dead    = live_mask.count(False)
    if n_dead:
        print(f"\nPre-filter: removed {n_dead}/{n_seeds} seeds with zero target access.")
        v0_list    = [v for v, keep in zip(v0_list,    live_mask) if keep]
        sub_layers = [s for s, keep in zip(sub_layers, live_mask) if keep]
        seed_pairs = [p for p, keep in zip(seed_pairs, live_mask) if keep]
        n_seeds    = len(v0_list)
    n_seeds_live = n_seeds

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

    feasible_proxy = trim_worst_margin >= r_required
    print(
        f"  Coverage proxy (all slots of selected seeds): "
        f"min={trim_worst_margin:.1f}  mean={float(trim_coverage.mean()):.1f}  "
        f"requirement={r_required}  "
        + ("✓ feasible" if feasible_proxy
           else "⚠ infeasible even after build+trim — try raising MARGIN_MULTIPLIER or MAX_SEEDS_FOR_MILP")
    )

    for rank, idx in enumerate(greedy_idx):
        inc, raan = seed_pairs[idx]
        tag = "  [phase 1: polar-band reservation]" if idx in polar_reserved_set else ""
        print(
            f"  {rank + 1:2d}: i={inc:5.1f}°  RAAN={raan:6.1f}°  "
            f"access/target={v0_list[idx].sum(axis=0).mean():.1f}{tag}"
        )

    # Reindex to MILP subset (seed_pairs intentionally NOT reindexed — greedy_idx still valid)
    v0_list_milp    = [v0_list[i]    for i in greedy_idx]
    sub_layers_milp = [sub_layers[i] for i in greedy_idx]

    # ------------------------------------------------------------------
    # Build constraint matrix
    # ------------------------------------------------------------------
    param_V = build_param_V(v0_list_milp)
    param_V_summary(param_V, n_seeds=n_selected)

    param_r = r_required * np.ones((L, n_targets), dtype=np.float64)
    print(
        f"\nCoverage requirement: salvo_size={salvo_size}, "
        f"interceptors_per_sat={n_interceptors_per_sat}, "
        f"r_required={r_required} satellite(s) in view per time step"
    )

    # ------------------------------------------------------------------
    # Solve SCLP
    # ------------------------------------------------------------------
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = None
    if LOG_DIR is not None:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_file = LOG_DIR / f"SCLP_{ts}.txt"

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
    orb_elems = build_orb_elems_table(sub_layers_milp, L)
    print_orb_table(orb_elems, result.selected_slots)

    out_dir  = _result_subdir_for(
        country, v_bo_km_s, actual_alt_km, n_interceptors_per_sat, salvo_size,
        detection_time_s=detection_time_s, decision_time_s=decision_time_s,
        target_missile_burnout_time_s=target_missile_burnout_time_s, a_g=a_g,
    )
    csv_path = out_dir / f"{ts}_result.csv"

    run_config = {
        # --- Run metadata ---
        "run_name":                       run_name,
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
        # --- Orbit ---
        "target_altitude_km":             target_altitude_km,
        "altitude_max_repeat_days":       ALTITUDE_MAX_REPEAT_DAYS,
        "actual_alt_km":                  float(actual_alt_km),
        "rgt_ratio":                      f"{N_P}:{N_D}",
        "L":                              int(L),
        "horizon_s":                      float(horizon_s),
        # --- Seed grid ---
        "inclination_sweep_bounds_deg":   list(inclination_sweep_bounds_deg),
        "inclination_screen_step_deg":    INCLINATION_SCREEN_STEP_DEG,
        "n_raan_offsets":                 N_RAAN_OFFSETS,
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
    bc_kg_m2 = float(np.asarray(sub_layers_milp[0].phys.bc_kg_m2).ravel()[0])

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
        SWEEP_ALTITUDES_KM,
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
            r_alt      = float(rerun_entry["altitude_km"])
            r_salvo    = int(rerun_entry["salvo_size"])
            r_n_int    = int(rerun_entry["n_interceptors_per_sat"])
            r_vbo      = float(rerun_entry["v_bo_km_s"])
            # New dims default to the current fixed constants for log entries
            # written before these dims existed (see _run_key_from_entry).
            r_detect   = float(rerun_entry.get("detection_time_s", DETECTION_TIME_S))
            r_decide   = float(rerun_entry.get("decision_time_s", DECISION_TIME_S))
            r_burnout  = float(rerun_entry.get("target_missile_burnout_time_s", TARGET_MISSILE_BURNOUT_TIME_S))
            r_a_g      = float(rerun_entry.get("a_g", A_G))
            r_combo    = (r_country, r_alt, r_salvo, r_n_int, r_vbo, r_detect, r_decide, r_burnout, r_a_g)
            r_run_name = f"{RUN_NAME}_{grid.index(r_combo) + 1}" if RUN_NAME and r_combo in grid else RUN_NAME
            print(
                f"\n[Phase 0]  {r_country}  alt={r_alt:.0f} km  "
                f"salvo={r_salvo}  int/sat={r_n_int}  v_bo={r_vbo:.1f} km/s  "
                f"detect={r_detect:.0f}s  decide={r_decide:.0f}s  burnout={r_burnout:.0f}s  a_g={r_a_g:.1f}"
            )
            t_run0 = time.perf_counter()
            try:
                summary  = run_single(
                    country=r_country,
                    target_altitude_km=r_alt,
                    salvo_size=r_salvo,
                    n_interceptors_per_sat=r_n_int,
                    v_bo_km_s=r_vbo,
                    inclination_sweep_bounds_deg=INCLINATION_SWEEP_BOUNDS_DEG.get(
                        r_country, (20.0, 90.0)
                    ),
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
                    "altitude_km":            r_alt,
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
                    "altitude_km":            r_alt,
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
          f"({' × '.join(str(len(x)) for x in [SWEEP_COUNTRIES, SWEEP_ALTITUDES_KM, SWEEP_SALVO_SIZES, SWEEP_INTERCEPTORS_PER_SAT, SWEEP_BURNOUT_VELOCITIES_KM_S, SWEEP_DETECTION_TIMES_S, SWEEP_DECISION_TIMES_S, SWEEP_BURNOUT_TIMES_S, SWEEP_A_G_LIST])})")
    print(f"  Already completed  : {n_already_done}")
    print(f"  Remaining          : {n_remaining}")
    print(f"  Time budget / run  : {SWEEP_TIME_LIMIT_S / 60:.0f} min  OR  MIP gap ≤ {SWEEP_MIP_GAP * 100:.1f}%")
    est_h = n_remaining * SWEEP_TIME_LIMIT_S / 3600.0
    print(f"  Est. max runtime   : {est_h:.1f} h  ({est_h / 24:.1f} days)")
    print(f"  Progress log       : {SWEEP_PROGRESS_FILE}")
    print("=" * 70)

    for combo_idx, (country, alt_km, salvo, n_int, v_bo, detect_s, decide_s, burnout_s, a_g_val) in enumerate(grid, 1):
        key = (country, float(alt_km), int(salvo), int(n_int), float(v_bo),
               float(detect_s), float(decide_s), float(burnout_s), float(a_g_val))
        combo_run_name = f"{RUN_NAME}_{combo_idx}" if RUN_NAME else None

        if key in completed_keys:
            print(
                f"\n[{combo_idx:3d}/{n_total}] SKIP  {country}  "
                f"alt={alt_km:.0f}km  salvo={salvo}  int/sat={n_int}  v_bo={v_bo:.0f}km/s  "
                f"detect={detect_s:.0f}s  decide={decide_s:.0f}s  burnout={burnout_s:.0f}s  a_g={a_g_val:.1f}"
            )
            continue

        print(f"\n{'=' * 70}")
        print(
            f"[{combo_idx:3d}/{n_total}]  {country}  alt={alt_km:.0f}km  "
            f"salvo={salvo}  int/sat={n_int}  v_bo={v_bo:.0f}km/s  "
            f"detect={detect_s:.0f}s  decide={decide_s:.0f}s  burnout={burnout_s:.0f}s  a_g={a_g_val:.1f}"
        )
        print(f"{'=' * 70}")

        t_run0 = time.perf_counter()
        try:
            summary = run_single(
                country=country,
                target_altitude_km=float(alt_km),
                salvo_size=int(salvo),
                n_interceptors_per_sat=int(n_int),
                v_bo_km_s=float(v_bo),
                inclination_sweep_bounds_deg=INCLINATION_SWEEP_BOUNDS_DEG.get(
                    country, (20.0, 90.0)
                ),
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
                "altitude_km":            alt_km,
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
                "altitude_km":            alt_km,
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
