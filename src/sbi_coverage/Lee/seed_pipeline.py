"""Seed pool pipeline used by Lee/run_pipeline.py.

Stages (all after seed *planning*, which is Lee/seed_families.plan_seed_families):

  1. build_seed_pool      -- one seed per (family, RAAN offset) at raan_step_deg spacing
  2. simulate_seed_pool   -- slot-0 access profile of every seed (threaded)
  3. filter_seed_pool     -- drop dead seeds, worst-case-first greedy build, reverse trim,
                             exact all-slots-on feasibility check, build param_V
  4. seed_run_config      -- the orbit / seed-generation / seed-filter block of the result JSON's run_config

Terminology: a *seed* is one (family, RAAN offset); its L time-shifted slots are the candidate
satellites (columns of param_V) and are never simulated individually -- the circulant property of
Lee et al. (2020) gives them from the seed's access profile. A seed's *profile* is its (L x n_targets)
0/1 access matrix.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

import numpy as np

from ..core.config import CoverageConfig, EarthConstants, SimConfig
from ..core.shell import Shell
from ..core.simulate import run_simulation
from .param_v import build_param_V
from .rgt_slots import rgt_common_ground_track_layer, seed_layer
from .seed_families import n_distinct_slots


# ---------------------------------------------------------------------------
# Greedy build + reverse trim
# ---------------------------------------------------------------------------

def rarity_weights(access: np.ndarray) -> np.ndarray:
    """Static, pool-level scarcity weight per target (rank based, scarcest = 1.0, most
    common = 0.1).  Used only as a tie-break by greedy_select_seeds."""
    total_access = access.sum(axis=0)
    n_targets = len(total_access)
    if n_targets <= 1:
        return np.ones(n_targets)
    order = np.argsort(total_access)
    rank = np.empty(n_targets, dtype=np.float64)
    rank[order] = np.arange(n_targets, dtype=np.float64)
    return 0.1 + 0.9 * (1.0 - rank / (n_targets - 1))


def greedy_select_seeds(
    seed_profiles: list[np.ndarray],
    r_required: int,
    max_seeds: int,
    *,
    margin_multiplier: float,
    slot_fractions: list[float] | None = None,
) -> tuple[list[int], np.ndarray, np.ndarray]:
    """Worst-case-first, margin-driven greedy seed selection.

    A seed's "access" is the per-target total of its profile, profile.sum(axis=0).
    Because every slot of a circulant block is a circular shift of the profile, turning ON all
    slots of a seed adds exactly that many satellites in view of each target at EVERY time
    step, so the sum of access over the selected seeds is an exact certificate that the
    requirement can be met (for seeds that keep all L slots).

    Ranking: the number of satellites needed is set by the WORST-covered target, so each
    pick is the seed that most raises the minimum accumulated access over targets (primary
    key, uncapped -- a cap at the margin would make every seed that already clears it tie
    and the pick arbitrary at small r_required).  Ties are separated by the rarity-weighted,
    deficit-capped sum of access.

    Seeds are added until the worst-covered target's accumulated access reaches
    margin_multiplier * r_required (margin_multiplier is a dimensionless multiple of the
    requirement), OR max_seeds is hit (a safety ceiling), OR no candidates remain.
    slot_fractions scales each seed's access by the fraction of its L slots that are kept
    (reducible families keep L//g).

    Returns (selected indices in selection order, final per-target coverage, access matrix).
    """
    n_seeds = len(seed_profiles)
    access = np.stack([v0.sum(axis=0) for v0 in seed_profiles]).astype(np.float64)
    if slot_fractions is not None:
        access = access * np.asarray(slot_fractions, dtype=np.float64)[:, None]
    rarity = rarity_weights(access)
    margin_target = margin_multiplier * r_required
    coverage = np.zeros(access.shape[1], dtype=np.float64)
    selected: list[int] = []
    remaining = list(range(n_seeds))

    def _pick(cands_idx: list[int]) -> int:
        cand = access[cands_idx]
        worst_after = (coverage[None, :] + cand).min(axis=1)
        tied = np.flatnonzero(worst_after >= worst_after.max() - 1e-9)
        deficit = np.maximum(0.0, margin_target - coverage)
        tie_score = (np.minimum(cand[tied], deficit) * rarity).sum(axis=1)
        return int(tied[int(np.argmax(tie_score))])

    while remaining and len(selected) < max_seeds:
        if float(coverage.min()) >= margin_target:
            break
        best_local = _pick(remaining)
        best_global = remaining[best_local]
        selected.append(best_global)
        coverage = coverage + access[best_global]
        remaining.pop(best_local)

    return selected, coverage, access


def trim_seeds(
    selected: list[int],
    access: np.ndarray,
    coverage: np.ndarray,
    r_required: int,
    trim_margin_multiplier: float,
) -> list[int]:
    """Reverse marginal-contribution trim -- single pass, no seed is exempt.

    Repeatedly removes whichever selected seed does the least damage to the worst-covered
    target, stopping once no further removal would leave every target at or above
    trim_margin_multiplier * r_required.
    """
    floor = trim_margin_multiplier * r_required
    current = list(selected)
    coverage = coverage.copy()
    while True:
        remaining_margins = {i: float((coverage - access[i]).min()) for i in current}
        if not remaining_margins:
            break
        best_i = max(remaining_margins, key=remaining_margins.get)
        if remaining_margins[best_i] < floor:
            break
        current.remove(best_i)
        coverage = coverage - access[best_i]
    return current


# ---------------------------------------------------------------------------
# Pool construction and simulation
# ---------------------------------------------------------------------------

def build_seed_pool(plan: dict, raan_step_deg: float, earth: EarthConstants):
    """One seed per (family, RAAN offset).  RAAN offsets run from 0 in steps of
    raan_step_deg across each family's distinct-track window (360 / reduced N_P).

    Returns (seed_meta, seed_layers, n_raan_offsets):
      seed_meta[i]    = {"family": index into plan["families"], "raan_deg": float}
      seed_layers[i]  = the L-slot Layer of seed i
      n_raan_offsets  = number of RAAN offsets per family (list aligned with plan["families"])
    """
    if raan_step_deg <= 0.0:
        raise ValueError("raan_step_deg must be > 0")
    families = plan["families"]
    L, dt = plan["L"], plan["dt_s"]
    seed_meta: list[dict] = []
    n_raan_offsets: list[int] = []
    for fi, f in enumerate(families):
        n_raan = int(np.ceil(f["raan_interval_deg"] / raan_step_deg - 1e-9))
        n_raan_offsets.append(n_raan)
        for k in range(n_raan):
            seed_meta.append({"family": fi, "raan_deg": float(k * raan_step_deg)})
    seed_layers = [
        rgt_common_ground_track_layer(
            inc_deg=families[m["family"]]["inc_deg"],
            N_P=families[m["family"]]["N_P"],
            N_D=families[m["family"]]["N_D"],
            L=L,
            earth=earth,
            raan0_deg=m["raan_deg"],
            a_km=families[m["family"]]["a_km"],
            dt_s=dt,
        )
        for m in seed_meta
    ]
    return seed_meta, seed_layers, n_raan_offsets


def simulate_seed_pool(seed_layers, *, sim: SimConfig, cov: CoverageConfig, earth: EarthConstants, shell: Shell,
                       propagator: str, workers: int, expected_steps: int | None = None,
                       log: Callable[[str], None] = print) -> list[np.ndarray]:
    """Slot-0 access profile (L x n_targets, uint8 0/1) for every seed, simulated in parallel.

    ``expected_steps`` (the plan's L): if given, every profile must have exactly that many
    time rows -- a mismatch would silently give the circulant matrix the wrong period.
    """
    n_seeds = len(seed_layers)

    def _one(idx: int):
        s_layer = seed_layer(seed_layers[idx])
        analysis = run_simulation(
            elems0=s_layer.elems, sim=sim, cov=cov, earth=earth,
            shell=shell, layer_specs=s_layer.spec, propagator=propagator, build_meta=False,
        )
        profile = (analysis.counts > 0).astype(np.uint8)
        if expected_steps is not None and profile.shape[0] != expected_steps:
            raise RuntimeError(f"seed {idx}: simulated {profile.shape[0]} time steps, expected L={expected_steps}")
        return idx, profile

    log(f"\nSimulating {n_seeds} seeds  (workers={workers}) ...")
    t0 = time.perf_counter()
    profiles: list[np.ndarray] = [None] * n_seeds  # type: ignore[list-item]
    progress_every = max(1, n_seeds // 20)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_one, i): i for i in range(n_seeds)}
        done = 0
        for fut in as_completed(futs):
            idx, profile = fut.result()
            profiles[idx] = profile
            done += 1
            if done % progress_every == 0 or done == n_seeds:
                log(f"  {done}/{n_seeds} seeds simulated ...")
    log(f"Total seed simulation time: {time.perf_counter() - t0:.2f} s")
    return profiles


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

def filter_seed_pool(
    seed_profiles, seed_layers, seed_meta, plan: dict, r_required: int, *,
    margin_multiplier: float, trim_margin_multiplier: float, max_seeds: int,
    log: Callable[[str], None] = print,
) -> dict:
    """Drop dead seeds, run the greedy build + reverse trim, build the (restricted-column)
    constraint matrix and verify exact feasibility.

    margin_multiplier / trim_margin_multiplier are dimensionless multiples of ``r_required``.
    Raises ValueError if no seed is live or the requirement cannot be met even with every
    slot of the selected seeds on.  Returns a dict with the kept seed_profiles / seed_layers /
    seed_meta, cols_per_seed, param_V, and the counts recorded in the result JSON.
    """
    families, L = plan["families"], plan["L"]
    n_total = len(seed_profiles)

    live_mask = [bool(v0.any()) for v0 in seed_profiles]
    n_dead = live_mask.count(False)
    if n_dead:
        log(f"\nPre-filter: removed {n_dead}/{n_total} seeds with zero target access.")
        seed_profiles = [v for v, keep in zip(seed_profiles, live_mask) if keep]
        seed_layers = [s for s, keep in zip(seed_layers, live_mask) if keep]
        seed_meta = [m for m, keep in zip(seed_meta, live_mask) if keep]
    n_live = len(seed_profiles)
    if n_live == 0:
        raise ValueError("No live seeds: no seed family has access to any target.")

    cols_all = [n_distinct_slots(families[m["family"]], L) for m in seed_meta]
    slot_frac = [nc / L for nc in cols_all]

    margin_target = margin_multiplier * r_required
    build_idx, build_coverage, access_full = greedy_select_seeds(
        seed_profiles, r_required, max_seeds, margin_multiplier=margin_multiplier, slot_fractions=slot_frac,
    )
    build_worst = float(build_coverage.min())
    n_built = len(build_idx)
    log(f"\nBuild phase: {n_built} of {n_live} live seeds selected  "
        f"(target: worst-slot margin >= {margin_multiplier:.2f}x requirement = {margin_target:.1f})")
    log(f"  Worst-slot margin achieved: {build_worst:.1f}  ({build_worst / r_required:.2f}x requirement)"
        + ("  [safety cap reached before target margin]"
           if n_built >= max_seeds and build_worst < margin_target - 1e-9 else ""))

    keep_idx = trim_seeds(build_idx, access_full, build_coverage, r_required, trim_margin_multiplier)
    n_kept = len(keep_idx)
    trim_coverage = access_full[keep_idx].sum(axis=0)
    trim_worst = float(trim_coverage.min())
    log(f"Trim phase : removed {n_built - n_kept} seed(s), {n_built} -> {n_kept}  "
        f"(floor: worst-slot margin >= {trim_margin_multiplier:.2f}x requirement = "
        f"{trim_margin_multiplier * r_required:.1f})")
    log(f"  Worst-slot margin after trim: {trim_worst:.1f}  ({trim_worst / r_required:.2f}x requirement)")
    for rank, idx in enumerate(keep_idx):
        f = families[seed_meta[idx]["family"]]
        log(f"  {rank + 1:3d}: {f['N_P']}:{f['N_D']}  i={f['inc_deg']:6.2f}  alt={f['alt_km']:6.1f}  "
            f"RAAN={seed_meta[idx]['raan_deg']:6.2f}  access/target={seed_profiles[idx].sum(axis=0).mean():.1f}")

    profiles_keep = [seed_profiles[i] for i in keep_idx]
    layers_keep = [seed_layers[i] for i in keep_idx]
    meta_keep = [seed_meta[i] for i in keep_idx]
    cols_per_seed = [cols_all[i] for i in keep_idx]

    param_V = build_param_V(profiles_keep, cols_per_seed)
    log(f"\nparam_V shape: {param_V.shape[0]} rows x {param_V.shape[1]} cols "
        f"({n_kept} seeds, {sum(cols_per_seed)} candidate satellites, nnz={int(param_V.nnz)})")
    margin_all = float((param_V @ np.ones(param_V.shape[1])).min())
    ok = margin_all >= r_required
    log(f"Feasibility check (all selected slots on): worst target/time coverage = {margin_all:.0f}  "
        f"requirement = {r_required}  " + ("OK" if ok else "INFEASIBLE"))
    if not ok:
        raise ValueError(
            f"Requirement r_required={r_required} cannot be met even with every selected seed's slots on "
            f"(worst-case coverage {margin_all:.0f}). Either the seed families cannot cover these targets, or "
            "the greedy proxy over-counted a reducible family's restricted slots -- raise the trim/margin multipliers."
        )

    return dict(
        seed_profiles=profiles_keep, seed_layers=layers_keep, seed_meta=meta_keep,
        cols_per_seed=cols_per_seed, param_V=param_V,
        n_simulated=n_total, n_dead=n_dead, n_live=n_live, n_built=n_built, n_kept=n_kept,
        n_trimmed=n_built - n_kept, build_worst_margin=build_worst, trim_worst_margin=trim_worst,
        feasibility_margin_all_slots=margin_all,
    )


def family_satellite_counts(selected_slots, cols_per_seed: list[int], seed_meta: list[dict], families: list[dict]) -> dict:
    """Number of selected satellites per family ('N_P:N_D' -> count)."""
    col_off = np.concatenate([[0], np.cumsum(cols_per_seed)])
    sel_seed = np.searchsorted(col_off, np.asarray(selected_slots), side="right") - 1
    sel_fam = np.array([seed_meta[s]["family"] for s in sel_seed], dtype=int)
    return {f"{f['N_P']}:{f['N_D']}": int((sel_fam == fi).sum()) for fi, f in enumerate(families)}


def slot_columns(seed_meta: list[dict], cols_per_seed: list[int], families: list[dict]) -> dict[str, list]:
    """Per-candidate-slot labels (aligned with the columns of param_V) for the result CSV:
    index of the kept seed, family 'N_P:N_D', and the seed's RAAN offset."""
    seed_index, family, raan = [], [], []
    for s, (m, nc) in enumerate(zip(seed_meta, cols_per_seed)):
        f = families[m["family"]]
        seed_index += [s] * nc
        family += [f"{f['N_P']}:{f['N_D']}"] * nc
        raan += [round(m["raan_deg"], 6)] * nc
    return {"seed_index": seed_index, "family": family, "seed_raan_deg": raan}


# ---------------------------------------------------------------------------
# run_config block
# ---------------------------------------------------------------------------

def seed_run_config(
    plan: dict, pool: dict, n_raan_offsets: list[int], per_family_counts: dict, *,
    raan_step_deg: float, margin_multiplier: float, trim_margin_multiplier: float, max_seeds: int,
    seed_sim_time_s: float | None = None,
) -> dict:
    """Orbit-model / seed-generation / seed-filter entries of the result JSON's run_config."""
    families, L = plan["families"], plan["L"]
    anchor = plan["anchor"]
    return {
        # --- Orbit model / shared repeat period ---
        "orbit_model": "lee_j2",
        "nd_repeat": int(plan["N_D"]),
        "altitude_bounds_km": [float(plan["alt_bounds_km"][0]), float(plan["alt_bounds_km"][1])],
        "anchor_lat_margin_deg": float(plan["anchor_lat_margin_deg"]),
        "anchor_alt_km": float(anchor["alt_km"]),
        "anchor_inc_deg": float(anchor["inc_deg"]),
        "anchor_ratio": f"{anchor['N_P']}:{plan['N_D']}",
        "T_r_s": float(plan["T_r_s"]),
        "L": int(L),
        "dt_s": float(plan["dt_s"]),          # actual time step T_r / L (the nominal one is dt_target_s)
        "families": [
            {
                "N_P": f["N_P"], "N_D": f["N_D"], "reducible": bool(f["reducible"]),
                "reduced_ratio": f["reduced_ratio"], "is_anchor": bool(f["is_anchor"]),
                "inc_deg": f["inc_deg"], "alt_km": f["alt_km"], "a_km": f["a_km"],
                "reach_deg": f["reach_deg"], "inc_floor_deg": f["inc_floor_deg"],
                "raan_interval_deg": f["raan_interval_deg"], "n_raan_offsets": int(n_raan_offsets[fi]),
                "n_distinct_slots": n_distinct_slots(f, L),
                "satellites_selected": per_family_counts[f"{f['N_P']}:{f['N_D']}"],
            }
            for fi, f in enumerate(families)
        ],
        # --- Seed generation summary (target latitudes, counts of altitude/inclination combinations) ---
        "seed_generation": {**plan["generation"], "n_families_kept": len(families)},
        # --- Seed pool, filter and MILP size ---
        "seed_filter": {
            "raan_step_deg": float(raan_step_deg),
            "margin_multiplier": float(margin_multiplier),
            "trim_margin_multiplier": float(trim_margin_multiplier),
            "max_seeds": int(max_seeds),
            "n_seeds_simulated": int(pool["n_simulated"]),
            "n_seeds_dead": int(pool["n_dead"]),
            "n_seeds_live": int(pool["n_live"]),
            "n_seeds_built": int(pool["n_built"]),
            "n_seeds_trimmed": int(pool["n_trimmed"]),
            "n_seeds_kept": int(pool["n_kept"]),
            "build_worst_margin": float(pool["build_worst_margin"]),
            "trim_worst_margin": float(pool["trim_worst_margin"]),
            "feasibility_margin_all_slots": float(pool["feasibility_margin_all_slots"]),
            "n_candidate_satellites": int(sum(pool["cols_per_seed"])),
            "param_V_nnz": int(pool["param_V"].nnz),
            "seed_sim_time_s": None if seed_sim_time_s is None else float(seed_sim_time_s),
        },
        # Seeds that went into the MILP (not the same as the satellites it finally selected).
        "seeds_kept": [
            {
                "rank": rank + 1,
                "family": f"{families[m['family']]['N_P']}:{families[m['family']]['N_D']}",
                "inc_deg": float(families[m["family"]]["inc_deg"]),
                "alt_km": float(families[m["family"]]["alt_km"]),
                "raan_deg": float(m["raan_deg"]),
            }
            for rank, m in enumerate(pool["seed_meta"])
        ],
    }
