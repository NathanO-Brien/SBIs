"""Synchronized exact-J2 seed families, anchored top-down from the target set.

With J2 on, every seed family must share one repeat period T_r for the
multi-sub-constellation circulant BILP to be valid (Lee et al. 2020, Eq. 33).
Under Lee's secular J2 model (circular orbits) a family is an (N_P, N_D, i, a)
tuple that satisfies the ground-track closure (Eq. 6)

    N_P (w_E - Omega_dot) = N_D (omega_dot + M_dot)

and its repeat period is T_r = N_D * 2*pi / (w_E - Omega_dot).  T_r depends on
(a, i) only through Omega_dot = -k n cos(i), so two families share T_r exactly
when they share Omega_dot.  Given the anchor family's Omega_dot*, each other N_P'
has a unique (a', i') on the shared-Omega_dot curve: cos(i') = -Omega_dot*/(k n)
is closed-form in a', and closure is then a 1-D root in a'.  Higher a' gives lower
i', so the anchor (highest inclination) is also the lowest-altitude family.

Altitude bounds: the lower bound is fixed at 200 km (the intercept shell altitude
by default).  There is NO hard-coded upper bound: the upper bound is the
altitude beyond which no seed can have the target shell within interceptor
range, h_max = intercept_alt + R_max, with R_max taken from the scenario's
CoverageConfig (cov.max_range_km).  Seeds near h_max see almost none of the shell
and are expected to be pruned by the seed filter, so no further margin is applied.

Anchor choice (top-down): i_top = lat_max - anchor_lat_margin_deg (default 0: the
anchor sits at the highest target latitude and the seed filter prunes families
that turn out not to be worth it).  For each candidate anchor N_P the closure
fixes the anchor altitude; the candidate whose synchronized members (altitude
within bounds, inclination between the useful floor and i_top) are most numerous
wins; ties go to the lower altitude.

Useful floor: a family at inclination i with ground reach R(h) (central angle,
see footprint_reach_deg) can only touch a target if i + R(h) >= lat_min, so a
member is kept when i >= lat_min - R(h) at its own altitude.  R comes from the
real coverage geometry when a CoverageConfig is supplied; otherwise the fixed
`useful_reach_deg` fallback is used.
"""
from __future__ import annotations

from math import gcd
from typing import Sequence

import numpy as np
from scipy.optimize import brentq

from ..core.config import CoverageConfig, EarthConstants
from .rgt_slots import a_km_j2_rgt, lee_j2_rates, rgt_closure_residual

DEFAULT_ND = 3
DEFAULT_ALT_MIN_KM = 200.0
DEFAULT_ANCHOR_LAT_MARGIN_DEG = 0.0
DEFAULT_USEFUL_REACH_DEG = 5.0
DEFAULT_DT_TARGET_S = 120.0


def resolve_alt_bounds(
    cov: CoverageConfig | None,
    alt_min_km: float = DEFAULT_ALT_MIN_KM,
    alt_max_km: float | None = None,
) -> tuple[float, float]:
    """(h_min, h_max) in km.  h_max defaults to intercept_alt + R_max (scenario-native)."""
    if alt_max_km is None:
        if cov is None:
            raise ValueError("alt_max_km not given and no CoverageConfig to derive h_max = h_s + R_max from")
        alt_max_km = float(cov.intercept_alt_km) + float(cov.max_range_km)
    if not alt_max_km > alt_min_km:
        raise ValueError(f"altitude upper bound {alt_max_km:.1f} km must exceed the lower bound {alt_min_km:.1f} km")
    return float(alt_min_km), float(alt_max_km)


def footprint_reach_deg(alt_km: float, cov: CoverageConfig, earth: EarthConstants) -> float:
    """Maximum central angle (deg) between a satellite's sub-point and a covered shell point.

    Uses exactly the coverage test of coverage_calc.coverage_counts_ecef: a point on
    the intercept shell is covered when range <= cov.max_range_km AND the satellite's
    elevation above the point's local horizon is >= cov.min_elev_deg.  Both limits
    tighten monotonically with central angle lambda, so the reach is the smaller of
    the two bisection roots.  Returns 0 when even the sub-satellite point is out of range.
    """
    r_s = earth.r_eq_km + float(alt_km)
    r_p = earth.r_eq_km + float(cov.intercept_alt_km)
    sin_min = np.sin(np.deg2rad(cov.min_elev_deg))

    def ok(lam: float) -> bool:
        rho = np.sqrt(r_s * r_s + r_p * r_p - 2.0 * r_s * r_p * np.cos(lam))
        if rho > cov.max_range_km:
            return False
        sin_el = (r_s * np.cos(lam) - r_p) / max(rho, 1e-12)
        return sin_el >= sin_min

    if not ok(0.0):
        return 0.0
    lo, hi = 0.0, np.pi
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if ok(mid):
            lo = mid
        else:
            hi = mid
    return float(np.rad2deg(lo))


def _np_candidates(N_D: int, alt_bounds_km: tuple[float, float], earth: EarthConstants) -> range:
    """N_P scan range that brackets the altitude bounds.

    N_P / N_D is (to within J2 corrections of a few percent) the number of orbits per
    sidereal day, T_sid / T_kepler(h); the range is that ratio at the upper and lower
    altitude bound times N_D, padded by 3 on each side.
    """
    T_sid = 2.0 * np.pi / earth.omega_earth_rad_s

    def nP(h_km: float) -> float:
        a = earth.r_eq_km + h_km
        return N_D * T_sid / (2.0 * np.pi * np.sqrt(a ** 3 / earth.mu_km3_s2))

    lo = max(1, int(np.floor(nP(alt_bounds_km[1]))) - 3)
    hi = int(np.ceil(nP(alt_bounds_km[0]))) + 3
    return range(lo, hi + 1)


def _shared_omega_dot_residual(a_km: float, N_P: int, N_D: int, omega_dot_star: float,
                               earth: EarthConstants) -> float:
    """Closure residual on the curve Omega_dot(a, i(a)) = omega_dot_star.

    Returns NaN where no real inclination exists (|cos i| > 1) or cos i <= 0.
    """
    n = np.sqrt(earth.mu_km3_s2 / a_km ** 3)
    kn = 1.5 * earth.j2 * (earth.r_eq_km / a_km) ** 2 * n
    cos_i = -omega_dot_star / kn
    if not (0.0 < cos_i <= 1.0):
        return float("nan")
    inc_deg = float(np.rad2deg(np.arccos(cos_i)))
    return rgt_closure_residual(a_km, N_P, N_D, inc_deg, earth)


def solve_member(N_P: int, N_D: int, omega_dot_star: float, earth: EarthConstants,
                 alt_bounds_km: tuple[float, float], allow_reducible: bool = True) -> dict | None:
    """Unique (a, i) for N_P:N_D with Omega_dot == omega_dot_star inside the altitude bounds.

    Reducible ratios (e.g. 45:3 = 15:1) satisfy the same closure and share T_r with
    the N_D-day grid (their track simply repeats every T_r / gcd), so they are valid
    rungs when allow_reducible is True.
    """
    if not allow_reducible and gcd(N_P, N_D) != 1:
        return None
    n_grid = int(max(801, 2 * (alt_bounds_km[1] - alt_bounds_km[0])))
    alts = np.linspace(alt_bounds_km[0], alt_bounds_km[1], n_grid)
    g = np.array([_shared_omega_dot_residual(earth.r_eq_km + h, N_P, N_D, omega_dot_star, earth)
                  for h in alts])
    root_alt = None
    for j in range(len(alts) - 1):
        if np.isfinite(g[j]) and np.isfinite(g[j + 1]) and g[j] * g[j + 1] <= 0.0:
            root_alt = brentq(
                lambda h: _shared_omega_dot_residual(earth.r_eq_km + h, N_P, N_D, omega_dot_star, earth),
                alts[j], alts[j + 1], xtol=1e-10, rtol=1e-14,
            )
            break
    if root_alt is None:
        return None
    return _family_record(N_P, N_D, earth.r_eq_km + root_alt, None, earth, omega_dot_star)


def _family_record(N_P: int, N_D: int, a_km: float, inc_deg: float | None,
                   earth: EarthConstants, omega_dot_star: float | None = None) -> dict:
    if inc_deg is None:
        n = np.sqrt(earth.mu_km3_s2 / a_km ** 3)
        kn = 1.5 * earth.j2 * (earth.r_eq_km / a_km) ** 2 * n
        inc_deg = float(np.rad2deg(np.arccos(-omega_dot_star / kn)))
    Om, om, Md = lee_j2_rates(a_km, inc_deg, earth)
    T_r = N_D * 2.0 * np.pi / (earth.omega_earth_rad_s - Om)
    return dict(
        N_P=int(N_P), N_D=int(N_D), inc_deg=float(inc_deg), a_km=float(a_km),
        alt_km=float(a_km - earth.r_eq_km),
        Omegadot_rad_s=Om, omegadot_rad_s=om, Mdot_rad_s=Md, T_r_s=float(T_r),
        raan_interval_deg=360.0 / (N_P // gcd(N_P, N_D)),  # distinct-track RAAN window uses the REDUCED N_P
        reducible=bool(gcd(N_P, N_D) != 1),
        reduced_ratio=f"{N_P // gcd(N_P, N_D)}:{N_D // gcd(N_P, N_D)}",
        repeat_factor=int(gcd(N_P, N_D)),   # g: the ground track repeats every T_r / g
        closure_residual_rad_s=float(rgt_closure_residual(a_km, N_P, N_D, inc_deg, earth)),
    )


def _combo_key(f: dict) -> tuple[float, float]:
    """Identity of an (altitude, inclination) combination for uniqueness counts."""
    return (round(float(f["alt_km"]), 3), round(float(f["inc_deg"]), 4))


def plan_seed_families(
    target_points: Sequence[Sequence],
    *,
    earth: EarthConstants | None = None,
    N_D: int = DEFAULT_ND,
    alt_min_km: float = DEFAULT_ALT_MIN_KM,
    alt_max_km: float | None = None,
    anchor_lat_margin_deg: float = DEFAULT_ANCHOR_LAT_MARGIN_DEG,
    useful_reach_deg: float = DEFAULT_USEFUL_REACH_DEG,
    cov: CoverageConfig | None = None,
    allow_reducible: bool = True,
    dt_target_s: float = DEFAULT_DT_TARGET_S,
) -> dict:
    """Plan the synchronized seed families for a target set (top-down anchor).

    target_points : [[name, lat_deg, lon_deg], ...]
    alt_min_km    : lower altitude bound (default 200 km).
    alt_max_km    : upper altitude bound; default None -> intercept_alt + R_max from `cov`.

    Returns dict(anchor, families, T_r_s, L, dt_s, i_top_deg, inc_floor_deg (lowest
    per-family floor), alt_bounds_km, n_anchor_candidates, plus a `generation` dict
    describing the seed generation (target latitude statistics and counts of the unique
    altitude/inclination combinations generated, kept and dropped)).  Each family also
    carries reach_deg and inc_floor_deg.  `families` is sorted by inclination
    descending; families[0] is the anchor.
    """
    earth = earth or EarthConstants()
    alt_bounds = resolve_alt_bounds(cov, alt_min_km, alt_max_km)
    signed = np.array([float(p[1]) for p in target_points], dtype=np.float64)
    if signed.size == 0:
        raise ValueError("target_points is empty")
    lats = np.abs(signed)
    lat_max, lat_min = float(lats.max()), float(lats.min())
    i_top = lat_max - anchor_lat_margin_deg
    if i_top <= 0.0:
        raise ValueError(f"anchor inclination {i_top:.2f} deg is not positive (lat_max={lat_max:.2f})")

    def _reach(alt_km: float) -> float:
        return footprint_reach_deg(alt_km, cov, earth) if cov is not None else float(useful_reach_deg)

    best = None
    n_cand = 0
    anchor_candidates: list[dict] = []
    all_combos: set[tuple[float, float]] = set()
    for N_P_a in _np_candidates(N_D, alt_bounds, earth):
        if not allow_reducible and gcd(N_P_a, N_D) != 1:
            continue
        a_a = a_km_j2_rgt(N_P_a, N_D, i_top, earth)
        alt_a = a_a - earth.r_eq_km
        if not (alt_bounds[0] <= alt_a <= alt_bounds[1]):
            continue
        if abs(rgt_closure_residual(a_a, N_P_a, N_D, i_top, earth)) > 1e-12:
            continue
        n_cand += 1
        anchor = _family_record(N_P_a, N_D, a_a, i_top, earth)
        anchor["reach_deg"] = _reach(anchor["alt_km"])
        anchor["inc_floor_deg"] = max(0.0, lat_min - anchor["reach_deg"])
        generated = [anchor]
        members = []
        dropped = []
        for N_P_m in _np_candidates(N_D, alt_bounds, earth):
            if N_P_m == N_P_a or (not allow_reducible and gcd(N_P_m, N_D) != 1):
                continue
            m = solve_member(N_P_m, N_D, anchor["Omegadot_rad_s"], earth, alt_bounds, allow_reducible)
            if m is None:
                continue
            m["reach_deg"] = _reach(m["alt_km"])
            m["inc_floor_deg"] = max(0.0, lat_min - m["reach_deg"])
            generated.append(m)
            if m["inc_deg"] >= i_top - 1e-9:
                dropped.append((m, "above_anchor_inclination"))
            elif m["inc_deg"] < m["inc_floor_deg"]:
                dropped.append((m, "below_reach_floor"))
            else:
                members.append(m)
        for f in generated:
            all_combos.add(_combo_key(f))
        anchor_candidates.append(dict(N_P=int(N_P_a), alt_km=float(alt_a), n_rungs_generated=len(generated),
                                      n_members_kept=len(members)))
        key = (len(members), -alt_a)
        if best is None or key > best[0]:
            best = (key, anchor, members, generated, dropped)

    if best is None:
        raise ValueError(
            f"No anchor family exists at i={i_top:.2f} deg within {alt_bounds} km for N_D={N_D}"
        )
    _, anchor, members, generated, dropped = best
    anchor["is_anchor"] = True
    fams = [anchor] + sorted(members, key=lambda f: -f["inc_deg"])
    for f in fams[1:]:
        f["is_anchor"] = False
    T_r = anchor["T_r_s"]
    L = int(round(T_r / dt_target_s))
    generation = dict(
        # target-set latitude statistics
        target_lat_min_deg=float(signed.min()), target_lat_max_deg=float(signed.max()),
        target_lat_span_deg=float(signed.max() - signed.min()),
        target_abs_lat_min_deg=lat_min, target_abs_lat_max_deg=lat_max,
        target_abs_lat_span_deg=float(lat_max - lat_min),
        n_target_points=int(signed.size),
        # altitude / anchor rules
        alt_min_km=float(alt_bounds[0]), alt_max_km=float(alt_bounds[1]),
        alt_max_rule="intercept_alt + R_max" if alt_max_km is None else "explicit",
        r_max_km=None if cov is None else float(cov.max_range_km),
        anchor_inc_deg=float(i_top), anchor_lat_margin_deg=float(anchor_lat_margin_deg),
        # counts of generated altitude/inclination combinations
        n_anchor_candidates=int(n_cand),
        n_unique_alt_inc_combos_all_candidates=int(len(all_combos)),
        n_unique_alt_inc_combos_chosen_ladder=int(len({_combo_key(f) for f in generated})),
        n_unique_alt_inc_combos_kept=int(len({_combo_key(f) for f in fams})),
        n_combos_dropped=int(len(dropped)),
        dropped_combos=[
            dict(N_P=int(m["N_P"]), N_D=int(m["N_D"]), inc_deg=float(m["inc_deg"]), alt_km=float(m["alt_km"]), reason=why)
            for m, why in dropped
        ],
        anchor_candidates=anchor_candidates,
    )
    return dict(
        anchor=anchor, families=fams, T_r_s=float(T_r), L=L, dt_s=float(T_r / L),
        N_D=int(N_D), i_top_deg=float(i_top),
        inc_floor_deg=float(min(f["inc_floor_deg"] for f in fams)),
        lat_max_deg=lat_max, lat_min_deg=lat_min, alt_bounds_km=tuple(alt_bounds),
        anchor_lat_margin_deg=float(anchor_lat_margin_deg), useful_reach_deg=float(useful_reach_deg),
        n_anchor_candidates=n_cand, generation=generation,
    )


def ladder_from_pin(
    N_P_pin: int,
    pin_inc_deg: float,
    *,
    earth: EarthConstants | None = None,
    N_D: int = DEFAULT_ND,
    alt_min_km: float = DEFAULT_ALT_MIN_KM,
    alt_max_km: float | None = None,
    cov: CoverageConfig | None = None,
    lat_min_deg: float | None = None,
    allow_reducible: bool = True,
) -> list[dict]:
    """All synchronized rungs (any inclination) for the ladder that passes through
    (N_P_pin, pin_inc_deg).  If cov and lat_min_deg are given, rungs whose inclination
    is below lat_min - reach(alt) are dropped.  Sorted by inclination descending."""
    earth = earth or EarthConstants()
    alt_bounds = resolve_alt_bounds(cov, alt_min_km, alt_max_km)
    a_p = a_km_j2_rgt(N_P_pin, N_D, pin_inc_deg, earth)
    if not (alt_bounds[0] <= a_p - earth.r_eq_km <= alt_bounds[1]):
        return []
    pin = _family_record(N_P_pin, N_D, a_p, pin_inc_deg, earth)
    rungs = [pin]
    for N_P_m in _np_candidates(N_D, alt_bounds, earth):
        if N_P_m == N_P_pin:
            continue
        m = solve_member(N_P_m, N_D, pin["Omegadot_rad_s"], earth, alt_bounds, allow_reducible)
        if m is not None:
            rungs.append(m)
    for r in rungs:
        r["reach_deg"] = footprint_reach_deg(r["alt_km"], cov, earth) if cov is not None else DEFAULT_USEFUL_REACH_DEG
        r["inc_floor_deg"] = max(0.0, (lat_min_deg or 0.0) - r["reach_deg"])
    if lat_min_deg is not None:
        rungs = [r for r in rungs if r["inc_deg"] >= r["inc_floor_deg"]]
    return sorted(rungs, key=lambda r: -r["inc_deg"])


def n_distinct_slots(family: dict, L: int) -> int:
    """Distinct satellite positions among a family's L time-shift slots.

    A family with repeat_factor g = gcd(N_P, N_D) > 1 has a ground track that
    repeats every T_r / g, so slot k + L/g sits at slot k's orbital position and
    only the first floor(L / g) slots are distinct.  g == 1 keeps all L.
    """
    return int(L // family.get("repeat_factor", 1))
