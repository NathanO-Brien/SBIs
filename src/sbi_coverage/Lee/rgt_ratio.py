from __future__ import annotations

from math import gcd

import numpy as np

from ..core.config import EarthConstants


def _orbital_period_s(a_km: float, earth: EarthConstants) -> float:
    return 2.0 * np.pi * np.sqrt(a_km**3 / earth.mu_km3_s2)


def _alt_from_np_nd(N_P: int, N_D: int, earth: EarthConstants) -> float:
    """Altitude (km) for the Keplerian RGT orbit with N_P orbits in N_D sidereal days."""
    T_sid = (2.0 * np.pi) / earth.omega_earth_rad_s
    T_S   = N_D * T_sid / N_P
    a_km  = (earth.mu_km3_s2 * (T_S / (2.0 * np.pi))**2) ** (1.0 / 3.0)
    return float(a_km - earth.r_eq_km)


def candidate_rgt_ratios(
    h_star_km: float,
    earth: EarthConstants,
    *,
    max_repeat_days: int = 7,
    dt_s: float = 120.0,
    max_candidates: int = 10,
) -> list[dict]:
    """Return ranked candidate N_P:N_D pairs whose RGT altitude is closest to h_star_km.

    For each N_D in 1..max_repeat_days, the function finds the N_P value(s) that
    minimise |h(N_P, N_D) - h_star_km| and checks that gcd(N_P, N_D) == 1 (the
    pair must be in lowest terms — a reducible pair is already represented by its
    reduced form with a shorter repeat cycle).

    Parameters
    ----------
    h_star_km       : target altitude from optimal_sat_altitude_km()
    earth           : EarthConstants
    max_repeat_days : search N_D from 1 to this value (inclusive)
    dt_s            : time step size used to compute L = T_r / dt_s
    max_candidates  : cap on returned list length

    Returns
    -------
    List of dicts, sorted ascending by |alt_km - h_star_km|.  Each dict has:
        N_P             : int   — repeat orbits
        N_D             : int   — repeat days
        alt_km          : float — Keplerian altitude for this pair
        alt_error_km    : float — |alt_km - h_star_km|
        T_r_s           : float — repeat period (seconds)
        T_S_s           : float — orbital period (seconds)
        L               : int   — number of time steps in T_r at dt_s
        ratio           : float — N_P / N_D

    Notes
    -----
    Altitudes are computed from the Keplerian period, consistent with rgt_layer() in
    src/sbi_coverage/constellations/rgt.py.  J2 shifts the true nodal period slightly,
    so the real RGT altitude will differ by a few km; treat h_star_km as a target,
    not an exact requirement.
    """
    if max_repeat_days < 1:
        raise ValueError("max_repeat_days must be >= 1")

    T_sid     = (2.0 * np.pi) / earth.omega_earth_rad_s
    a_star    = earth.r_eq_km + h_star_km
    T_S_star  = _orbital_period_s(a_star, earth)
    ideal_ratio = T_sid / T_S_star   # = N_P / N_D target

    seen: set[tuple[int, int]] = set()
    candidates: list[dict] = []

    for N_D in range(1, max_repeat_days + 1):
        # Both the nearest integer and its neighbours to catch near-ties
        N_P_float = ideal_ratio * N_D
        for N_P in {max(1, round(N_P_float)), max(1, int(N_P_float)), int(N_P_float) + 1}:
            if N_P <= 0:
                continue
            if gcd(N_P, N_D) != 1:
                continue                 # reducible — skip
            key = (N_P, N_D)
            if key in seen:
                continue
            seen.add(key)

            alt_km = _alt_from_np_nd(N_P, N_D, earth)
            if alt_km <= 0.0:
                continue

            T_r_s = N_D * T_sid          # total repeat period
            T_S_s = T_r_s / N_P          # orbital period for this pair
            L     = int(round(T_r_s / dt_s))

            candidates.append({
                "N_P":          N_P,
                "N_D":          N_D,
                "alt_km":       round(alt_km, 3),
                "alt_error_km": round(abs(alt_km - h_star_km), 3),
                "T_r_s":        round(T_r_s, 2),
                "T_S_s":        round(T_S_s, 2),
                "L":            L,
                "ratio":        round(N_P / N_D, 6),
            })

    candidates.sort(key=lambda d: d["alt_error_km"])
    return candidates[:max_candidates]