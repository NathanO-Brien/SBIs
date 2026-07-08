"""Repeating-ground-track (RGT) constellation layer generator."""
from __future__ import annotations

import numpy as np

from ..core.config import CoverageConfig, EarthConstants
from ..core.elements import OrbitalElements, SatellitePhysical
from .layer import Layer


def _validate_tp(*, n_sats_total: int, n_planes: int) -> tuple[int, int, int]:
    """Validate T/P divisibility and return (T, P, sats_per_plane)."""
    T = int(n_sats_total)
    P = int(n_planes)
    if T <= 0:
        raise ValueError("RGT layer: n_sats_total must be > 0")
    if P <= 0:
        raise ValueError("RGT layer: n_planes must be > 0")
    if T % P != 0:
        raise ValueError(
            f"RGT layer: n_sats_total must be divisible by n_planes. Got T={T}, P={P}."
        )
    S = T // P
    if S <= 0:
        raise ValueError("RGT layer: computed sats_per_plane must be > 0")
    return T, P, S


def _rgt_period_s(
    *,
    earth: EarthConstants,
    repeat_orbits: int,
    repeat_days: int,
) -> float:
    """Orbit period satisfying the repeat condition: repeat_orbits revolutions
    in repeat_days sidereal days."""
    if int(repeat_orbits) <= 0:
        raise ValueError("RGT layer: repeat_orbits must be > 0")
    if int(repeat_days) <= 0:
        raise ValueError("RGT layer: repeat_days must be > 0")
    sidereal_day_s = (2.0 * np.pi) / float(earth.omega_earth_rad_s)
    return float(repeat_days) * sidereal_day_s / float(repeat_orbits)


def _semi_major_axis_from_period_km(*, earth: EarthConstants, period_s: float) -> float:
    """Invert Kepler's third law: a = (mu * (T / 2pi)^2)^(1/3)."""
    if period_s <= 0.0:
        raise ValueError("RGT layer: derived period must be > 0")
    return float((earth.mu_km3_s2 * (period_s / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0))


def rgt_layer(
    earth: EarthConstants,
    *,
    cov: CoverageConfig,
    n_sats_total: int,
    n_planes: int,
    repeat_orbits: int,
    repeat_days: int = 1,
    inc_deg: float,
    f: int = 1,
    ecc: float = 0.0,
    raan_offset_deg: float = 0.0,
    m0_offset_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
) -> Layer:
    """
    Build a circular / near-circular repeating-ground-track (RGT) layer.

    Inputs:
      - repeat_orbits / repeat_days define the repeat cycle:
            repeat_orbits completed in repeat_days sidereal days
      - n_sats_total / n_planes define population across the repeating family
      - f applies a Walker-like plane phasing shift for a first practical slotting scheme

    Notes:
      - The repeat condition derives the orbit period, semi-major axis, and altitude.
      - Satellites are distributed across planes and along-track slots following
        Walker-style layer generation conventions.
    """
    T, P, S = _validate_tp(n_sats_total=n_sats_total, n_planes=n_planes)

    period_s = _rgt_period_s(
        earth=earth,
        repeat_orbits=int(repeat_orbits),
        repeat_days=int(repeat_days),
    )
    a_km = _semi_major_axis_from_period_km(earth=earth, period_s=period_s)
    constellation_alt_km = float(a_km - earth.r_eq_km)
    if constellation_alt_km <= 0.0:
        raise ValueError(
            "RGT layer: derived non-positive altitude. Choose a different repeat_orbits/repeat_days pair."
        )
    inc_deg = float(inc_deg)

    plane_idx = np.repeat(np.arange(P), S)
    sat_idx = np.tile(np.arange(S), P)

    raan_deg = (360.0 / P) * plane_idx + float(raan_offset_deg)
    phase_deg = (int(f) * plane_idx) * (360.0 / T)
    M_deg = (360.0 / S) * sat_idx + phase_deg + float(m0_offset_deg)

    elems = OrbitalElements(
        a_km=np.full(T, a_km, dtype=np.float64),
        e=np.full(T, float(ecc), dtype=np.float64),
        i_rad=np.deg2rad(np.full(T, inc_deg, dtype=np.float64)),
        raan_rad=np.deg2rad(raan_deg.astype(np.float64)),
        argp_rad=np.zeros(T, dtype=np.float64),
        M0_rad=np.deg2rad(M_deg.astype(np.float64)),
    )

    phys = SatellitePhysical(
        bc_kg_m2=np.full(T, float(bc_kg_m2), dtype=np.float64)
    )

    spec = {
        "type": "rgt",
        "method": "repeat_orbits_repeat_days",
        "notation": "T/P/F + repeat_orbits/repeat_days",
        "n_total": int(T),
        "n_planes": int(P),
        "sats_per_plane": int(S),
        "repeat_orbits": int(repeat_orbits),
        "repeat_days": int(repeat_days),
        "repeat_period_s": float(period_s),
        "constellation_alt_km": float(constellation_alt_km),
        "a_km": float(a_km),
        "inc_deg": float(inc_deg),
        "e": float(ecc),
        "f": int(f),
        "intercept_alt_km": float(cov.intercept_alt_km),
        "max_range_km": float(cov.max_range_km),
        "min_elev_deg": float(cov.min_elev_deg),
        "raan_offset_deg": float(raan_offset_deg),
        "m0_offset_deg": float(m0_offset_deg),
        "argp_deg": 0.0,
        "bc_kg_m2": float(bc_kg_m2),
    }

    return Layer(elems=elems, phys=phys, spec=spec)
