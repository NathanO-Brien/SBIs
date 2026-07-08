"""Lattice-flower-constellation (LFC-style) layer generator."""
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
        raise ValueError("LFC layer: n_sats_total must be > 0")
    if P <= 0:
        raise ValueError("LFC layer: n_planes must be > 0")
    if T % P != 0:
        raise ValueError(
            f"LFC layer: n_sats_total must be divisible by n_planes. Got T={T}, P={P}."
        )
    S = T // P
    if S <= 0:
        raise ValueError("LFC layer: computed sats_per_plane must be > 0")
    return T, P, S


def _repeat_period_s(
    *,
    earth: EarthConstants,
    repeat_orbits: int,
    repeat_days: int,
) -> float:
    """Orbit period satisfying the repeat condition: repeat_orbits revolutions
    in repeat_days sidereal days."""
    if int(repeat_orbits) <= 0:
        raise ValueError("LFC layer: repeat_orbits must be > 0")
    if int(repeat_days) <= 0:
        raise ValueError("LFC layer: repeat_days must be > 0")
    sidereal_day_s = (2.0 * np.pi) / float(earth.omega_earth_rad_s)
    return float(repeat_days) * sidereal_day_s / float(repeat_orbits)


def _semi_major_axis_from_period_km(*, earth: EarthConstants, period_s: float) -> float:
    """Invert Kepler's third law: a = (mu * (T / 2pi)^2)^(1/3)."""
    if period_s <= 0.0:
        raise ValueError("LFC layer: derived period must be > 0")
    return float((earth.mu_km3_s2 * (period_s / (2.0 * np.pi)) ** 2) ** (1.0 / 3.0))


def lfc_layer(
    earth: EarthConstants,
    *,
    cov: CoverageConfig,
    n_sats_total: int,
    n_planes: int,
    repeat_orbits: int,
    repeat_days: int = 1,
    inc_deg: float,
    ecc: float = 0.0,
    argp_deg: float = 0.0,
    lattice_shift: int = 1,
    slot_offset_deg: float = 0.0,
    raan_offset_deg: float = 0.0,
    m0_offset_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
) -> Layer:
    """
    Build a first-pass lattice-flower-constellation (LFC-style) layer.

    Notes:
      - This is a practical LFC-style generator, not a full literature-faithful
        closed-form lattice implementation.
      - It differs from the current RGT layer by making eccentricity and argument of perigee
        first-class shaping inputs and by applying an explicit lattice slot shift between planes.
      - The repeat condition defines the common orbit family; the lattice_shift / slot_offset
        define how satellites occupy the repeating lattice.
    """
    T, P, S = _validate_tp(n_sats_total=n_sats_total, n_planes=n_planes)

    e = float(ecc)
    if e < 0.0 or e >= 1.0:
        raise ValueError("LFC layer: ecc must satisfy 0 <= ecc < 1")

    period_s = _repeat_period_s(
        earth=earth,
        repeat_orbits=int(repeat_orbits),
        repeat_days=int(repeat_days),
    )
    a_km = _semi_major_axis_from_period_km(earth=earth, period_s=period_s)
    constellation_alt_km = float(a_km - earth.r_eq_km)
    if constellation_alt_km <= 0.0:
        raise ValueError(
            "LFC layer: derived non-positive altitude. Choose a different repeat_orbits/repeat_days pair."
        )
    inc_deg = float(inc_deg)

    plane_idx = np.repeat(np.arange(P), S)
    sat_idx = np.tile(np.arange(S), P)

    raan_deg = (360.0 / P) * plane_idx + float(raan_offset_deg)

    # LFC-style lattice slotting:
    #   - satellites are uniformly spaced within a plane
    #   - each plane receives an integer lattice shift in slot space
    #   - slot_offset_deg enables a global phase rotation of the lattice
    plane_slot_shift_deg = (int(lattice_shift) * plane_idx) * (360.0 / T)
    in_plane_slot_deg = (360.0 / S) * sat_idx
    M_deg = in_plane_slot_deg + plane_slot_shift_deg + float(slot_offset_deg) + float(m0_offset_deg)

    elems = OrbitalElements(
        a_km=np.full(T, a_km, dtype=np.float64),
        e=np.full(T, e, dtype=np.float64),
        i_rad=np.deg2rad(np.full(T, inc_deg, dtype=np.float64)),
        raan_rad=np.deg2rad(raan_deg.astype(np.float64)),
        argp_rad=np.deg2rad(np.full(T, float(argp_deg), dtype=np.float64)),
        M0_rad=np.deg2rad(M_deg.astype(np.float64)),
    )

    phys = SatellitePhysical(
        bc_kg_m2=np.full(T, float(bc_kg_m2), dtype=np.float64)
    )

    spec = {
        "type": "lfc",
        "method": "first_pass_lattice_flower",
        "notation": "T/P + repeat_orbits/repeat_days + lattice_shift",
        "n_total": int(T),
        "n_planes": int(P),
        "sats_per_plane": int(S),
        "repeat_orbits": int(repeat_orbits),
        "repeat_days": int(repeat_days),
        "repeat_period_s": float(period_s),
        "constellation_alt_km": float(constellation_alt_km),
        "a_km": float(a_km),
        "inc_deg": float(inc_deg),
        "e": float(e),
        "argp_deg": float(argp_deg),
        "intercept_alt_km": float(cov.intercept_alt_km),
        "max_range_km": float(cov.max_range_km),
        "min_elev_deg": float(cov.min_elev_deg),
        "lattice_shift": int(lattice_shift),
        "slot_offset_deg": float(slot_offset_deg),
        "raan_offset_deg": float(raan_offset_deg),
        "m0_offset_deg": float(m0_offset_deg),
        "bc_kg_m2": float(bc_kg_m2),
    }

    return Layer(elems=elems, phys=phys, spec=spec)
