"""Walker layer generators that automatically size plane/satellite counts
from coverage geometry.
"""
from __future__ import annotations

import numpy as np
from ..core.config import EarthConstants
from ..core.elements import OrbitalElements, SatellitePhysical
from .layer import Layer

def auto_walker_delta(
    earth: EarthConstants,
    *,
    intercept_radius_km: float,
    intercept_alt_km: float,
    constellation_alt_km: float,
    inc_deg: float,
    f: int = 1,
    ecc: float = 0.0,
    argp_deg: float = 0.0,
    raan_offset_deg: float = 0.0,
    m0_offset_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
) -> Layer:
    """Classic Walker-Δ baseline, auto-sized from interceptor geometry (Option C, hex spacing).

    This is NOT Wright-optimized. It is a physics-sized baseline constellation meant
    for fair comparison against optimized designs.

    Sizing:
      - target spacing d = sqrt(3) * r_h
      - r_h = sqrt(r^2 - (h-h_int)^2)
    """

    # ---- geometry ----
    r = float(intercept_radius_km)
    h_int = float(intercept_alt_km)
    h = float(constellation_alt_km)
    inc_deg = float(inc_deg)

    dz = h - h_int
    inside = r * r - dz * dz
    if inside <= 0:
        raise ValueError("walker_delta: intercept geometry invalid (r^2 <= (h-h_int)^2)")
    r_h = np.sqrt(inside)

    d_target = np.sqrt(3.0) * r_h
    R_shell = earth.r_eq_km + h

    S = int(np.ceil(2.0 * np.pi * R_shell / d_target))
    P = int(np.ceil(2.0 * np.pi * R_shell / d_target))
    T = P * S

    # ---- indices ----
    plane_idx = np.repeat(np.arange(P), S)
    sat_idx = np.tile(np.arange(S), P)

    # ---- elements ----
    a_km = earth.r_eq_km + h

    raan_deg = (360.0 / P) * plane_idx + raan_offset_deg
    phase_deg = (f * plane_idx) * (360.0 / T)
    M_deg = (360.0 / S) * sat_idx + phase_deg + m0_offset_deg

    elems = OrbitalElements(
        a_km=np.full(T, a_km),
        e=np.full(T, ecc),
        i_rad=np.deg2rad(np.full(T, inc_deg)),
        raan_rad=np.deg2rad(raan_deg),
        argp_rad=np.deg2rad(np.full(T, argp_deg)),
        M0_rad=np.deg2rad(M_deg),
    )

    phys = SatellitePhysical(
        bc_kg_m2=np.full(T, bc_kg_m2)
    )

    spec = {
        "type": "walker_delta",
        "sizing": "option_c_hex",
        "n_planes": P,
        "sats_per_plane": S,
        "n_total": T,
        "d_target_km": float(d_target),
        "coverage_disc_radius_km": float(r_h),
        "intercept_radius_km": r,
        "intercept_alt_km": h_int,
        "constellation_alt_km": h,
        "inc_deg": inc_deg,
        "f": f,
    }

    return Layer(elems=elems, phys=phys, spec=spec)

def auto_walker_star(
    earth: EarthConstants,
    *,
    intercept_radius_km: float,
    intercept_alt_km: float,
    constellation_alt_km: float,
    inc_deg: float = 90.0,
    ecc: float = 0.001,
    argp_deg: float = 0.0,
    raan_offset_deg: float = 0.0,
    m0_offset_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
) -> Layer:
    """Classic Walker-Star baseline, auto-sized from interceptor geometry (Option C, hex spacing).

    Fixes epoch clumping by using a continuous plane-dependent along-track offset:
        M(p,s) = 360/S * (s + p/P) + M0_offset

    This avoids the 2-class (even/odd plane) aliasing that can make satellites collapse
    onto a single latitude ring at t=0.
    """

    # ---- geometry / Option C sizing ----
    r = float(intercept_radius_km)
    h_int = float(intercept_alt_km)
    h = float(constellation_alt_km)
    inc_deg = float(inc_deg)

    dz = h - h_int
    inside = r * r - dz * dz
    if inside <= 0:
        raise ValueError("walker_star: intercept geometry invalid (r^2 <= (h-h_int)^2)")
    r_h = np.sqrt(inside)

    d_target = np.sqrt(3.0) * r_h
    R_shell = earth.r_eq_km + h

    S = int(np.ceil(2.0 * np.pi * R_shell / d_target))
    P = int(np.ceil(2.0 * np.pi * R_shell / d_target))

    # Enforce even planes for symmetry (optional but nice for "star" look)
    if P % 2 != 0:
        P += 1

    T = P * S

    # ---- indices ----
    plane_idx = np.repeat(np.arange(P), S)
    sat_idx = np.tile(np.arange(S), P)

    # ---- elements ----
    a_km = earth.r_eq_km + h

    raan_deg = (360.0 / P) * plane_idx + float(raan_offset_deg)

    # Key fix: continuous plane offset (not cumulative Walker f)
    # offset fraction in [0,1) across planes, scaled by one in-plane slot (360/S).
    plane_frac = plane_idx / float(P)
    M_deg = (360.0 / S) * (sat_idx + plane_frac) + float(m0_offset_deg)

    elems = OrbitalElements(
        a_km=np.full(T, a_km, dtype=np.float64),
        e=np.full(T, float(ecc), dtype=np.float64),
        i_rad=np.deg2rad(np.full(T, inc_deg, dtype=np.float64)),
        raan_rad=np.deg2rad(raan_deg),
        argp_rad=np.deg2rad(np.full(T, float(argp_deg), dtype=np.float64)),
        M0_rad=np.deg2rad(M_deg),
    )

    phys = SatellitePhysical(bc_kg_m2=np.full(T, float(bc_kg_m2), dtype=np.float64))

    spec = {
        "type": "walker_star",
        "sizing": "option_c_hex",
        "method": "plane_frac_offset",
        "n_planes": int(P),
        "sats_per_plane": int(S),
        "n_total": int(T),
        "d_target_km": float(d_target),
        "coverage_disc_radius_km": float(r_h),
        "intercept_radius_km": float(r),
        "intercept_alt_km": float(h_int),
        "constellation_alt_km": float(h),
        "inc_deg": float(inc_deg),
        "raan_offset_deg": float(raan_offset_deg),
        "m0_offset_deg": float(m0_offset_deg),
        "argp_deg": float(argp_deg),
        "bc_kg_m2": float(bc_kg_m2),
    }

    return Layer(elems=elems, phys=phys, spec=spec)
