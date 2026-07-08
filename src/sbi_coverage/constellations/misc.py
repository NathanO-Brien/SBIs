"""Miscellaneous constellation layer generators: uniform, random, and
Wright hexagonal-packing layers.
"""
from __future__ import annotations

import numpy as np
from ..core.config import CoverageConfig, EarthConstants
from ..core.elements import OrbitalElements, SatellitePhysical
from .layer import Layer


def uniform_layer(
    earth: EarthConstants,
    *,
    cov: CoverageConfig,
    n_sats: int,
    a_km: float,
    inc_deg: float,
    ecc: float = 0.001,
    argp_deg: float = 0.0,
    raan_offset_deg: float = 0.0,
    m0_offset_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
) -> Layer:
    """
    Uniform layer with evenly spaced RAAN and M0 across all satellites.

    IMPORTANT:
      - `a_km` is SEMI-MAJOR AXIS in km (not altitude).
      - For circular orbits (ecc ~ 0), altitude ~ a_km - Re.
      - For eccentric orbits, perigee/apogee altitudes differ; do NOT interpret a_km
        as a physical altitude.
      - `inc_deg` is used directly as the orbit inclination in degrees.
    """
    n_sats = int(n_sats)
    if n_sats <= 0:
        raise ValueError("uniform_layer: n_sats must be > 0")

    a_km = float(a_km)
    ecc = float(ecc)
    inc_deg = float(inc_deg)
    argp_deg = float(argp_deg)
    raan_offset_deg = float(raan_offset_deg)
    m0_offset_deg = float(m0_offset_deg)
    bc_kg_m2 = float(bc_kg_m2)

    a_arr = np.full(n_sats, a_km, dtype=np.float64)
    e_arr = np.full(n_sats, ecc, dtype=np.float64)
    i_rad = np.deg2rad(np.full(n_sats, inc_deg, dtype=np.float64))
    argp_rad = np.deg2rad(np.full(n_sats, argp_deg, dtype=np.float64))

    raan_rad = np.deg2rad(
        np.linspace(0.0, 360.0, n_sats, endpoint=False, dtype=np.float64) + raan_offset_deg
    )
    M0_rad = np.deg2rad(
        np.linspace(0.0, 360.0, n_sats, endpoint=False, dtype=np.float64) + m0_offset_deg
    )

    elems = OrbitalElements(a_km=a_arr, e=e_arr, i_rad=i_rad, raan_rad=raan_rad, argp_rad=argp_rad, M0_rad=M0_rad)
    phys = SatellitePhysical(bc_kg_m2=np.full(n_sats, bc_kg_m2, dtype=np.float64))

    spec = {
        "type": "uniform_layer",
        "n_sats": n_sats,
        "a_km": a_km,
        "e": ecc,
        "i_deg": inc_deg,
        "intercept_alt_km": float(cov.intercept_alt_km),
        "max_range_km": float(cov.max_range_km),
        "min_elev_deg": float(cov.min_elev_deg),
        "raan_offset_deg": raan_offset_deg,
        "argp_deg": argp_deg,
        "m0_offset_deg": m0_offset_deg,
        "bc_kg_m2": bc_kg_m2,
    }

    return Layer(elems=elems, phys=phys, spec=spec)


def random_layer(
    earth: EarthConstants,
    *,
    cov: CoverageConfig,
    n_sats: int,
    a_km: float,
    inc_deg: float,
    ecc_max: float = 0.01,
    bc_kg_m2: float = 80.0,
    seed: int = 0,
) -> Layer:
    """
    Randomized layer: random RAAN/argp/M0 and random e in [0, ecc_max].

    IMPORTANT:
      - `a_km` is SEMI-MAJOR AXIS in km (not altitude).
      - For a circular shell at altitude h_km, pass a_km = earth.r_eq_km + h_km.

    Notes:
      - Perigee altitude is not constrained; enforcing a minimum perigee
        (e.g. >= 200 km) would require generating e subject to rp = a(1-e).
      - `inc_deg` is used directly as the orbit inclination in degrees.
    """
    n_sats = int(n_sats)
    if n_sats <= 0:
        raise ValueError("random_layer: n_sats must be > 0")

    a_km = float(a_km)
    inc_deg = float(inc_deg)
    ecc_max = float(ecc_max)
    bc_kg_m2 = float(bc_kg_m2)
    seed = int(seed)

    rng = np.random.default_rng(seed)

    a_arr = np.full(n_sats, a_km, dtype=np.float64)
    e_arr = rng.uniform(0.0, ecc_max, size=n_sats).astype(np.float64)
    i_rad = np.deg2rad(np.full(n_sats, inc_deg, dtype=np.float64))

    raan_rad = rng.uniform(0.0, 2.0 * np.pi, size=n_sats).astype(np.float64)
    argp_rad = rng.uniform(0.0, 2.0 * np.pi, size=n_sats).astype(np.float64)
    M0_rad = rng.uniform(0.0, 2.0 * np.pi, size=n_sats).astype(np.float64)

    elems = OrbitalElements(a_km=a_arr, e=e_arr, i_rad=i_rad, raan_rad=raan_rad, argp_rad=argp_rad, M0_rad=M0_rad)
    phys = SatellitePhysical(bc_kg_m2=np.full(n_sats, bc_kg_m2, dtype=np.float64))

    spec = {
        "type": "random_layer",
        "n_sats": n_sats,
        "a_km": a_km,
        "ecc_max": ecc_max,
        "i_deg": inc_deg,
        "intercept_alt_km": float(cov.intercept_alt_km),
        "max_range_km": float(cov.max_range_km),
        "min_elev_deg": float(cov.min_elev_deg),
        "seed": seed,
        "bc_kg_m2": bc_kg_m2,
        "raan": "uniform(0, 2pi)",
        "argp": "uniform(0, 2pi)",
        "M0": "uniform(0, 2pi)",
        "e": f"uniform(0, {ecc_max})",
    }

    return Layer(elems=elems, phys=phys, spec=spec)


def wright_hex(
    earth: EarthConstants,
    *,
    intercept_radius_km: float,
    intercept_alt_km: float,
    constellation_alt_km: float,
    lmax_deg: float,
    lmin_deg: float,
    multiplier: float,
    sat_margin: float = 1.0,
    ecc: float = 0.000,
    argp_deg: float = 0.0,
    raan_offset_deg: float = 0.0,
    m0_offset_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
) -> Layer:
    """Even-coverage Walker-Δ layer sized from SBI intercept geometry (Wright hex lattice).

    This constructs a single-inclination (i = lmax) Walker-style constellation whose
    *coverage-disk centers* form an approximately hexagonally-packed lattice at latitude
    |L| = lmin. The sizing (P planes, S sats/plane) comes from Wright (1/30/26).

    Parameters
    ----------
    earth : EarthConstants
        Earth constants container (must provide r_eq_km).
    intercept_radius_km : float
        The 3D flyout distance r (km) available to the SBI to reach the intercept point.
    intercept_alt_km : float
        Intercept altitude h_int (km).
    constellation_alt_km : float
        Constellation/parking altitude h (km).
    lmax_deg : float
        Maximum latitude of interest, used directly as inclination i (deg).
    lmin_deg : float
        Minimum latitude of interest (deg). Design targets no gaps at |L| = lmin_deg.
    multiplier : float
        Multiplier on the calculated number of satellites per plane. Results in greater defense-in-depth.
    sat_margin : float
        Additional multiplicative margin applied only to satellites-per-plane sizing.
        Use values slightly above 1.0 (e.g., 1.02-1.10) to patch near-seam undercoverage
        without forcing an extra orbital plane.
    ecc : float
        Orbit eccentricity (kept small; default 0.001).
    argp_deg : float
        Argument of perigee (deg). For near-circular orbits this is largely irrelevant.
    raan_offset_deg : float
        Constant RAAN offset applied to all planes (deg).
    m0_offset_deg : float
        Constant mean anomaly offset applied to all satellites (deg).
    bc_kg_m2 : float
        Ballistic coefficient for the satellites (kg/m^2).

    Returns
    -------
    Layer
        Layer(elems, phys, spec) compatible with combine_layers().

    Notes
    -----
    * lmin/lmax are treated as geodetic latitude magnitudes in degrees, symmetric about
      the equator.
    * Counts are ceiled to avoid under-coverage.
    * Phasing is NOT classic "cumulative Walker f" phasing. Instead an
      *alternating* even/odd-plane stagger is applied, as required for hex
      packing at |L| = lmin, with a latitude-corrected along-track shift:
          Δu = (360/S) * (1 / (2 cos i_Lmin))
      where cos(i_L) = cos(i)/cos(L) (Wright Eq. 10).
    """

    # ---- Parse + validate inputs ----
    r_km = float(intercept_radius_km)
    h_int_km = float(intercept_alt_km)
    h_km = float(constellation_alt_km)
    lmax_deg = float(lmax_deg)
    lmin_deg = float(lmin_deg)

    ecc = float(ecc)
    argp_deg = float(argp_deg)
    raan_offset_deg = float(raan_offset_deg)
    m0_offset_deg = float(m0_offset_deg)
    bc_kg_m2 = float(bc_kg_m2)
    sat_margin = float(sat_margin)

    if r_km <= 0:
        raise ValueError("wright_hex: intercept_radius_km must be > 0")
    if h_km <= 0:
        raise ValueError("wright_hex: constellation_alt_km must be > 0")
    if h_int_km < 0:
        raise ValueError("wright_hex: intercept_alt_km must be >= 0")
    if sat_margin < 1.0:
        raise ValueError("wright_hex: sat_margin must be >= 1.0")

    Lmin = np.deg2rad(abs(lmin_deg))
    inc_deg = abs(lmax_deg)
    inc = np.deg2rad(inc_deg)

    if Lmin >= inc:
        raise ValueError("wright_hex: require lmin_deg < lmax_deg (orbits must reach above Lmin).")

    # ---- Wright Eq. 5: horizontal disc radius at parking altitude ----
    dz = h_km - h_int_km
    inside = r_km * r_km - dz * dz
    if inside <= 0.0:
        raise ValueError(
            "wright_hex: intercept_radius_km is too small for the altitude difference "
            "(r^2 <= (h-h_int)^2). Increase intercept_radius_km or reduce |h-h_int|."
        )
    r_h_km = float(np.sqrt(inside))

    # ---- Wright Eq. 7/8/9/13: size the lattice ----
    d_km = float(np.sqrt(3.0) * r_h_km)   # spacing within a plane (centers) at Lmin lattice
    D_Lmin_km = float(1.5 * r_h_km)       # adjacent-plane spacing at Lmin (orthogonal to track)

    R_shell_km = float(earth.r_eq_km + h_km)  # radius at satellite altitude

    # Eq. 9: sats per plane
    n_s_est = (2.0 * np.pi * R_shell_km) / d_km * multiplier
    n_s_est_with_margin = n_s_est * sat_margin

    # Eq. 13 factor: sqrt(cos^2(Lmin) - cos^2(i))
    cosL = float(np.cos(Lmin))
    cosi = float(np.cos(inc))
    root_term = cosL * cosL - cosi * cosi
    if root_term <= 0.0:
        raise ValueError("wright_hex: invalid latitude geometry; check lmin_deg/lmax_deg")

    # Eq. 13: number of planes
    n_p_est = (2.0 * np.pi * R_shell_km) / (2.0 * D_Lmin_km) * float(np.sqrt(root_term))

    # Round up to guarantee no gaps at Lmin
    S = int(np.ceil(n_s_est_with_margin))
    P = int(np.ceil(n_p_est))
    if P <= 0 or S <= 0:
        raise ValueError("wright_hex: computed non-positive plane/sat counts")

    T = P * S

    # ---- Build elements arrays (same layout as walker_delta_layer) ----
    a_km = float(earth.r_eq_km + h_km)

    a_arr = np.full(T, a_km, dtype=np.float64)
    e_arr = np.full(T, ecc, dtype=np.float64)
    i_rad = np.deg2rad(np.full(T, inc_deg, dtype=np.float64))
    argp_rad = np.deg2rad(np.full(T, argp_deg, dtype=np.float64))

    plane_idx = np.repeat(np.arange(P), S)
    sat_idx = np.tile(np.arange(S), P)

    # RAAN per plane
    raan_plane_deg = (360.0 / P) * plane_idx + raan_offset_deg
    raan_rad = np.deg2rad(raan_plane_deg)

    # ---- Latitude-aware hex staggering at |L| = Lmin ----
    # Wright Eq. 10: cos(i_L) = cos(i)/cos(L)
    cos_iL = cosi / cosL
    if cos_iL <= 0.0:
        raise ValueError("wright_hex: computed cos(i_Lmin) <= 0; check lmin_deg/lmax_deg")

    # Derived: Δu_deg = (360/S) * (1/(2*cos(i_Lmin)))
    delta_u_deg = (360.0 / S) * (1.0 / (2.0 * cos_iL))

    # Alternate planes are shifted by +Δu_deg (NOT cumulative).
    phase_deg = (plane_idx % 2) * delta_u_deg

    # Mean anomaly per sat within plane + staggering + constant offset
    M_deg = (360.0 / S) * sat_idx + phase_deg + m0_offset_deg
    M0_rad = np.deg2rad(M_deg)

    elems = OrbitalElements(
        a_km=a_arr, e=e_arr, i_rad=i_rad, raan_rad=raan_rad, argp_rad=argp_rad, M0_rad=M0_rad
    )
    phys = SatellitePhysical(bc_kg_m2=np.full(T, bc_kg_m2, dtype=np.float64))

    spec = {
        "type": "wright_hex",
        "method": "wright_hex_lattice",
        "n_planes": P,
        "sats_per_plane": S,
        "constellation_alt_km": h_km,
        "intercept_alt_km": h_int_km,
        "intercept_radius_km": r_km,
        "coverage_disc_radius_km": r_h_km,
        "d_km": d_km,
        "D_Lmin_km": D_Lmin_km,
        "lmin_deg": float(abs(lmin_deg)),
        "lmax_deg": float(abs(lmax_deg)),
        "n_s_est": float(n_s_est),
        "sat_margin": float(sat_margin),
        "n_s_est_with_margin": float(n_s_est_with_margin),
        "n_p_est": float(n_p_est),
        "a_km": a_km,
        "e": ecc,
        "i_deg": inc_deg,
        "raan_offset_deg": raan_offset_deg,
        "argp_deg": argp_deg,
        "m0_offset_deg": m0_offset_deg,
        "delta_u_deg_between_even_odd_planes_at_Lmin": float(delta_u_deg),
        "cos_iLmin": float(cos_iL),
        "bc_kg_m2": bc_kg_m2,
    }

    return Layer(elems=elems, phys=phys, spec=spec)
