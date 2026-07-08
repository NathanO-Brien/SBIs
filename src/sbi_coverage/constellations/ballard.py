"""Ballard rosette (street-of-coverage) constellation layer generator.
"""
from __future__ import annotations

import math

import numpy as np

from ..core.config import CoverageConfig, EarthConstants
from ..core.elements import OrbitalElements, SatellitePhysical
from .layer import Layer


def _shell_half_angle_bounds_rad_from_cov(
    *,
    earth: EarthConstants,
    cov: CoverageConfig,
    constellation_alt_km: float,
) -> tuple[float, float, float]:
    """Earth-central half-angles limited by interceptor range and by minimum
    elevation; returns (psi_range, psi_elev, psi_bound) in radians."""
    r_p = float(earth.r_eq_km + cov.intercept_alt_km)
    r_s = float(earth.r_eq_km + constellation_alt_km)
    range_km = float(cov.max_range_km)
    eps_rad = math.radians(float(cov.min_elev_deg))

    cos_psi_range = (r_s * r_s + r_p * r_p - range_km * range_km) / (2.0 * r_s * r_p)
    cos_psi_range = float(np.clip(cos_psi_range, -1.0, 1.0))
    psi_range = float(math.acos(cos_psi_range))

    beta = r_p / r_s
    cos_eps = math.cos(eps_rad)
    sin_eps = math.sin(eps_rad)
    root_arg = max(1.0 - (beta * cos_eps) * (beta * cos_eps), 0.0)
    cos_psi_elev = beta * cos_eps * cos_eps + sin_eps * math.sqrt(root_arg)
    cos_psi_elev = float(np.clip(cos_psi_elev, -1.0, 1.0))
    psi_elev = float(math.acos(cos_psi_elev))

    cos_psi_bound = max(cos_psi_range, cos_psi_elev)
    psi_bound = float(math.acos(float(np.clip(cos_psi_bound, -1.0, 1.0))))
    return psi_range, psi_elev, psi_bound


def _latitude_grid_deg(lat_min_deg: float, lat_max_deg: float, n_samples: int) -> np.ndarray:
    """Evenly spaced latitude samples between the given bounds (deg)."""
    lo = float(min(lat_min_deg, lat_max_deg))
    hi = float(max(lat_min_deg, lat_max_deg))
    n = int(max(3, n_samples))
    return np.linspace(lo, hi, n, dtype=np.float64)


def _cross_track_plane_estimate(
    *,
    alpha_eff_rad: float,
    lat_rad: float,
    inc_rad: float,
) -> tuple[float, float, float]:
    """
    Return a Ballard-style plane estimate and the local track-angle terms.

    The v1 constructor used a single-latitude estimate:
        P ~ pi cos(lat) / (alpha_eff sin(i_L))

    This is the conservative Ballard-style plane count used for the repo-native
    symmetric continuous-coverage approximation. It intentionally does not apply
    the optimistic factor-of-two street reduction that proved too weak in
    practice.
    """
    cos_lat = max(math.cos(lat_rad), 1e-12)
    cos_iL = math.cos(inc_rad) / cos_lat
    cos_iL = float(np.clip(cos_iL, -1.0, 1.0))
    sin_iL = math.sqrt(max(1.0 - cos_iL * cos_iL, 0.0))
    if sin_iL <= 1e-9:
        return math.inf, cos_iL, sin_iL

    p_est = (math.pi * cos_lat) / (alpha_eff_rad * sin_iL)
    return float(p_est), float(cos_iL), float(sin_iL)


def _along_track_satellite_estimate(
    *,
    alpha_eff_rad: float,
    lat_rad: float,
    inc_rad: float,
) -> float:
    """
    Return a Ballard-style satellites-per-plane estimate.

    Conservative Ballard-style satellites-per-plane estimate. This stays
    latitude-independent and avoids the optimistic along-track stretch
    correction that underpredicted required density.
    """
    _ = lat_rad
    _ = inc_rad
    s_est = math.pi / max(alpha_eff_rad, 1e-9)
    return float(s_est)


def ballard_street_layer(
    earth: EarthConstants,
    *,
    cov: CoverageConfig,
    constellation_alt_km: float,
    inc_deg: float,
    lat_min_deg: float = 0.0,
    lat_max_deg: float,
    required_count: int = 1,
    overlap_fraction: float = 0.0,
    plane_margin: float = 1.0,
    sat_margin: float = 1.0,
    stagger_fraction: float | None = None,
    latitude_samples: int = 33,
    ecc: float = 0.0,
    argp_deg: float = 0.0,
    raan_offset_deg: float = 0.0,
    m0_offset_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
) -> Layer:
    """
    Symmetric Ballard/street-of-coverage layer sized for instantaneous no-gap coverage.

    This generator is coverage-first rather than catalog-first:
      1) derive one-satellite shell half-angle from intercept geometry
      2) size planes and satellites/plane from no-gap spacing conditions
      3) instantiate a symmetric circular shell with alternating half-slot staggering

    Design model
    ------------
    The sizing is a repo-native analytic Ballard-style approximation for symmetric
    continuous coverage over a target latitude belt. It stays purely analytic and
    is aligned to the repo's coverage model by:

      - deriving one-satellite shell reach from both max range and min elevation
      - sizing over the full latitude band instead of a single edge latitude
      - using conservative cross-track and along-track density estimates
      - scaling analytically for higher continuous multiplicity targets

    Inputs
    ------
    cov:
        Coverage configuration. The constructor uses cov.max_range_km,
        cov.intercept_alt_km, and cov.min_elev_deg.
    constellation_alt_km:
        Satellite shell altitude in km.
    inc_deg:
        Orbit inclination in degrees.
    lat_min_deg, lat_max_deg:
        Target latitude belt. The design is driven by the outer edge in magnitude.
    required_count:
        Analytic continuous multiplicity target. 1 means single-interceptor continuous
        coverage, 2 means approximate 2-cover, etc.
    overlap_fraction:
        Extra overlap demand in [0, 1). 0 means "just touching" streets/slots.
    plane_margin, sat_margin:
        Multiplicative safety margins applied after the analytic estimates.
    stagger_fraction:
        Alternating odd/even plane phase shift as a fraction of one in-plane slot.
        Default None selects the analytic default (currently half-slot interlacing).
    latitude_samples:
        Number of latitude samples used to size against the worst case over the belt.
    """
    h_km = float(constellation_alt_km)
    inc_deg = float(abs(inc_deg))
    lat_min_deg = float(lat_min_deg)
    lat_max_deg = float(lat_max_deg)
    required_count = int(required_count)
    overlap_fraction = float(overlap_fraction)
    plane_margin = float(plane_margin)
    sat_margin = float(sat_margin)
    latitude_samples = int(latitude_samples)
    ecc = float(ecc)
    argp_deg = float(argp_deg)
    raan_offset_deg = float(raan_offset_deg)
    m0_offset_deg = float(m0_offset_deg)
    bc_kg_m2 = float(bc_kg_m2)

    if h_km <= 0.0:
        raise ValueError("ballard_street_layer: constellation_alt_km must be > 0")
    if not (0.0 < inc_deg <= 90.0):
        raise ValueError("ballard_street_layer: inc_deg must be in (0, 90]")
    if required_count <= 0:
        raise ValueError("ballard_street_layer: required_count must be > 0")
    if not (0.0 <= overlap_fraction < 1.0):
        raise ValueError("ballard_street_layer: overlap_fraction must be in [0, 1)")
    if plane_margin < 1.0:
        raise ValueError("ballard_street_layer: plane_margin must be >= 1.0")
    if sat_margin < 1.0:
        raise ValueError("ballard_street_layer: sat_margin must be >= 1.0")
    if stagger_fraction is not None and not (0.0 <= float(stagger_fraction) <= 1.0):
        raise ValueError("ballard_street_layer: stagger_fraction must be in [0, 1]")
    if latitude_samples < 3:
        raise ValueError("ballard_street_layer: latitude_samples must be >= 3")

    psi_range, psi_elev, psi_bound = _shell_half_angle_bounds_rad_from_cov(
        earth=earth,
        cov=cov,
        constellation_alt_km=h_km,
    )
    max_abs_lat_deg = float(max(abs(lat_min_deg), abs(lat_max_deg)))
    if max_abs_lat_deg >= inc_deg:
        raise ValueError(
            "ballard_street_layer: target latitude band requires inc_deg above the "
            "design edge. Increase inc_deg, reduce the shell latitude band, or increase shell reach."
        )
    psi_eff = psi_bound * (1.0 - overlap_fraction)
    if psi_eff <= 0.0:
        raise ValueError("ballard_street_layer: effective half-angle collapsed to zero.")

    inc_rad = math.radians(inc_deg)
    design_lat_rad = math.radians(max_abs_lat_deg)
    cos_iL_design = math.cos(inc_rad) / max(math.cos(design_lat_rad), 1e-12)
    cos_iL_design = float(np.clip(cos_iL_design, -1.0, 1.0))
    sin_iL_design = float(math.sqrt(max(1.0 - cos_iL_design * cos_iL_design, 0.0)))

    lat_grid_deg = _latitude_grid_deg(lat_min_deg, lat_max_deg, latitude_samples)
    plane_est_samples: list[float] = []
    sat_est_samples: list[float] = []
    local_cos_iL: list[float] = []
    local_sin_iL: list[float] = []

    for lat_deg in lat_grid_deg:
        lat_rad = math.radians(float(lat_deg))
        p_local, cos_iL, sin_iL = _cross_track_plane_estimate(
            alpha_eff_rad=psi_eff,
            lat_rad=lat_rad,
            inc_rad=inc_rad,
        )
        s_local = _along_track_satellite_estimate(
            alpha_eff_rad=psi_eff,
            lat_rad=lat_rad,
            inc_rad=inc_rad,
        )
        plane_est_samples.append(float(p_local))
        sat_est_samples.append(float(s_local))
        local_cos_iL.append(float(cos_iL))
        local_sin_iL.append(float(sin_iL))

    if not plane_est_samples or not np.all(np.isfinite(plane_est_samples)):
        raise ValueError(
            "ballard_street_layer: invalid cross-track geometry across the target latitude band. "
            "Increase inclination or narrow the latitude band."
        )
    if not sat_est_samples or not np.all(np.isfinite(sat_est_samples)):
        raise ValueError("ballard_street_layer: invalid along-track geometry across the target latitude band.")

    plane_binding_idx = int(np.argmax(np.asarray(plane_est_samples, dtype=np.float64)))
    sat_binding_idx = int(np.argmax(np.asarray(sat_est_samples, dtype=np.float64)))
    plane_binding_lat_deg = float(lat_grid_deg[plane_binding_idx])
    sat_binding_lat_deg = float(lat_grid_deg[sat_binding_idx])
    p_est_raw = float(max(plane_est_samples))
    s_est_raw = float(max(sat_est_samples))
    density_scale = float(math.sqrt(required_count))

    p_est = p_est_raw * density_scale * plane_margin
    s_est = s_est_raw * density_scale * sat_margin

    n_planes = int(max(1, math.ceil(p_est)))
    sats_per_plane = int(max(1, math.ceil(s_est)))
    n_total = int(n_planes * sats_per_plane)

    if stagger_fraction is None:
        cos_iL_bind = max(local_cos_iL[plane_binding_idx], 1e-12)
        stagger_fraction = min(1.0, 1.0 / (2.0 * cos_iL_bind))
        stagger_fraction_source = "analytic_default"
    else:
        stagger_fraction_source = "user"
    stagger_fraction = float(stagger_fraction)

    a_km = float(earth.r_eq_km + h_km)
    plane_idx = np.repeat(np.arange(n_planes), sats_per_plane)
    sat_idx = np.tile(np.arange(sats_per_plane), n_planes)

    raan_deg = (360.0 / n_planes) * plane_idx + raan_offset_deg
    slot_spacing_deg = 360.0 / sats_per_plane
    phase_deg = (plane_idx % 2) * (stagger_fraction * slot_spacing_deg)
    m_deg = slot_spacing_deg * sat_idx + phase_deg + m0_offset_deg

    elems = OrbitalElements(
        a_km=np.full(n_total, a_km, dtype=np.float64),
        e=np.full(n_total, ecc, dtype=np.float64),
        i_rad=np.deg2rad(np.full(n_total, inc_deg, dtype=np.float64)),
        raan_rad=np.deg2rad(raan_deg.astype(np.float64)),
        argp_rad=np.deg2rad(np.full(n_total, argp_deg, dtype=np.float64)),
        M0_rad=np.deg2rad(m_deg.astype(np.float64)),
    )
    phys = SatellitePhysical(
        bc_kg_m2=np.full(n_total, bc_kg_m2, dtype=np.float64)
    )

    spec = {
        "type": "ballard_street",
        "method": "ballard_instantaneous_no_gap_symmetric",
        "n_total": int(n_total),
        "n_planes": int(n_planes),
        "sats_per_plane": int(sats_per_plane),
        "constellation_alt_km": float(h_km),
        "intercept_alt_km": float(cov.intercept_alt_km),
        "intercept_radius_km": float(cov.max_range_km),
        "max_range_km": float(cov.max_range_km),
        "min_elev_deg": float(cov.min_elev_deg),
        "required_count": int(required_count),
        "coverage_half_angle_deg": float(math.degrees(psi_bound)),
        "coverage_half_angle_deg_range": float(math.degrees(psi_range)),
        "coverage_half_angle_deg_elev": float(math.degrees(psi_elev)),
        "coverage_half_angle_deg_bound": float(math.degrees(psi_bound)),
        "effective_half_angle_deg": float(math.degrees(psi_eff)),
        "overlap_fraction": float(overlap_fraction),
        "lat_min_deg": float(lat_min_deg),
        "lat_max_deg": float(lat_max_deg),
        "design_lat_deg": float(max_abs_lat_deg),
        "latitude_samples": int(latitude_samples),
        "inc_deg": float(inc_deg),
        "track_angle_cos_at_design_lat": float(cos_iL_design),
        "track_angle_sin_at_design_lat": float(sin_iL_design),
        "binding_plane_lat_deg": float(plane_binding_lat_deg),
        "binding_sat_lat_deg": float(sat_binding_lat_deg),
        "plane_margin": float(plane_margin),
        "sat_margin": float(sat_margin),
        "plane_count_est": float(p_est),
        "sats_per_plane_est": float(s_est),
        "plane_count_est_raw": float(p_est_raw),
        "sats_per_plane_est_raw": float(s_est_raw),
        "density_scale": float(density_scale),
        "stagger_fraction": float(stagger_fraction),
        "stagger_fraction_source": str(stagger_fraction_source),
        "slot_spacing_deg": float(slot_spacing_deg),
        "raan_offset_deg": float(raan_offset_deg),
        "argp_deg": float(argp_deg),
        "m0_offset_deg": float(m0_offset_deg),
        "a_km": float(a_km),
        "e": float(ecc),
        "bc_kg_m2": float(bc_kg_m2),
    }

    return Layer(elems=elems, phys=phys, spec=spec)
