"""Geometric optimal satellite altitude for boost-phase intercept coverage.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

from ..core.config import EarthConstants


def _rho_range(r_sat: float, r_tgt: float, max_range_km: float) -> float:
    """Earth-central angle at the range-limited footprint edge."""
    cos_rho = (r_sat**2 + r_tgt**2 - max_range_km**2) / (2.0 * r_sat * r_tgt)
    return float(np.arccos(np.clip(cos_rho, -1.0, 1.0)))


def _rho_elev(r_sat: float, r_tgt: float, min_elev_rad: float) -> float:
    """Earth-central angle at the elevation-limited footprint edge."""
    cos_arg = r_tgt * np.cos(min_elev_rad) / r_sat
    return float(np.arccos(np.clip(cos_arg, -1.0, 1.0))) - min_elev_rad


def optimal_sat_altitude_km(
    max_range_km: float,
    min_elev_deg: float,
    intercept_alt_km: float,
    earth: EarthConstants,
) -> float:
    """Return the satellite altitude h* (km) that maximises engagement footprint.

    At h* the range constraint and the elevation constraint bind simultaneously
    at the footprint edge — below h* elevation is limiting, above h* range is
    limiting.  The optimum is the unique root of:

        rho_elev(h) - rho_range(h) = 0

    Parameters
    ----------
    max_range_km      : interceptor max slant range from CoverageConfig.max_range_km
    min_elev_deg      : minimum elevation angle from CoverageConfig.min_elev_deg
    intercept_alt_km  : target intercept altitude from CoverageConfig.intercept_alt_km
    earth             : EarthConstants (provides r_eq_km)

    Returns
    -------
    h_star_km : optimal satellite altitude above Earth's surface (km)
    """
    if max_range_km <= 0.0:
        raise ValueError("max_range_km must be > 0")
    if intercept_alt_km < 0.0:
        raise ValueError("intercept_alt_km must be >= 0")

    r_tgt        = earth.r_eq_km + intercept_alt_km
    min_elev_rad = np.deg2rad(min_elev_deg)

    def imbalance(h_sat: float) -> float:
        """Difference between elevation-limited and range-limited footprint
        radii; the optimal altitude is its root."""
        r_sat = earth.r_eq_km + h_sat
        return _rho_elev(r_sat, r_tgt, min_elev_rad) - _rho_range(r_sat, r_tgt, max_range_km)

    # Search between just above the intercept altitude and just below
    # the altitude at which the range footprint collapses to a single point.
    h_lo = intercept_alt_km + 1.0
    h_hi = intercept_alt_km + max_range_km - 1.0

    if h_hi <= h_lo:
        raise ValueError(
            f"max_range_km={max_range_km:.1f} km is too small to admit a valid search interval "
            f"above intercept_alt_km={intercept_alt_km:.1f} km."
        )

    return float(brentq(imbalance, h_lo, h_hi, xtol=1e-6, rtol=1e-9))


def footprint_half_angles_deg(
    h_sat_km: float,
    max_range_km: float,
    min_elev_deg: float,
    intercept_alt_km: float,
    earth: EarthConstants,
) -> dict[str, float]:
    """Return the range- and elevation-limited footprint half-angles at a given altitude.

    Useful for diagnosing how close a candidate altitude is to the optimum and
    which constraint is binding.

    Returns a dict with keys:
        rho_range_deg   : footprint half-angle from the range constraint
        rho_elev_deg    : footprint half-angle from the elevation constraint
        rho_eff_deg     : effective footprint half-angle (min of the two)
        binding         : 'range' | 'elevation' | 'balanced'
    """
    r_sat        = earth.r_eq_km + h_sat_km
    r_tgt        = earth.r_eq_km + intercept_alt_km
    min_elev_rad = np.deg2rad(min_elev_deg)

    rho_r = np.rad2deg(_rho_range(r_sat, r_tgt, max_range_km))
    rho_e = np.rad2deg(_rho_elev(r_sat, r_tgt, min_elev_rad))
    rho_eff = min(rho_r, rho_e)

    tol = 0.01  # degrees
    if abs(rho_r - rho_e) < tol:
        binding = "balanced"
    elif rho_e < rho_r:
        binding = "elevation"
    else:
        binding = "range"

    return {
        "rho_range_deg": float(rho_r),
        "rho_elev_deg":  float(rho_e),
        "rho_eff_deg":   float(rho_eff),
        "binding":       binding,
    }