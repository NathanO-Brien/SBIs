"""Vectorized orbital element and satellite property containers.

Each field holds a NumPy array of shape (n_sats,), so a single instance
describes an entire constellation.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrbitalElements:
    """Classical orbital elements for a set of satellites.

    All angular quantities are in radians. Mean anomaly at epoch (M0_rad) is
    the canonical anomaly used by the constellation/layer builders; true
    anomaly (ta_rad) is optional and preferred by conversion routines when
    present.
    """
    a_km: float        # semi-major axis
    e: float           # eccentricity
    i_rad: float       # inclination
    raan_rad: float    # right ascension of the ascending node
    argp_rad: float    # argument of perigee
    M0_rad: float      # mean anomaly at epoch

    # Optional true anomaly for convenience; conversion routines may prefer it.
    ta_rad: float | None = None


@dataclass
class SatellitePhysical:
    """Physical satellite properties used by drag models.

    Only the ballistic coefficient is required. Mass, area, and drag
    coefficient enable drag models that need them explicitly; a uniform
    satellite design across the constellation is assumed.
    """
    bc_kg_m2: float

    mass_kg: float | None = None
    area_m2: float | None = None
    cd: float | None = None
