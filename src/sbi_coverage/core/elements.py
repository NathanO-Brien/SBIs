from __future__ import annotations

from dataclasses import dataclass


@dataclass
class OrbitalElements:
    a_km: float
    e: float
    i_rad: float
    raan_rad: float
    argp_rad: float

    # Canonical anomaly used across your constellation/layer builders
    M0_rad: float

    # Optional true anomaly for backward compatibility / convenience.
    # If provided, downstream conversion routines may prefer it.
    ta_rad: float | None = None


@dataclass
class SatellitePhysical:
    # Existing representation
    bc_kg_m2: float

    # Optional: enable drag models that require explicit mass/area/Cd.
    # Assumption: uniform satellite design across constellation.
    mass_kg: float | None = None
    area_m2: float | None = None
    cd: float | None = None