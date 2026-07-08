"""Configuration dataclasses for Earth constants, engagement/coverage
parameters, and simulation settings.
"""
from dataclasses import dataclass, field
from typing import Optional

from .engagement import interceptor_range_km


@dataclass(frozen=True)
class EarthConstants:
    """Physical constants for a spherical-Earth model (WGS-84 values)."""
    mu_km3_s2: float = 398600.4418            # gravitational parameter, km^3/s^2
    r_eq_km: float = 6378.137                 # equatorial radius, km
    j2: float = 1.08262668e-3                 # J2 zonal harmonic coefficient
    omega_earth_rad_s: float = 7.2921150e-5   # Earth rotation rate, rad/s
    g0_m_s2: float = 9.80665                  # standard gravity, m/s^2


@dataclass(frozen=True)
class CoverageConfig:
    """Engagement parameters and derived interceptor reach.

    The key derived value is `max_range_km`: the maximum straight-line
    distance an interceptor can travel within the engagement window,
    computed from the accelerate-then-coast performance model in
    engagement.interceptor_range_km() unless explicitly overridden.
    """

    # Engagement / sizing inputs
    earth: EarthConstants
    T_window_s: float          # engagement time window, s
    v_bo_km_s: float           # interceptor burnout velocity, km/s
    a_g: float                 # interceptor acceleration, g's
    intercept_alt_km: float    # intercept altitude above Earth surface, km

    # Coverage geometry
    min_elev_deg: float = 10.0  # minimum elevation of satellite above target's local horizon

    # Optional override of the derived reach (for experiments)
    max_range_km_override: Optional[float] = None

    # Derived interceptor reach (computed in __post_init__)
    max_range_km: float = field(init=False)

    def __post_init__(self):
        if self.T_window_s <= 0.0:
            raise ValueError("CoverageConfig: T_window_s must be > 0")
        if self.v_bo_km_s <= 0.0:
            raise ValueError("CoverageConfig: v_bo_km_s must be > 0")
        if self.a_g <= 0.0:
            raise ValueError("CoverageConfig: a_g must be > 0")

        if self.max_range_km_override is not None:
            max_range = float(self.max_range_km_override)
        else:
            max_range = float(
                interceptor_range_km(
                    T_s=float(self.T_window_s),
                    v_km_s=float(self.v_bo_km_s),
                    a_g=float(self.a_g),
                    g0_m_s2=float(getattr(self.earth, "g0_m_s2", 9.80665)),
                )
            )

        object.__setattr__(self, "max_range_km", max_range)


@dataclass
class SimConfig:
    """Simulation timeframe, propagation model toggles, and output options."""

    # Simulation timeframe
    horizon_s: float = 24 * 3600   # total simulated duration, s
    dt_s: float = 30.0             # coverage evaluation timestep, s
    point_chunk: int = 4096        # shell points processed per batch (memory control)

    # Epoch identifier stored for reproducibility (currently "J2000" only).
    epoch0: str = "J2000"

    # Propagation toggles
    use_j2: bool = True
    use_drag: bool = False

    # Exponential atmosphere model for the nominal propagator when use_drag=True:
    #   rho(h) = rho_scale * rho0_kg_m3 * exp(-(h - h0_km) / H_km)
    rho0_kg_m3: float = 3.614e-13
    h0_km: float = 700.0
    H_km: float = 88.667
    rho_scale: float = 1.0
    drag_floor_alt_km: float = 120.0   # below this altitude the satellite is considered decayed

    # Storage dtype for the coverage counts matrix. uint8 halves storage but
    # clips counts above 255; uint16 is safer for large constellations.
    analysis_matrix_dtype: str = "uint16"

    def __post_init__(self):
        valid = {"uint8", "uint16"}
        if self.analysis_matrix_dtype not in valid:
            raise ValueError(
                f"SimConfig: analysis_matrix_dtype must be one of {sorted(valid)}, "
                f"got {self.analysis_matrix_dtype!r}"
            )
        if self.H_km <= 0.0:
            raise ValueError("SimConfig: H_km must be > 0")
        if self.rho0_kg_m3 < 0.0:
            raise ValueError("SimConfig: rho0_kg_m3 must be >= 0")
        if self.rho_scale < 0.0:
            raise ValueError("SimConfig: rho_scale must be >= 0")
