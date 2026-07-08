from dataclasses import dataclass, field
from typing import Optional

from .engagement import interceptor_range_km


@dataclass(frozen=True)
class EarthConstants:
    mu_km3_s2: float = 398600.4418            # km^3/s^2
    r_eq_km: float = 6378.137                 # km (WGS-84 equatorial radius)
    j2: float = 1.08262668e-3
    omega_earth_rad_s: float = 7.2921150e-5   # rad/s
    g0_m_s2: float = 9.80665                  # m/s^2


@dataclass(frozen=True)
class CoverageConfig:
    """Coverage configuration.

    This config owns both user-facing engagement knobs and derived geometry.
    The key derived value is `max_range_km`, computed from the interceptor
    performance model unless overridden.
    """

    # Engagement / sizing inputs (Option C style)
    earth: EarthConstants
    T_window_s: float
    v_bo_km_s: float
    a_g: float
    intercept_alt_km: float

    # Coverage / geometry knobs
    min_elev_deg: float = 10.0

    # Optional override (for experiments)
    max_range_km_override: Optional[float] = None

    # Derived geometry (computed in __post_init__)
    max_range_km: float = field(init=False)

    def __post_init__(self):
        # Basic validation
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
    # Simulation timeframe
    horizon_s: float = 24 * 3600
    dt_s: float = 30.0
    point_chunk: int = 4096

    # NOTE: epoch0 is stored as a string enum for ease of use and reproducibility.
    # Current supported values are strings like "J2000".
    epoch0: str = "J2000"

    # Propagation toggles / knobs
    use_j2: bool = True
    use_drag: bool = False
    # Simple exponential atmosphere knobs for Nominal_Propagator when use_drag=True.
    # rho(h) = rho_scale * rho0_kg_m3 * exp(-(h-h0_km)/H_km)
    rho0_kg_m3: float = 3.614e-13
    h0_km: float = 700.0
    H_km: float = 88.667
    rho_scale: float = 1.0
    drag_floor_alt_km: float = 120.0

    # HohmannPy Cowell integration knobs (used by propagators/hohmannpy_cowell.py)
    hohmannpy_cowell_step_s: float = 10.0
    hohmannpy_rtol: float = 1e-9
    hohmannpy_atol: float = 1e-12
    hohmannpy_use_j2: bool = True
    hohmannpy_use_drag: bool = False

    # Controls how counts are represented in the returned Analysis object and persisted.
    # uint8 saves substantial storage/bandwidth but can clip values >255. uint16 is safer.
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
