"""Configuration dataclasses for Earth constants, engagement/coverage
parameters, and simulation settings.
"""
from dataclasses import dataclass, field

from .engagement import interceptor_range_km


@dataclass(frozen=True)
class EarthConstants:
    """Physical constants for a spherical-Earth model (WGS-84 values; r_eq_km is used as the sphere radius)."""
    mu_km3_s2: float = 398600.4418            # gravitational parameter, km^3/s^2
    r_eq_km: float = 6378.137                 # equatorial radius, km
    j2: float = 1.08262668e-3                 # J2 zonal harmonic coefficient
    omega_earth_rad_s: float = 7.2921150e-5   # Earth rotation rate, rad/s
    g0_m_s2: float = 9.80665                  # standard gravity, m/s^2


@dataclass(frozen=True)
class CoverageConfig:
    """Engagement parameters and derived interceptor reach.

    Timing model: the target missile burns out `target_missile_burnout_time_s`
    after launch, and the intercept must happen before then. Of that budget,
    `detection_time_s` is consumed before the launch is even detected and
    `decision_time_s` before the decision to engage is made -- neither is
    available to the interceptor's own flight. What's left,
    `interceptor_engagement_time_s = target_missile_burnout_time_s -
    detection_time_s - decision_time_s`, is the ONLY one of the four that
    feeds the flyout-range physics. Modeling zero detection/decision time
    (the default) makes interceptor_engagement_time_s equal to
    target_missile_burnout_time_s -- i.e. the interceptor is assumed to
    launch the instant the target does.

    The key derived value is `max_range_km`: the maximum straight-line
    distance an interceptor can travel within interceptor_engagement_time_s,
    computed from the accelerate-then-coast performance model in
    engagement.interceptor_range_km() (a straight-line, gravity-free flyout model).
    """

    # Engagement / sizing inputs
    earth: EarthConstants
    detection_time_s: float             # time from target launch to detection, s
    decision_time_s: float              # time from detection to engage decision, s
    target_missile_burnout_time_s: float  # time from target launch to target burnout, s
    burnout_velocity_km_s: float  # interceptor burnout velocity, km/s
    max_accel_g: float            # interceptor maximum acceleration, in multiples of g0
    intercept_alt_km: float       # altitude above the surface of the target points (the intercept shell), km

    # Coverage geometry
    min_elev_deg: float = 10.0  # minimum elevation of satellite above target's local horizon

    # Derived timing and interceptor reach (computed in __post_init__)
    interceptor_engagement_time_s: float = field(init=False)
    max_range_km: float = field(init=False)

    def __post_init__(self):
        if self.detection_time_s < 0.0:
            raise ValueError("CoverageConfig: detection_time_s must be >= 0")
        if self.decision_time_s < 0.0:
            raise ValueError("CoverageConfig: decision_time_s must be >= 0")
        if self.target_missile_burnout_time_s <= 0.0:
            raise ValueError("CoverageConfig: target_missile_burnout_time_s must be > 0")
        if self.burnout_velocity_km_s <= 0.0:
            raise ValueError("CoverageConfig: burnout_velocity_km_s must be > 0")
        if self.max_accel_g <= 0.0:
            raise ValueError("CoverageConfig: max_accel_g must be > 0")

        engagement_time = (
            self.target_missile_burnout_time_s - self.detection_time_s - self.decision_time_s
        )
        if engagement_time <= 0.0:
            raise ValueError(
                "CoverageConfig: detection_time_s + decision_time_s "
                f"({self.detection_time_s + self.decision_time_s:.3f} s) must be less than "
                f"target_missile_burnout_time_s ({self.target_missile_burnout_time_s:.3f} s) -- "
                "no time left for interceptor engagement"
            )
        object.__setattr__(self, "interceptor_engagement_time_s", engagement_time)

        max_range = float(
            interceptor_range_km(
                T_s=float(engagement_time),
                burnout_velocity_km_s=float(self.burnout_velocity_km_s),
                max_accel_g=float(self.max_accel_g),
                g0_m_s2=float(self.earth.g0_m_s2),
            )
        )

        object.__setattr__(self, "max_range_km", max_range)


def coverage_config_from_sidecar(c: dict, earth: EarthConstants) -> "CoverageConfig":
    """Reconstruct a CoverageConfig from a saved result JSON's "cov" block.

    Accepts current keys (burnout_velocity_km_s, max_accel_g) and the pre-rename keys
    (v_bo_km_s, a_g). Supports both timing schemas: a current sidecar (detection_time_s/decision_time_s/
    target_missile_burnout_time_s) is used directly. A legacy sidecar (only
    "T_window_s", from before the detection/decision timing split) is read as
    detection_time_s=0, decision_time_s=0, target_missile_burnout_time_s=
    T_window_s -- exactly the old behavior, where the whole window was
    dedicated to interceptor flyout.

    """
    if "target_missile_burnout_time_s" in c:
        detection_time_s = float(c["detection_time_s"])
        decision_time_s = float(c["decision_time_s"])
        target_missile_burnout_time_s = float(c["target_missile_burnout_time_s"])
    else:
        detection_time_s = 0.0
        decision_time_s = 0.0
        target_missile_burnout_time_s = float(c["T_window_s"])

    kwargs = dict(
        earth=earth,
        detection_time_s=detection_time_s,
        decision_time_s=decision_time_s,
        target_missile_burnout_time_s=target_missile_burnout_time_s,
        burnout_velocity_km_s=float(c.get("burnout_velocity_km_s", c.get("v_bo_km_s"))),
        max_accel_g=float(c.get("max_accel_g", c.get("a_g"))),
        intercept_alt_km=float(c["intercept_alt_km"]),
        min_elev_deg=float(c["min_elev_deg"]),
    )
    return CoverageConfig(**kwargs)


@dataclass
class SimConfig:
    """Simulation timeframe, propagation model toggles, and output options."""

    # Simulation timeframe
    horizon_s: float = 24 * 3600   # total simulated duration, s
    dt_s: float = 30.0             # coverage evaluation timestep, s
    point_chunk: int = 4096        # shell points processed per batch (memory control)

    # Epoch identifier stored for reproducibility (currently "J2000" only).
    epoch0: str = "J2000"

    # Propagation model. use_j2=True is the Lee (2020) secular-J2 model (RAAN, argument of
    # perigee and mean-anomaly rates) and is the only model the optimizer uses; False is plain
    # two-body motion, kept only so J2-off results archived before the J2 work can be replayed.
    use_j2: bool = True

    # Storage dtype for the coverage counts matrix. uint8 halves storage but counts above 255
    # wrap modulo 256 (a warning is printed); uint16 is safer for large constellations.
    analysis_matrix_dtype: str = "uint16"

    def __post_init__(self):
        valid = {"uint8", "uint16"}
        if self.analysis_matrix_dtype not in valid:
            raise ValueError(
                f"SimConfig: analysis_matrix_dtype must be one of {sorted(valid)}, "
                f"got {self.analysis_matrix_dtype!r}"
            )
