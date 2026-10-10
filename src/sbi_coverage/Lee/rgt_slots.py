"""Repeat-ground-track slot construction under Lee (2020) secular J2: closure solve for the
semi-major axis, and the L time-shifted orbital slots of one seed (common ground track).
"""
from __future__ import annotations

import numpy as np

from ..core.config import EarthConstants
from ..core.elements import OrbitalElements
from .layer import Layer
from ..propagators.nominal_propagator import (
    j2_secular_rates_rad_s,
    lee_mean_anomaly_rate_rad_s,
)


def _a_km_from_np_nd(N_P: int, N_D: int, earth: EarthConstants) -> float:
    """Keplerian semi-major axis for N_P orbits in N_D sidereal days."""
    T_sid = (2.0 * np.pi) / earth.omega_earth_rad_s
    T_S   = N_D * T_sid / N_P
    return float((earth.mu_km3_s2 * (T_S / (2.0 * np.pi))**2) ** (1.0 / 3.0))


def lee_j2_rates(a_km: float, inc_deg: float, earth: EarthConstants, e: float = 0.0) -> tuple[float, float, float]:
    """Lee (2020) Eqs. 3-5 secular rates (rad/s): (Omega_dot, omega_dot, M_dot)."""
    i_rad = np.deg2rad(inc_deg)
    a = np.asarray(a_km, dtype=np.float64)
    Omegadot, omegadot = j2_secular_rates_rad_s(a, e, i_rad, earth)
    Mdot = lee_mean_anomaly_rate_rad_s(a, e, i_rad, earth)
    return float(Omegadot), float(omegadot), float(Mdot)


def inclination_for_omega_dot(a_km: float, omega_dot_star: float, earth: EarthConstants) -> float:
    """Inclination (deg) of a circular orbit at ``a_km`` whose secular RAAN rate equals ``omega_dot_star``.

    Omega_dot = -k n cos(i) with k n = 1.5 J2 (Re/a)^2 n, so cos(i) = -Omega_dot*/(k n). Returns NaN when
    no real prograde inclination exists (cos i <= 0 or > 1). Two orbits with equal Omega_dot share the
    repeat period T_r = N_D 2 pi / (w_E - Omega_dot) exactly (Lee 2020 Eq. 33).
    """
    n = np.sqrt(earth.mu_km3_s2 / a_km ** 3)
    kn = 1.5 * earth.j2 * (earth.r_eq_km / a_km) ** 2 * n
    cos_i = -omega_dot_star / kn
    if not (0.0 < cos_i <= 1.0):
        return float("nan")
    return float(np.rad2deg(np.arccos(cos_i)))


def rgt_closure_residual(a_km: float, N_P: int, N_D: int, inc_deg: float, earth: EarthConstants) -> float:
    """Lee Eq. 6 closure residual (rad/s): N_P (w_E - Omega_dot) - N_D (omega_dot + M_dot)."""
    Omegadot, omegadot, Mdot = lee_j2_rates(a_km, inc_deg, earth)
    return N_P * (earth.omega_earth_rad_s - Omegadot) - N_D * (omegadot + Mdot)


def a_km_j2_rgt(N_P: int, N_D: int, inc_deg: float, earth: EarthConstants) -> float:
    """J2-corrected semi-major axis for an RGT orbit (Newton-Raphson), Lee 2020 model.

    Finds a (km) satisfying the exact secular-J2 repeating-ground-track condition
    for an eastward-rotating Earth (frames.py convention: ECEF_lon = RAAN - w_E*t),
    Lee et al. (2020) Eq. 6 with the mean-anomaly rate of Eq. 5:

        N_P * (w_E - Omega_dot(a)) = N_D * (omega_dot(a) + M_dot(a))

    At the returned a, T_r = N_D * 2*pi / (w_E - Omega_dot) and the orbit repeats
    exactly in ECEF after T_r seconds under the secular J2 propagator
    (propagators/nominal_propagator.py with use_j2=True), making the circulant V
    matrix exact for ALL entries (including wrap-around ones).  The Keplerian a from
    _a_km_from_np_nd() is the initial guess.

    Parameters
    ----------
    N_P     : repeat orbits (numerator of RGT ratio)
    N_D     : repeat days   (denominator of RGT ratio)
    inc_deg : inclination (degrees)
    earth   : EarthConstants

    Returns
    -------
    J2-corrected semi-major axis in km.
    """
    a = _a_km_from_np_nd(N_P, N_D, earth)  # Keplerian initial guess

    for _ in range(100):
        f = rgt_closure_residual(a, N_P, N_D, inc_deg, earth)
        da = max(a * 1e-7, 1e-4)
        f2 = rgt_closure_residual(a + da, N_P, N_D, inc_deg, earth)
        df_da = (f2 - f) / da
        if abs(df_da) < 1e-40:
            break
        step = f / df_da
        a = float(np.clip(a - step, earth.r_eq_km + 80.0, earth.r_eq_km + 40000.0))
        if abs(step) < 1e-10:
            break

    return float(a)


def rgt_common_ground_track_layer(
    inc_deg: float,
    N_P: int,
    N_D: int,
    L: int,
    earth: EarthConstants,
    *,
    a_km: float,
    dt_s: float,
    raan0_deg: float = 0.0,
    m0_deg: float = 0.0,
) -> Layer:
    """Build the L orbital slots of one seed (a common-ground-track block).

    For the APC decomposition (Lee et al. 2020), all L satellites of a seed share a single
    repeating ground track. Slot k arrives at each ground-track point exactly k*dt_s later than
    slot 0 (the seed satellite). That time offset maps to:

        RAAN_k  = RAAN_0 + k * (w_E - Omega_dot) * dt_s
        M0_k    = M0_0  - k * (M_dot + omega_dot) * dt_s

    where Omega_dot / omega_dot are the secular-J2 RAAN / argument-of-perigee rates and M_dot is
    Lee's J2-corrected mean-anomaly rate (Eq. 5), all evaluated at (a_km, inc_deg). Because the
    rates use this seed's own inclination, the circulant V[j,k] = v0[(j-k) mod L] is exact for every
    family of a shared-T_r ladder (Lee Eq. 37 is the special case where this orbit is the reference).

    Sign convention: Earth rotates eastward, so the ECEF longitude of the ascending node is
    RAAN - w_E*t (core/frames.py). A slot arriving k*dt_s LATER at the same ECEF node therefore
    needs its RAAN shifted EAST by (w_E - Omega_dot)*k*dt_s (the + sign above).

    NOT the same as placing L satellites in one orbital plane: equal-RAAN slots do not share a
    ground track and do not give circulant access profiles.

    Parameters
    ----------
    inc_deg    : inclination, degrees
    N_P, N_D   : repeat orbits / repeat days of this seed's ratio; need not be in lowest terms
                 (45:3 is a valid rung of a 3-day grid: its track repeats every T_r/3, so slots
                 k and k + L/3 coincide)
    L          : number of slots = number of time steps in the shared repeat period T_r
    earth      : EarthConstants
    a_km       : semi-major axis of this seed (from the shared-T_r ladder), km
    dt_s       : shared simulation time step T_r / L, seconds
    raan0_deg  : RAAN of the seed satellite (slot 0) at epoch, degrees
    m0_deg     : mean anomaly of the seed satellite (slot 0) at epoch, degrees

    Returns
    -------
    Layer with L slots. Slot 0 is the seed satellite that is simulated to obtain v0; the full
    layer is used to recover orbital elements of the slots the MILP selects.
    """
    if L <= 0:
        raise ValueError("L must be > 0")

    Omegadot, omegadot, M_dot = lee_j2_rates(a_km, inc_deg, earth)
    T_r = N_D * 2.0 * np.pi / (earth.omega_earth_rad_s - Omegadot)

    k = np.arange(L, dtype=np.float64)
    raan_rad = (np.deg2rad(raan0_deg) + k * (earth.omega_earth_rad_s - Omegadot) * dt_s) % (2.0 * np.pi)
    m0_rad = (np.deg2rad(m0_deg) - k * (M_dot + omegadot) * dt_s) % (2.0 * np.pi)

    elems = OrbitalElements(
        a_km=np.full(L, a_km, dtype=np.float64),
        e=np.zeros(L, dtype=np.float64),
        i_rad=np.full(L, np.deg2rad(inc_deg), dtype=np.float64),
        raan_rad=raan_rad,
        argp_rad=np.zeros(L, dtype=np.float64),
        M0_rad=m0_rad,
    )

    # Record-only metadata (written into the result JSON's family records; nothing reads it back).
    spec: dict = {
        "type":            "rgt_common_ground_track",
        "inc_deg":         float(inc_deg),
        "N_P":             int(N_P),
        "N_D":             int(N_D),
        "L":               int(L),
        "a_km":            float(a_km),
        "T_r_s":           float(T_r),
        "raan0_deg":       float(raan0_deg),
        "m0_deg":          float(m0_deg),
        "Omegadot_rad_s":  float(Omegadot),
        "omegadot_rad_s":  float(omegadot),
        "Mdot_rad_s":      float(M_dot),
        "closure_residual_rad_s": float(N_P * (earth.omega_earth_rad_s - Omegadot) - N_D * (omegadot + M_dot)),
    }
    return Layer(elems=elems, spec=spec)


def seed_layer(layer: Layer) -> Layer:
    """Return a single-satellite Layer containing only slot 0 of a common-ground-track layer.

    Used to simulate v0: only the seed satellite is propagated, since the circulant property
    gives every other column analytically.
    """
    e = layer.elems

    def _first(arr: np.ndarray) -> np.ndarray:
        return np.asarray(arr, dtype=np.float64)[[0]]

    seed_elems = OrbitalElements(
        a_km=_first(e.a_km), e=_first(e.e), i_rad=_first(e.i_rad),
        raan_rad=_first(e.raan_rad), argp_rad=_first(e.argp_rad), M0_rad=_first(e.M0_rad),
    )
    return Layer(elems=seed_elems, spec=dict(layer.spec))
