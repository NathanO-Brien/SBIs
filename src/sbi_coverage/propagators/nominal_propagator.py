"""Nominal analytical propagator: secular-J2 motion following Lee et al. (2020).

With ``SimConfig.use_j2=True`` (the setting the optimizer always uses) RAAN, argument of
perigee and mean anomaly advance at Lee's secular rates (Eqs. 3-5), which makes repeat-ground-track
closure exact. ``use_j2=False`` is plain two-body motion, kept only to replay results archived
before the J2 work. There is no drag model.

Implements the propagator interface expected by simulate.run_simulation():
init_state() and step_state_to_eci_positions().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from sbi_coverage.core.config import EarthConstants, SimConfig
from sbi_coverage.core.elements import OrbitalElements


def mean_motion_rad_s(a_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
    """Two-body mean motion n = sqrt(mu / a^3), rad/s."""
    return np.sqrt(mu_km3_s2 / np.maximum(a_km, 1e-9) ** 3)


def solve_kepler_E(M: np.ndarray, e: np.ndarray, iters: int = 10) -> np.ndarray:
    """
    Solve Kepler's equation M = E - e*sin(E) for E (vectorized Newton).
    """
    M = (M + np.pi) % (2.0 * np.pi) - np.pi
    E = np.copy(M)
    for _ in range(iters):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        E = E - f / np.maximum(fp, 1e-12)
    return (E + np.pi) % (2.0 * np.pi) - np.pi


def elements_to_eci_km(
    a_km: np.ndarray,
    e: np.ndarray,
    i: np.ndarray,
    raan: np.ndarray,
    argp: np.ndarray,
    nu: np.ndarray,
    earth: EarthConstants,
) -> np.ndarray:
    """
    Convert elements + true anomaly to ECI position vectors (km), vectorized.
    """
    p = a_km * (1.0 - e * e)
    r = p / np.maximum(1.0 + e * np.cos(nu), 1e-12)

    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(i), np.sin(i)
    cw, sw = np.cos(argp), np.sin(argp)

    R11 = cO * cw - sO * sw * ci
    R12 = -cO * sw - sO * cw * ci
    R21 = sO * cw + cO * sw * ci
    R22 = -sO * sw + cO * cw * ci
    R31 = sw * si
    R32 = cw * si

    x = R11 * x_p + R12 * y_p
    y = R21 * x_p + R22 * y_p
    z = R31 * x_p + R32 * y_p
    return np.stack([x, y, z], axis=1)


def j2_secular_rates_rad_s(a_km, e, i_rad, earth: EarthConstants) -> tuple[np.ndarray, np.ndarray]:
    """
    Secular rates for RAAN and argument of perigee under J2.
    """
    mu = earth.mu_km3_s2
    Re = earth.r_eq_km
    J2 = earth.j2

    n = mean_motion_rad_s(a_km, mu)
    p = a_km * (1.0 - e * e)
    fac = 1.5 * J2 * (Re * Re) * n / np.maximum(p * p, 1e-12)

    Omegadot = -fac * np.cos(i_rad)
    omegadot = 0.5 * fac * (5.0 * np.cos(i_rad) ** 2 - 1.0)
    return Omegadot, omegadot


def lee_mean_anomaly_rate_rad_s(a_km, e, i_rad, earth: EarthConstants) -> np.ndarray:
    """
    J2-corrected mean-anomaly rate (Lee et al. 2020, Eq. 5):

        Mdot = n * [1 - 1.5 * J2 * (Re/p)^2 * sqrt(1 - e^2) * (1.5 sin^2 i - 1)]

    with p = a (1 - e^2).  Together with the J2 secular rates of RAAN and argument
    of perigee (j2_secular_rates_rad_s), this closes the ground track exactly
    under the secular J2 model: the argument-of-latitude rate of a circular orbit
    is omegadot + Mdot.
    """
    n = mean_motion_rad_s(a_km, earth.mu_km3_s2)
    p = a_km * (1.0 - e * e)
    k = 1.5 * earth.j2 * (earth.r_eq_km / np.maximum(p, 1e-9)) ** 2
    return n * (1.0 - k * np.sqrt(1.0 - e * e) * (1.5 * np.sin(i_rad) ** 2 - 1.0))


@dataclass
class PropState:
    """
    Mutable propagation state for stepwise propagation.
    All arrays are length Nsats.
    """
    a_km: np.ndarray
    e: np.ndarray
    i_rad: np.ndarray
    raan_rad: np.ndarray
    argp_rad: np.ndarray
    M_rad: np.ndarray


def init_state(*, elems0: OrbitalElements, sim: SimConfig, earth: EarthConstants) -> PropState:
    """Initialize stepwise state from the epoch elements (always starts from M0_rad)."""
    return PropState(
        a_km=np.asarray(elems0.a_km, dtype=np.float64).copy(),
        e=np.asarray(elems0.e, dtype=np.float64).copy(),
        i_rad=np.asarray(elems0.i_rad, dtype=np.float64).copy(),
        raan_rad=np.asarray(elems0.raan_rad, dtype=np.float64).copy(),
        argp_rad=np.asarray(elems0.argp_rad, dtype=np.float64).copy(),
        M_rad=np.asarray(elems0.M0_rad, dtype=np.float64).copy(),
    )


def step_state_to_eci_positions(
    *,
    state: PropState,
    dt_s: float,
    sim: SimConfig,
    earth: EarthConstants,
) -> np.ndarray:
    """Advance the propagation state by dt_s and return ECI positions (Nsats x 3)."""
    use_j2 = sim.use_j2

    if use_j2 and dt_s > 0.0:
        Omegadot, omegadot = j2_secular_rates_rad_s(state.a_km, state.e, state.i_rad, earth)
        state.raan_rad = state.raan_rad + Omegadot * dt_s
        state.argp_rad = state.argp_rad + omegadot * dt_s

    # Lee's J2-corrected mean-anomaly rate (Eq. 5) so ground-track closure is exact;
    # J2 off uses the plain two-body mean motion.
    if use_j2:
        Mdot = lee_mean_anomaly_rate_rad_s(state.a_km, state.e, state.i_rad, earth)
    else:
        Mdot = mean_motion_rad_s(state.a_km, earth.mu_km3_s2)
    state.M_rad = state.M_rad + Mdot * dt_s
    state.M_rad = (state.M_rad + np.pi) % (2.0 * np.pi) - np.pi

    # Kepler solve -> true anomaly
    E = solve_kepler_E(state.M_rad, state.e, iters=10)
    sin_nu = (np.sqrt(1.0 - state.e * state.e) * np.sin(E)) / np.maximum(1.0 - state.e * np.cos(E), 1e-12)
    cos_nu = (np.cos(E) - state.e) / np.maximum(1.0 - state.e * np.cos(E), 1e-12)
    nu = np.arctan2(sin_nu, cos_nu)

    return elements_to_eci_km(state.a_km, state.e, state.i_rad, state.raan_rad, state.argp_rad, nu, earth)


def get_metadata(*, sim: SimConfig) -> dict[str, Any]:
    """Return propagator identification and settings for run metadata."""
    return {
        "name": "Nominal_Propagator",
        "epoch0": sim.epoch0,
        "units": {"r": "km", "v": "km/s", "t": "s"},
        "use_j2": bool(sim.use_j2),
    }
