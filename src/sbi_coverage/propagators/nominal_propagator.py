from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from sbi_coverage.core.config import EarthConstants, SimConfig
from sbi_coverage.core.elements import OrbitalElements, SatellitePhysical


def mean_motion_rad_s(a_km: np.ndarray, mu_km3_s2: float) -> np.ndarray:
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


def density_exponential_kg_m3(h_km: np.ndarray, sim: SimConfig) -> np.ndarray:
    rho_scale = float(getattr(sim, "rho_scale", 1.0))
    rho0 = float(getattr(sim, "rho0_kg_m3", 3.614e-13))
    h0_km = float(getattr(sim, "h0_km", 700.0))
    H_km = float(getattr(sim, "H_km", 88.667))
    expo = -((h_km - h0_km) / np.maximum(H_km, 1e-12))
    expo = np.clip(expo, -80.0, 80.0)  # Keep exp() numerically stable.
    return rho_scale * rho0 * np.exp(expo)


def drag_update_a_avg(
    a_km: np.ndarray,
    e: np.ndarray,
    phys: SatellitePhysical,
    sim: SimConfig,
    earth: EarthConstants,
    dt_s: float,
) -> np.ndarray:
    """
    Smooth averaged drag proxy (same as your original file).
    """
    rp_km = a_km * (1.0 - e)
    ra_km = a_km * (1.0 + e)

    hp_km = np.maximum(rp_km - earth.r_eq_km, 0.0)
    ha_km = np.maximum(ra_km - earth.r_eq_km, 0.0)

    rho_p = density_exponential_kg_m3(hp_km, sim)
    rho_a = density_exponential_kg_m3(ha_km, sim)
    rho = 0.5 * (rho_p + rho_a)

    v_m_s = 1000.0 * np.sqrt(earth.mu_km3_s2 / np.maximum(a_km, 1e-9))
    a_drag_m_s2 = 0.5 * rho * (v_m_s**2) / np.maximum(phys.bc_kg_m2, 1e-12)

    a_m = 1000.0 * a_km
    mu_m3_s2 = earth.mu_km3_s2 * (1000.0**3)
    da_dt_m_s = -2.0 * (a_m**2) * v_m_s * a_drag_m_s2 / np.maximum(mu_m3_s2, 1e-30)

    a_m_new = a_m + da_dt_m_s * dt_s
    drag_floor_alt_km = float(getattr(sim, "drag_floor_alt_km", 120.0))
    a_m_new = np.maximum(a_m_new, 1000.0 * (earth.r_eq_km + drag_floor_alt_km))
    return a_m_new / 1000.0


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


def init_state(
    *,
    elems0: OrbitalElements,
    r0_eci_km: np.ndarray,
    v0_eci_km_s: np.ndarray,
    phys: SatellitePhysical,
    sim: SimConfig,
    earth: EarthConstants,
) -> PropState:
    """
    Initialize stepwise state from epoch elements.
    (r0/v0 accepted for interface compatibility but not needed here.)
    """
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
    phys: SatellitePhysical,
    dt_s: float,
    t_s: Optional[float] = None,
    sim: SimConfig,
    earth: EarthConstants,
) -> np.ndarray:
    """
    Advance the propagation state by dt_s and return ECI positions (Nsats x 3).
    """
    use_drag = bool(getattr(sim, "use_drag", False))
    use_j2 = bool(getattr(sim, "use_j2", True))

    if use_drag and dt_s > 0.0:
        state.a_km = drag_update_a_avg(state.a_km, state.e, phys, sim, earth, dt_s=dt_s)

    # Optional J2 secular drift
    if use_j2 and dt_s > 0.0:
        Omegadot, omegadot = j2_secular_rates_rad_s(state.a_km, state.e, state.i_rad, earth)
        state.raan_rad = state.raan_rad + Omegadot * dt_s
        state.argp_rad = state.argp_rad + omegadot * dt_s

    # Mean anomaly update (with updated a(t))
    n = mean_motion_rad_s(state.a_km, earth.mu_km3_s2)
    state.M_rad = state.M_rad + n * dt_s
    state.M_rad = (state.M_rad + np.pi) % (2.0 * np.pi) - np.pi

    # Kepler solve -> true anomaly
    E = solve_kepler_E(state.M_rad, state.e, iters=10)
    sin_nu = (np.sqrt(1.0 - state.e * state.e) * np.sin(E)) / np.maximum(1.0 - state.e * np.cos(E), 1e-12)
    cos_nu = (np.cos(E) - state.e) / np.maximum(1.0 - state.e * np.cos(E), 1e-12)
    nu = np.arctan2(sin_nu, cos_nu)

    return elements_to_eci_km(
        state.a_km,
        state.e,
        state.i_rad,
        state.raan_rad,
        state.argp_rad,
        nu,
        earth,
    )


def get_metadata(*, state: PropState, sim: SimConfig, earth: EarthConstants, phys: SatellitePhysical) -> dict[str, Any]:
    return {
        "name": "Nominal_Propagator",
        "epoch0": getattr(sim, "epoch0", "J2000"),
        "units": {"r": "km", "v": "km/s", "t": "s"},
        "use_j2": bool(getattr(sim, "use_j2", True)),
        "use_drag": bool(getattr(sim, "use_drag", False)),
    }
