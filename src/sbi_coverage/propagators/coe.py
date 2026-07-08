"""
Shared orbital elements -> ECI r,v conversion helpers (km / km/s).

IMPORTANT:
- Your repo uses "vectorized OrbitalElements": elems.a_km, elems.e, ... are arrays (Nsats,).
- This file supports BOTH:
  (a) scalar OrbitalElements (single satellite)
  (b) vectorized OrbitalElements (arrays for many satellites)

Anomaly handling:
- If elems.ta_rad is not None, use it (true anomaly, scalar or array)
- Else derive nu from mean anomaly elems.M0_rad using Kepler solve (vectorized Newton)
"""

from __future__ import annotations

import numpy as np

from sbi_coverage.core.elements import OrbitalElements
from sbi_coverage.core.config import EarthConstants


def _wrap_to_pi(x: np.ndarray) -> np.ndarray:
    return (x + np.pi) % (2.0 * np.pi) - np.pi


def _solve_kepler_E_vec(M: np.ndarray, e: np.ndarray, iters: int = 10) -> np.ndarray:
    """
    Vectorized Newton solve for elliptic Kepler's equation:
        M = E - e sin E

    Args:
        M: mean anomaly [rad], shape (N,)
        e: eccentricity, shape (N,)
        iters: fixed Newton iterations

    Returns:
        E: eccentric anomaly [rad], shape (N,)
    """
    # Wrap M for numerical stability
    M = _wrap_to_pi(M)
    E = M.copy()

    # Newton iterations
    for _ in range(iters):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        E = E - f / np.maximum(fp, 1e-12)

    return _wrap_to_pi(E)


def _true_anomaly_from_E_vec(E: np.ndarray, e: np.ndarray) -> np.ndarray:
    """
    Vectorized conversion: eccentric anomaly E -> true anomaly nu (elliptic).
    """
    # For circular-ish, nu ~ E
    near_circ = e < 1e-14
    nu = np.empty_like(E)

    if np.any(~near_circ):
        Ee = E[~near_circ]
        ee = e[~near_circ]
        s = np.sin(Ee / 2.0)
        c = np.cos(Ee / 2.0)
        num = np.sqrt(1.0 + ee) * s
        den = np.sqrt(1.0 - ee) * c
        nu[~near_circ] = 2.0 * np.arctan2(num, den)

    if np.any(near_circ):
        nu[near_circ] = E[near_circ]

    return _wrap_to_pi(nu)


def _pqw_to_eci_rotation_cosines(raan: np.ndarray, inc: np.ndarray, argp: np.ndarray):
    """
    Return expanded direction cosines for R = R3(raan) * R1(inc) * R3(argp)
    for vectorized arrays.
    """
    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)

    R11 = cO * cw - sO * sw * ci
    R12 = -cO * sw - sO * cw * ci

    R21 = sO * cw + cO * sw * ci
    R22 = -sO * sw + cO * cw * ci

    R31 = sw * si
    R32 = cw * si

    return R11, R12, R21, R22, R31, R32


def elements_to_eci_rv_km(
    elems: OrbitalElements, earth: EarthConstants
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert orbital elements to ECI Cartesian state at epoch.

    Returns:
      - r_eci_km: (3,) for scalar elems or (N,3) for vectorized elems
      - v_eci_km_s: (3,) for scalar elems or (N,3) for vectorized elems
    """
    mu = float(earth.mu_km3_s2)

    # Pull fields as arrays for unified handling
    a = np.asarray(elems.a_km, dtype=float)
    e = np.asarray(elems.e, dtype=float)
    inc = np.asarray(elems.i_rad, dtype=float)
    raan = np.asarray(elems.raan_rad, dtype=float)
    argp = np.asarray(elems.argp_rad, dtype=float)

    # Determine if vectorized
    vectorized = a.ndim != 0

    # Determine true anomaly nu
    ta = getattr(elems, "ta_rad", None)
    if ta is not None:
        nu = np.asarray(ta, dtype=float)
        if nu.ndim == 0 and vectorized:
            # If someone passes scalar ta but vectorized other fields, broadcast
            nu = np.full_like(a, float(nu))
    else:
        M0 = np.asarray(elems.M0_rad, dtype=float)
        if M0.ndim == 0 and vectorized:
            M0 = np.full_like(a, float(M0))
        E = _solve_kepler_E_vec(M0, e, iters=10) if vectorized else _solve_kepler_E_vec(np.array([float(M0)]), np.array([float(e)]), iters=10)[0]
        if vectorized:
            nu = _true_anomaly_from_E_vec(E, e)
        else:
            nu = _true_anomaly_from_E_vec(np.array([float(E)]), np.array([float(e)]))[0]

    # Semi-latus rectum p and radius r
    p = a * (1.0 - e * e)
    r = p / np.maximum(1.0 + e * np.cos(nu), 1e-12)

    # Perifocal position
    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)

    # Perifocal velocity
    # v_pf = sqrt(mu/p) * [-sin nu, e + cos nu, 0]
    fac = np.sqrt(mu / np.maximum(p, 1e-12))
    vx_p = -fac * np.sin(nu)
    vy_p = fac * (e + np.cos(nu))

    if not vectorized:
        # Scalar: compute rotation cosines as scalars via vector form
        raan_s = float(raan)
        inc_s = float(inc)
        argp_s = float(argp)

        cO, sO = np.cos(raan_s), np.sin(raan_s)
        ci, si = np.cos(inc_s), np.sin(inc_s)
        cw, sw = np.cos(argp_s), np.sin(argp_s)

        R11 = cO * cw - sO * sw * ci
        R12 = -cO * sw - sO * cw * ci
        R21 = sO * cw + cO * sw * ci
        R22 = -sO * sw + cO * cw * ci
        R31 = sw * si
        R32 = cw * si

        rx = R11 * float(x_p) + R12 * float(y_p)
        ry = R21 * float(x_p) + R22 * float(y_p)
        rz = R31 * float(x_p) + R32 * float(y_p)

        vx = R11 * float(vx_p) + R12 * float(vy_p)
        vy = R21 * float(vx_p) + R22 * float(vy_p)
        vz = R31 * float(vx_p) + R32 * float(vy_p)

        r_eci = np.array([rx, ry, rz], dtype=float)
        v_eci = np.array([vx, vy, vz], dtype=float)
        return r_eci, v_eci

    # Vectorized rotation
    R11, R12, R21, R22, R31, R32 = _pqw_to_eci_rotation_cosines(raan, inc, argp)

    rx = R11 * x_p + R12 * y_p
    ry = R21 * x_p + R22 * y_p
    rz = R31 * x_p + R32 * y_p

    vx = R11 * vx_p + R12 * vy_p
    vy = R21 * vx_p + R22 * vy_p
    vz = R31 * vx_p + R32 * vy_p

    r_eci = np.stack([rx, ry, rz], axis=1)
    v_eci = np.stack([vx, vy, vz], axis=1)
    return r_eci, v_eci
