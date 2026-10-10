"""RGT orbital slot construction: common-ground-track layers, J2-adjusted
semi-major axis, and seed satellite layers.
"""
from __future__ import annotations

from math import gcd

import numpy as np

from ..core.config import EarthConstants
from ..core.elements import OrbitalElements, SatellitePhysical
from ..constellations.layer import Layer
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
    raan0_deg: float = 0.0,
    m0_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
    use_j2: bool = True,
    a_km_override: float | None = None,
    dt_s: float | None = None,
) -> Layer:
    """Build the L orbital slots of one RGT common-ground-track sub-constellation.

    For the APC decomposition (Lee et al. 2020), all L satellites in a sub-constellation
    share a single repeating ground track.  Slot k arrives at each ground track point
    exactly k·Δt later than slot 0 (the seed satellite).  This time offset maps to:

        RAAN_k  = RAAN_0 + k · (ω_E − Ω̇) · dt
        M₀_k    = M₀_0  − k · (Ṁ + ω̇) · dt

    where dt = T_r / L, Ω̇ / ω̇ are the J2 secular RAAN / argument-of-perigee rates and
    Ṁ is Lee's J2-corrected mean-anomaly rate (Eq. 5; plain mean motion when
    use_j2=False).  With J2 on, T_r = N_D · 2π / (ω_E − Ω̇); with J2 off,
    T_r = N_D · T_sidereal.

    NOTE on sign: Earth rotates eastward, so a ground-track point's inertial
    longitude increases with time; the ECEF longitude of the ascending node is
    RAAN − ω_E·t (frames.py convention, verified in scripts/verify_frames.py).
    A slot arriving k·Δt LATER at the same ECEF node therefore needs its RAAN
    shifted EAST by (ω_E − Ω̇)·k·Δt — the + sign in the RAAN formula.  With J2,
    RAAN drifts at Ω̇ (negative for prograde), giving ECEF node rate Ω̇ − ω_E.
    All corrections are zeroed when use_j2=False to match a propagator running
    without J2.

    NOTE: This is NOT the same as rgt_layer(n_planes=1), which places all L satellites
    in a single orbital plane (RAAN identical for all slots).  Equal-RAAN slots do NOT
    share a common ground track and do NOT produce circulant access profiles.

    Parameters
    ----------
    inc_deg    : inclination (degrees)
    N_P        : repeat orbits  (numerator of the RGT ratio)
    N_D        : repeat days    (denominator of the RGT ratio); the ratio need not be in
                 lowest terms (45:3 is a valid rung of a 3-day grid: its track
                 repeats every T_r/3, so slots k and k + L/3 coincide)
    L          : number of orbital slots = number of time steps in T_r
                 (= round(T_r / dt_s); must match the simulation time axis)
    earth      : EarthConstants
    raan0_deg  : RAAN of the seed satellite (slot 0) at epoch, degrees
    m0_deg     : mean anomaly of the seed satellite (slot 0) at epoch, degrees
    bc_kg_m2   : ballistic coefficient for all slots (drag model, kg/m²)

    Returns
    -------
    Layer with L satellites.  Slot 0 (index 0) is the seed satellite used to
    compute v_0.  The full layer is used to recover orbital elements of BILP-
    selected slots after solving.
    """
    if L <= 0:
        raise ValueError("L must be > 0")

    i_rad = np.deg2rad(inc_deg)

    if use_j2:
        # a_km_override: seed families (Lee/seed_families.py) pass the exact
        # (a, i) pair solved for a shared T_r.  Otherwise solve closure at this
        # inclination (single-family use).
        if a_km_override is not None:
            a_km = float(a_km_override)
        else:
            a_km = a_km_j2_rgt(N_P, N_D, inc_deg, earth)
        Omegadot, omegadot, Mdot = lee_j2_rates(a_km, inc_deg, earth)
        n_kep    = Mdot          # along-track mean-anomaly rate incl. Lee's J2 term
        T_r      = N_D * 2.0 * np.pi / (earth.omega_earth_rad_s - Omegadot)
    else:
        a_km     = _a_km_from_np_nd(N_P, N_D, earth)
        n_kep    = np.sqrt(earth.mu_km3_s2 / a_km**3)
        Omegadot = 0.0
        omegadot = 0.0
        T_r      = N_D * (2.0 * np.pi / earth.omega_earth_rad_s)

    k = np.arange(L, dtype=np.float64)

    # Slot spacing — pre-simplified form of Lee (2020) Eq. 37.
    # Slot k must be exactly one simulation time-step (dt_slot) behind slot k-1
    # in both ECEF longitude and argument of latitude:
    #
    #   RAAN_k = RAAN_0 + k · (ω_E − Ω̇_inc) · dt_slot
    #   M_k    = M_0    − k · (n_kep + ω̇_inc) · dt_slot
    #
    # dt_slot is the shared simulation dt (T_r_ref / L).  Using the inclination-
    # specific secular rates (Ω̇_inc, ω̇_inc) ensures the circulant V[j,k] = v0[(j-k) mod L]
    # is correct for ALL inclinations, not only the reference one.  Lee Eq. 37
    # (2π·N_D/L, 2π·N_P/L) is the special case where this orbit IS the reference
    # inclination and T_r_ref equals this orbit's own T_r.
    _dt_slot = dt_s if dt_s is not None else T_r / L

    raan_rad = np.deg2rad(raan0_deg) + k * (earth.omega_earth_rad_s - Omegadot) * _dt_slot
    raan_rad = raan_rad % (2.0 * np.pi)

    m0_rad = np.deg2rad(m0_deg) - k * (n_kep + omegadot) * _dt_slot
    m0_rad = m0_rad % (2.0 * np.pi)

    elems = OrbitalElements(
        a_km    = np.full(L, a_km,  dtype=np.float64),
        e       = np.zeros(L,       dtype=np.float64),
        i_rad   = np.full(L, np.deg2rad(inc_deg), dtype=np.float64),
        raan_rad= raan_rad,
        argp_rad= np.zeros(L,       dtype=np.float64),
        M0_rad  = m0_rad,
    )
    phys = SatellitePhysical(bc_kg_m2=np.full(L, float(bc_kg_m2), dtype=np.float64))

    spec: dict = {
        "type":            "rgt_common_ground_track",
        "inc_deg":         float(inc_deg),
        "N_P":             int(N_P),
        "N_D":             int(N_D),
        "L":               int(L),
        "a_km":            float(a_km),
        "constellation_alt_km": float(a_km - earth.r_eq_km),
        "T_r_s":           float(T_r),
        "T_S_s":           float(T_r / N_P),
        "raan0_deg":       float(raan0_deg),
        "m0_deg":          float(m0_deg),
        "bc_kg_m2":        float(bc_kg_m2),
        "dt_s_implied":    float(T_r / L),
        "use_j2":          bool(use_j2),
        "Omegadot_rad_s":  float(Omegadot),
        "omegadot_rad_s":  float(omegadot),
        "Mdot_rad_s":      float(n_kep),
        "closure_residual_rad_s": (
            float(N_P * (earth.omega_earth_rad_s - Omegadot) - N_D * (omegadot + n_kep))
            if use_j2 else 0.0
        ),
    }

    return Layer(elems=elems, phys=phys, spec=spec)


def seed_layer(layer: Layer) -> Layer:
    """Return a single-satellite Layer containing only slot 0 of a common-ground-track layer.

    Use this to run the seed simulation that produces v_0 — simulating all L slots is
    unnecessary because the APC circulant property derives all other columns analytically.
    """
    e = layer.elems
    s = layer.phys

    def _first(arr: np.ndarray) -> np.ndarray:
        """Return the first element as a length-1 float64 array."""
        return np.asarray(arr, dtype=np.float64)[[0]]

    seed_elems = OrbitalElements(
        a_km    = _first(e.a_km),
        e       = _first(e.e),
        i_rad   = _first(e.i_rad),
        raan_rad= _first(e.raan_rad),
        argp_rad= _first(e.argp_rad),
        M0_rad  = _first(e.M0_rad),
    )
    seed_phys = SatellitePhysical(
        bc_kg_m2=_first(s.bc_kg_m2)
    )
    return Layer(elems=seed_elems, phys=seed_phys, spec=dict(layer.spec))


def rgt_subconstellation_layers(
    inclinations_deg: list[float],
    N_P: int,
    N_D: int,
    L: int,
    earth: EarthConstants,
    *,
    raan0_deg: float = 0.0,
    m0_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
    use_j2: bool = True,
    a_km_override: float | None = None,
    dt_s: float | None = None,
) -> list[Layer]:
    """Build one common-ground-track Layer per inclination.

    Each element of the returned list is one sub-constellation.  Use seed_layer()
    on each to extract the single-satellite layer for v_0 simulation, or pass the
    full layers to recover orbital elements of BILP-selected slots.

    Parameters
    ----------
    inclinations_deg : list of inclinations, one per sub-constellation
    N_P, N_D, L      : shared RGT parameters (all sub-constellations must share T_r)
    earth            : EarthConstants
    raan0_deg        : seed RAAN for all sub-constellations (degrees)
    m0_deg           : seed mean anomaly for all sub-constellations (degrees)
    bc_kg_m2         : ballistic coefficient (kg/m²)
    a_km_override    : shared semi-major axis (km) to use for all sub-constellations
                       when use_j2=True; pass the J2-correct a_km at the reference
                       inclination so all seeds share the same repeat period T_r.

    Returns
    -------
    List[Layer], one per inclination, each containing L slots.
    """
    return [
        rgt_common_ground_track_layer(
            inc_deg=float(inc),
            N_P=N_P, N_D=N_D, L=L,
            earth=earth,
            raan0_deg=raan0_deg,
            m0_deg=m0_deg,
            bc_kg_m2=bc_kg_m2,
            use_j2=use_j2,
            a_km_override=a_km_override,
            dt_s=dt_s,
        )
        for inc in inclinations_deg
    ]
