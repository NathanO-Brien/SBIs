from __future__ import annotations

from math import gcd

import numpy as np

from ..core.config import EarthConstants
from ..core.elements import OrbitalElements, SatellitePhysical
from ..constellations.layer import Layer


def _a_km_from_np_nd(N_P: int, N_D: int, earth: EarthConstants) -> float:
    T_sid = (2.0 * np.pi) / earth.omega_earth_rad_s
    T_S   = N_D * T_sid / N_P
    return float((earth.mu_km3_s2 * (T_S / (2.0 * np.pi))**2) ** (1.0 / 3.0))


def a_km_j2_rgt(N_P: int, N_D: int, inc_deg: float, earth: EarthConstants) -> float:
    """J2-corrected semi-major axis for an RGT orbit (Newton-Raphson).

    Finds a (km) satisfying the exact J2 repeating-ground-track condition for
    our frames.py ECEF convention (ECEF_lon = RAAN + ω_E·t):

        N_P · (ω_E + Ω̇(a)) = N_D · (n_kep(a) + ω̇(a))

    At the returned a, T_r = N_P · 2π/(n_kep + ω̇) = N_D · 2π/(ω_E + Ω̇) and
    the orbit repeats exactly in ECEF after T_r seconds under the secular J2
    propagator, making the circulant V matrix exact for ALL entries (including
    wrap-around ones).  The Keplerian a from _a_km_from_np_nd() is used as the
    initial guess.

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
    i_rad = np.deg2rad(inc_deg)
    cos_i = np.cos(i_rad)
    cos2_i = cos_i * cos_i

    a = _a_km_from_np_nd(N_P, N_D, earth)  # Keplerian initial guess

    for _ in range(100):
        n = np.sqrt(earth.mu_km3_s2 / a**3)
        fac = 1.5 * earth.j2 * (earth.r_eq_km / a)**2 * n
        Omegadot = -fac * cos_i
        omegadot = 0.5 * fac * (5.0 * cos2_i - 1.0)

        f = N_P * (earth.omega_earth_rad_s + Omegadot) - N_D * (n + omegadot)

        # Numerical derivative via central difference (1 m step to avoid cancellation)
        da = max(a * 1e-7, 1e-4)
        a2 = a + da
        n2 = np.sqrt(earth.mu_km3_s2 / a2**3)
        fac2 = 1.5 * earth.j2 * (earth.r_eq_km / a2)**2 * n2
        Omega2 = -fac2 * cos_i
        omega2 = 0.5 * fac2 * (5.0 * cos2_i - 1.0)
        f2 = N_P * (earth.omega_earth_rad_s + Omega2) - N_D * (n2 + omega2)

        df_da = (f2 - f) / da
        if abs(df_da) < 1e-40:
            break

        step = f / df_da
        a = float(np.clip(a - step, earth.r_eq_km + 80.0, earth.r_eq_km + 3000.0))

        if abs(step) < 1e-8:
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

        RAAN_k  = RAAN_0 - k · (ω_E + Ω̇) · T_r / L   (note: + not -, see frames.py sign convention)
        M₀_k    = M₀_0  − k · (2π·N_P/L + ω̇·T_r/L)   (compensates Keplerian advance AND J2 argp drift)

    where T_r = N_D · T_sidereal, Ω̇ is the J2 secular RAAN rate, and ω̇ is the J2
    secular argument-of-perigee rate.

    NOTE on sign: frames.py implements eci_to_ecef as r_eci @ rot_z(+ω_E·t), so the
    apparent ECEF longitude of the ascending node is RAAN + ω_E·t (not RAAN - ω_E·t).
    With J2, RAAN drifts at Ω̇ (negative for prograde), giving ECEF_AN rate = ω_E + Ω̇.
    This gives the + sign in the RAAN slot formula.  All corrections are zeroed when
    use_j2=False to match a propagator running without J2.

    NOTE: This is NOT the same as rgt_layer(n_planes=1), which places all L satellites
    in a single orbital plane (RAAN identical for all slots).  Equal-RAAN slots do NOT
    share a common ground track and do NOT produce circulant access profiles.

    Parameters
    ----------
    inc_deg    : inclination (degrees)
    N_P        : repeat orbits  (numerator of the RGT ratio)
    N_D        : repeat days    (denominator of the RGT ratio)
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
    if gcd(N_P, N_D) != 1:
        raise ValueError(
            f"N_P={N_P} and N_D={N_D} must be coprime (in lowest terms). "
            f"Reduce to {N_P//gcd(N_P,N_D)}:{N_D//gcd(N_P,N_D)} first."
        )
    if L <= 0:
        raise ValueError("L must be > 0")

    i_rad = np.deg2rad(inc_deg)

    if use_j2:
        # Use shared reference a_km when provided (multi-inclination constellations).
        # All sub-constellations must share the same a_km so they share the same
        # repeat period T_r, keeping the circulant V matrix exact across the sweep.
        # Fall back to per-inclination J2-correct a_km only for single-inclination use.
        if a_km_override is not None:
            a_km = float(a_km_override)
        else:
            a_km = a_km_j2_rgt(N_P, N_D, inc_deg, earth)
        n_kep    = np.sqrt(earth.mu_km3_s2 / a_km**3)
        fac      = 1.5 * earth.j2 * (earth.r_eq_km / a_km)**2 * n_kep
        Omegadot = -fac * np.cos(i_rad)
        omegadot =  0.5 * fac * (5.0 * np.cos(i_rad)**2 - 1.0)
        T_r      = N_P * 2.0 * np.pi / (n_kep + omegadot)
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
    #   RAAN_k = RAAN_0 − k · (ω_E + Ω̇_inc) · dt_slot
    #   M_k    = M_0    − k · (n_kep + ω̇_inc) · dt_slot
    #
    # dt_slot is the shared simulation dt (T_r_ref / L).  Using the inclination-
    # specific secular rates (Ω̇_inc, ω̇_inc) ensures the circulant V[j,k] = v0[(j-k) mod L]
    # is correct for ALL inclinations, not only the reference one.  Lee Eq. 37
    # (2π·N_D/L, 2π·N_P/L) is the special case where this orbit IS the reference
    # inclination and T_r_ref equals this orbit's own T_r.
    _dt_slot = dt_s if dt_s is not None else T_r / L

    raan_rad = np.deg2rad(raan0_deg) - k * (earth.omega_earth_rad_s + Omegadot) * _dt_slot
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
