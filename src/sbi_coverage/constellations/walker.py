"""Walker delta and Walker star constellation layer generators."""
from __future__ import annotations

import numpy as np

from ..core.config import CoverageConfig, EarthConstants
from ..core.elements import OrbitalElements, SatellitePhysical
from .layer import Layer


def _validate_tp(*, n_sats_total: int, n_planes: int) -> tuple[int, int, int]:
    """Validate Walker T/P inputs and return (T, P, S)."""
    T = int(n_sats_total)
    P = int(n_planes)
    if T <= 0:
        raise ValueError("Walker layer: n_sats_total must be > 0")
    if P <= 0:
        raise ValueError("Walker layer: n_planes must be > 0")
    if T % P != 0:
        raise ValueError(
            f"Walker layer: n_sats_total must be divisible by n_planes. Got T={T}, P={P}."
        )
    S = T // P
    if S <= 0:
        raise ValueError("Walker layer: computed sats_per_plane must be > 0")
    return T, P, S


def walker_delta_layer(
    earth: EarthConstants,
    *,
    cov: CoverageConfig,
    n_sats_total: int,
    n_planes: int,
    constellation_alt_km: float,
    inc_deg: float,
    f: int = 1,
    ecc: float = 0.0,
    raan_offset_deg: float = 0.0,
    m0_offset_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
) -> Layer:
    """
    Build a Walker-Delta layer from discrete catalog inputs (T/P/F + altitude + inclination).

    Notation (discrete catalog):
      - T = n_sats_total
      - P = n_planes
      - S = T/P satellites per plane
      - F = f (integer phasing parameter)

    Elements:
      - Circular / near-circular assumed; argp fixed to 0 (ignored).
      - RAAN is evenly spaced: Ω_p = 360/P * p + raan_offset
      - Mean anomaly:
            M(p,s) = 360/S * s + (F * p) * (360/T) + m0_offset

    Returns:
      Layer(elems=..., phys=..., spec=...) compatible with combine_layers().
    """
    T, P, S = _validate_tp(n_sats_total=n_sats_total, n_planes=n_planes)
    inc_deg = float(inc_deg)

    # --- indices ---
    plane_idx = np.repeat(np.arange(P), S)  # 0..P-1 repeated S times
    sat_idx = np.tile(np.arange(S), P)     # 0..S-1 tiled across planes

    # --- elements ---
    a_km = float(earth.r_eq_km + float(constellation_alt_km))

    raan_deg = (360.0 / P) * plane_idx + float(raan_offset_deg)
    phase_deg = (int(f) * plane_idx) * (360.0 / T)
    M_deg = (360.0 / S) * sat_idx + phase_deg + float(m0_offset_deg)

    elems = OrbitalElements(
        a_km=np.full(T, a_km, dtype=np.float64),
        e=np.full(T, float(ecc), dtype=np.float64),
        i_rad=np.deg2rad(np.full(T, inc_deg, dtype=np.float64)),
        raan_rad=np.deg2rad(raan_deg.astype(np.float64)),
        argp_rad=np.zeros(T, dtype=np.float64),  # fixed (near-circular assumption)
        M0_rad=np.deg2rad(M_deg.astype(np.float64)),
    )

    phys = SatellitePhysical(
        bc_kg_m2=np.full(T, float(bc_kg_m2), dtype=np.float64)
    )

    spec = {
        "type": "walker_delta",
        "notation": "T/P/F",
        "n_total": int(T),
        "n_planes": int(P),
        "sats_per_plane": int(S),
        "constellation_alt_km": float(constellation_alt_km),
        "a_km": float(a_km),
        "inc_deg": float(inc_deg),
        "e": float(ecc),
        "f": int(f),
        "intercept_alt_km": float(cov.intercept_alt_km),
        "max_range_km": float(cov.max_range_km),
        "min_elev_deg": float(cov.min_elev_deg),
        "raan_offset_deg": float(raan_offset_deg),
        "m0_offset_deg": float(m0_offset_deg),
        "argp_deg": 0.0,
        "bc_kg_m2": float(bc_kg_m2),
    }

    return Layer(elems=elems, phys=phys, spec=spec)


def walker_star_layer(
    earth: EarthConstants,
    *,
    cov: CoverageConfig,
    n_sats_total: int,
    n_planes: int,
    constellation_alt_km: float,
    inc_deg: float = 90.0,
    f: int = 1,
    ecc: float = 0.0,
    raan_offset_deg: float = 0.0,
    m0_offset_deg: float = 0.0,
    bc_kg_m2: float = 80.0,
) -> Layer:
    """
    Build a Walker-Star layer from discrete catalog inputs (T/P/F + altitude + inclination).

    This implementation follows the same discrete phasing law as Walker-Delta (plane-dependent
    mean anomaly shift), but uses a Walker-Star style RAAN spacing of 180/P.

    Common convention:
      - Star RAAN increment uses 180 degrees rather than 360 degrees because RAAN and RAAN+180
        describe the same geometric plane with opposite direction of motion; star constellations
        often distribute planes over 180 degrees.

    Elements:
      - Circular / near-circular assumed; argp fixed to 0.
      - RAAN: Ω_p = 180/P * p + raan_offset
      - Mean anomaly:
            M(p,s) = 360/S * s + (F * p) * (360/T) + m0_offset
    """
    T, P, S = _validate_tp(n_sats_total=n_sats_total, n_planes=n_planes)
    inc_deg = float(inc_deg)

    # --- indices ---
    plane_idx = np.repeat(np.arange(P), S)
    sat_idx = np.tile(np.arange(S), P)

    # --- elements ---
    a_km = float(earth.r_eq_km + float(constellation_alt_km))

    raan_deg = (180.0 / P) * plane_idx + float(raan_offset_deg)
    phase_deg = (int(f) * plane_idx) * (360.0 / T)
    M_deg = (360.0 / S) * sat_idx + phase_deg + float(m0_offset_deg)

    elems = OrbitalElements(
        a_km=np.full(T, a_km, dtype=np.float64),
        e=np.full(T, float(ecc), dtype=np.float64),
        i_rad=np.deg2rad(np.full(T, inc_deg, dtype=np.float64)),
        raan_rad=np.deg2rad(raan_deg.astype(np.float64)),
        argp_rad=np.zeros(T, dtype=np.float64),
        M0_rad=np.deg2rad(M_deg.astype(np.float64)),
    )

    phys = SatellitePhysical(
        bc_kg_m2=np.full(T, float(bc_kg_m2), dtype=np.float64)
    )

    spec = {
        "type": "walker_star",
        "notation": "T/P/F",
        "raan_span_deg": 180.0,
        "n_total": int(T),
        "n_planes": int(P),
        "sats_per_plane": int(S),
        "constellation_alt_km": float(constellation_alt_km),
        "a_km": float(a_km),
        "inc_deg": float(inc_deg),
        "e": float(ecc),
        "f": int(f),
        "intercept_alt_km": float(cov.intercept_alt_km),
        "max_range_km": float(cov.max_range_km),
        "min_elev_deg": float(cov.min_elev_deg),
        "raan_offset_deg": float(raan_offset_deg),
        "m0_offset_deg": float(m0_offset_deg),
        "argp_deg": 0.0,
        "bc_kg_m2": float(bc_kg_m2),
    }

    return Layer(elems=elems, phys=phys, spec=spec)
