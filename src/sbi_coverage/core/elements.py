"""Vectorized orbital element container.

Each field holds a NumPy array of shape (n_sats,), so a single instance
describes an entire constellation.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OrbitalElements:
    """Classical orbital elements for a set of satellites.

    All angular quantities are in radians. Mean anomaly at epoch (M0_rad) is
    the canonical anomaly used by the seed/layer builders.
    """
    a_km: np.ndarray        # semi-major axis
    e: np.ndarray           # eccentricity
    i_rad: np.ndarray       # inclination
    raan_rad: np.ndarray    # right ascension of the ascending node
    argp_rad: np.ndarray    # argument of perigee
    M0_rad: np.ndarray      # mean anomaly at epoch

    # Optional true anomaly (informational only; the propagator always starts from M0_rad).
    ta_rad: np.ndarray | None = None

