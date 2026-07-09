"""Reference frame conversions between Earth-centered inertial (ECI) and
Earth-centered Earth-fixed (ECEF) coordinates.

Uses a simple uniform-rotation model: the frames are aligned at t = 0 and the
rotation angle grows as theta = omega_earth * t. This is sufficient for
coverage studies; a full ERA/GMST epoch model is not required.
"""
import numpy as np
from .config import EarthConstants


def rot_z(theta: float) -> np.ndarray:
    """Return the 3x3 rotation matrix for a rotation of `theta` radians about
    the +Z axis (frame rotation convention)."""
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def ecef_to_eci(points_ecef_km: np.ndarray, t_s: float, earth: EarthConstants) -> np.ndarray:
    """Rotate Earth-fixed points (N, 3) into the inertial frame at time t_s
    seconds after frame alignment.

    Earth rotates eastward (+omega about +Z), so a fixed ground point's
    inertial longitude increases with time.
    """
    theta = earth.omega_earth_rad_s * t_s
    R = rot_z(theta)
    return points_ecef_km @ R


def eci_to_ecef(points_eci_km: np.ndarray, t_s: float, earth: EarthConstants) -> np.ndarray:
    """Rotate inertial points (N, 3) into the Earth-fixed frame at time t_s
    seconds after frame alignment.

    Verified against a geostationary orbit (stationary ECEF position) and
    prograde-LEO westward node drift; see scripts/verify_frames.py.
    """
    theta = earth.omega_earth_rad_s * t_s
    R = rot_z(theta)
    return points_eci_km @ R.T
