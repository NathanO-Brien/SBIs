import numpy as np
from .config import EarthConstants

def rot_z(theta: float) -> np.ndarray:
    c = np.cos(theta)
    s = np.sin(theta)
    return np.array([[ c,  s, 0.0],
                     [-s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)

def ecef_to_eci(points_ecef_km: np.ndarray, t_s: float, earth: EarthConstants) -> np.ndarray:
    """
    V1: use theta = omega_earth * t to rotate Earth-fixed points into inertial.
    For coverage studies, this is typically sufficient. Upgrade to ERA/GMST later if needed.
    """
    theta = earth.omega_earth_rad_s * t_s
    R = rot_z(theta)
    return points_ecef_km @ R.T  # Nx3

def eci_to_ecef(points_eci_km: np.ndarray, t_s: float, earth: EarthConstants) -> np.ndarray:
    theta = earth.omega_earth_rad_s * t_s
    R = rot_z(theta)
    return points_eci_km @ R  # Nx3
