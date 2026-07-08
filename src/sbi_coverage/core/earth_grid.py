import numpy as np
from .config import EarthConstants


def fibonacci_sphere_points(n: int) -> np.ndarray:
    """
    Returns Nx3 unit vectors approximately evenly distributed on a sphere.
    """
    i = np.arange(n, dtype=np.float64)
    phi = (1 + 5**0.5) / 2  # golden ratio
    theta = 2 * np.pi * i / phi
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(np.maximum(0.0, 1 - z*z))
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.stack([x, y, z], axis=1)

def points_ecef_with_normals_latlon_km(
    n: int,
    earth: EarthConstants,
    *,
    shell_alt_km: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Returns:
      pts_ecef_km: (N,3) ECEF points on spherical Earth (km)
      n_hat_ecef : (N,3) unit normals (dimensionless)
      lat_deg    : (N,)
      lon_deg    : (N,) in [-180, 180]
      meta       : dict (n_total)
    """
    u = fibonacci_sphere_points(n)      # unit vectors

    # Points on a spherical shell at Earth radius + optional altitude
    r_shell_km = float(earth.r_eq_km + float(shell_alt_km))
    pts = r_shell_km * u                # km
    n_hat = u                           # normal = unit vector on sphere

    # Latitude/longitude from unit vectors
    lat_rad = np.arcsin(np.clip(u[:, 2], -1.0, 1.0))
    lon_rad = np.arctan2(u[:, 1], u[:, 0])

    lat_deg = np.rad2deg(lat_rad)
    lon_deg = np.rad2deg(lon_rad)

    # Normalize lon to [-180, 180]
    lon_deg = (lon_deg + 180.0) % 360.0 - 180.0

    meta = {
        "n_total": int(n),
        "shell_alt_km": float(shell_alt_km),
        "r_shell_km": r_shell_km,
        "point_generation_method": "fibonacci_sphere",
    }
    return pts, n_hat, lat_deg, lon_deg, meta


def latitude_line_points_ecef_with_normals_latlon_km(
    earth: EarthConstants,
    *,
    lat_step_deg: float = 0.5,
    longitude_deg: float = 0.0,
    shell_alt_km: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build a symmetry-aware shell using a single fixed longitude and many latitude samples.

    This preserves the time-by-point matrix structure used throughout the repo while
    collapsing the spatial dimension to a latitude-only representation for optimization.
    """
    step = float(lat_step_deg)
    if not np.isfinite(step) or step <= 0.0:
        raise ValueError("lat_step_deg must be finite and > 0")

    lon_fixed = float(((float(longitude_deg) + 180.0) % 360.0) - 180.0)
    lat_deg = np.arange(-90.0, 90.0 + 0.5 * step, step, dtype=np.float64)
    lat_deg = np.clip(lat_deg, -90.0, 90.0)
    lon_deg = np.full(lat_deg.shape, lon_fixed, dtype=np.float64)

    lat_rad = np.deg2rad(lat_deg)
    lon_rad = np.deg2rad(lon_deg)

    cos_lat = np.cos(lat_rad)
    u = np.stack(
        [
            cos_lat * np.cos(lon_rad),
            cos_lat * np.sin(lon_rad),
            np.sin(lat_rad),
        ],
        axis=1,
    )

    r_shell_km = float(earth.r_eq_km + float(shell_alt_km))
    pts = r_shell_km * u
    n_hat = u
    meta = {
        "n_total": int(lat_deg.size),
        "shell_alt_km": float(shell_alt_km),
        "r_shell_km": r_shell_km,
        "point_generation_method": "symmetric_latitude_line",
        "analysis_shell_mode": "symmetric",
        "analysis_lat_step_deg": step,
        "analysis_longitude_deg": lon_fixed,
    }
    return pts, n_hat, lat_deg, lon_deg, meta
