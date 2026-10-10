"""Target "shell": the set of points (on a sphere at the intercept altitude) that must be covered."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .config import EarthConstants
from .earth_grid import points_ecef_with_normals_latlon_km
from . import regions


@dataclass(frozen=True)
class Shell:
    """A spherical sampling grid in ECEF at ``shell_alt_km`` above the surface.

    Bundles the simulation point set (points + outward normals + lat/lon + metadata) into one object.
    Two ways to build one:
      - ``Shell(earth, n_points=N, shell_alt_km=h)``: quasi-uniform Fibonacci sampling of the whole sphere
        (``analysis_shell_mode == "full"``), typically masked afterwards to a country or box;
      - ``Shell.from_points(earth, lat_deg=..., lon_deg=..., shell_alt_km=h)``: exact caller-supplied
        target points (``analysis_shell_mode == "predefined"``) -- what the optimizer uses.

    Typical usage:
        shell = Shell(earth, n_points=5000, shell_alt_km=200.0).mask_country("United States")
        out = run_simulation(..., shell=shell)
    """

    earth: EarthConstants
    n_points: int | None = None            # requested point count ("full" mode)
    shell_alt_km: float = 0.0
    analysis_shell_mode: str = "full"      # "full" | "predefined" (set by from_points)

    # Generated fields
    points_ecef_km: np.ndarray = field(init=False, repr=False)
    normals_ecef: np.ndarray = field(init=False, repr=False)
    lat_deg: np.ndarray = field(init=False, repr=False)
    lon_deg: np.ndarray = field(init=False, repr=False)
    meta: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self):
        if str(self.analysis_shell_mode).strip().lower() != "full":
            raise ValueError("Shell(): only analysis_shell_mode='full' can be constructed directly; "
                             "use Shell.from_points() for exact target points")
        if self.n_points is None:
            raise ValueError("Shell(full): n_points is required")
        n_points = int(self.n_points)
        pts, n_hat, lat, lon, meta = points_ecef_with_normals_latlon_km(
            n_points, self.earth, shell_alt_km=float(self.shell_alt_km))
        m = dict(meta) if isinstance(meta, dict) else {"meta": meta}
        m.update({"n_requested": n_points, "shell_alt_km": float(self.shell_alt_km), "analysis_shell_mode": "full"})
        self._set_arrays(pts, n_hat, lat, lon, m, n_points=n_points, mode="full")

    def _set_arrays(self, pts, n_hat, lat, lon, meta, *, n_points: int | None, mode: str) -> None:
        """Assign the generated fields of this frozen dataclass and validate them."""
        object.__setattr__(self, "n_points", n_points)
        object.__setattr__(self, "analysis_shell_mode", mode)
        object.__setattr__(self, "points_ecef_km", np.asarray(pts, dtype=np.float64))
        object.__setattr__(self, "normals_ecef", np.asarray(n_hat, dtype=np.float64))
        object.__setattr__(self, "lat_deg", np.asarray(lat, dtype=np.float64))
        object.__setattr__(self, "lon_deg", np.asarray(lon, dtype=np.float64))
        object.__setattr__(self, "meta", meta)
        self._validate()

    @classmethod
    def _bypass(cls, earth: EarthConstants, shell_alt_km: float) -> "Shell":
        """A Shell whose arrays are filled in by the caller via _set_arrays (skips point generation)."""
        new = object.__new__(cls)
        object.__setattr__(new, "earth", earth)
        object.__setattr__(new, "shell_alt_km", float(shell_alt_km))
        return new

    @classmethod
    def from_points(
        cls,
        earth: EarthConstants,
        *,
        lat_deg: "np.ndarray | list[float]",
        lon_deg: "np.ndarray | list[float]",
        shell_alt_km: float = 0.0,
        meta_extra: "Optional[dict[str, Any]]" = None,
    ) -> "Shell":
        """Build a Shell from a predefined set of lat/lon target points.

        No point generation occurs -- the caller supplies the exact points of interest (named
        geographic targets for the MILP). Each point is placed at earth.r_eq_km + shell_alt_km with an
        outward radial unit normal.
        """
        lat = np.asarray(lat_deg, dtype=np.float64).ravel()
        lon = np.asarray(lon_deg, dtype=np.float64).ravel()
        if lat.shape != lon.shape:
            raise ValueError("Shell.from_points(): lat_deg and lon_deg must have the same length")
        if lat.size == 0:
            raise ValueError("Shell.from_points(): at least one point is required")

        lat_rad, lon_rad = np.deg2rad(lat), np.deg2rad(lon)
        cos_lat = np.cos(lat_rad)
        u = np.stack([cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)], axis=1)
        r_shell_km = float(earth.r_eq_km) + float(shell_alt_km)

        m: dict[str, Any] = {
            "n_requested": int(lat.size), "n_used": int(lat.size), "shell_alt_km": float(shell_alt_km),
            "r_shell_km": r_shell_km, "analysis_shell_mode": "predefined", "point_generation_method": "from_points",
        }
        if meta_extra:
            m.update(meta_extra)
        new = cls._bypass(earth, shell_alt_km)
        new._set_arrays(r_shell_km * u, u, lat, lon, m, n_points=int(lat.size), mode="predefined")
        return new

    # Convenience aliases
    @property
    def points_ecef(self) -> np.ndarray:
        """Alias for points_ecef_km."""
        return self.points_ecef_km

    @property
    def n_points_used(self) -> int:
        """Actual number of points in this shell (after any masking)."""
        return int(self.points_ecef_km.shape[0])

    def _validate(self) -> None:
        """Check array shapes are internally consistent."""
        pts = self.points_ecef_km
        n_hat = self.normals_ecef
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("Shell: points_ecef_km must have shape (N,3)")
        if n_hat.ndim != 2 or n_hat.shape != pts.shape:
            raise ValueError("Shell: normals_ecef must have same shape as points_ecef_km")
        if self.lat_deg.shape[0] != pts.shape[0] or self.lon_deg.shape[0] != pts.shape[0]:
            raise ValueError("Shell: lat/lon arrays must have length N")

    def mask(self, mask: np.ndarray, *, meta_update: Optional[dict[str, Any]] = None) -> "Shell":
        """Return a new Shell containing only points where mask is True."""
        mask = np.asarray(mask, dtype=bool)
        if mask.shape[0] != self.n_points_used:
            raise ValueError("Shell.mask(): mask length must match number of points")
        m = dict(self.meta)
        m.update({"n_used": int(np.count_nonzero(mask))})
        if meta_update:
            m.update(dict(meta_update))
        new = self._bypass(self.earth, self.shell_alt_km)
        new._set_arrays(self.points_ecef_km[mask].copy(), self.normals_ecef[mask].copy(), self.lat_deg[mask].copy(),
                        self.lon_deg[mask].copy(), m, n_points=None if self.n_points is None else int(self.n_points),
                        mode=str(self.analysis_shell_mode))
        return new

    def mask_latlon_box(self, *, lat_min_deg: float, lat_max_deg: float, lon_min_deg: float, lon_max_deg: float) -> "Shell":
        """Return a new Shell restricted to a lat/lon bounding box."""
        mask = regions.mask_latlon_box(self.lat_deg, self.lon_deg, lat_min_deg=lat_min_deg, lat_max_deg=lat_max_deg,
                                       lon_min_deg=lon_min_deg, lon_max_deg=lon_max_deg)
        return self.mask(mask, meta_update={
            "roi_mode": "latlon_box", "lat_min_deg": float(lat_min_deg), "lat_max_deg": float(lat_max_deg),
            "lon_min_deg": float(lon_min_deg), "lon_max_deg": float(lon_max_deg),
        })

    def mask_country(self, country: str | Sequence[str], *countries: str, bbox_prefilter: bool = True) -> "Shell":
        """Return a new Shell restricted to points inside one or more countries.

        Accepts a single name, a sequence of names, or multiple positional names. ``bbox_prefilter``
        speeds up point-in-polygon testing.
        """
        if countries:
            country_arg: str | Sequence[str] = [country, *countries] if isinstance(country, str) else [*country, *countries]
        else:
            country_arg = country
        mask, roi_meta = regions.mask_country(self.lat_deg, self.lon_deg, country=country_arg,
                                              bbox_prefilter=bbox_prefilter)
        return self.mask(mask, meta_update=roi_meta)
