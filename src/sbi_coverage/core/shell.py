from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from .config import EarthConstants
from .earth_grid import latitude_line_points_ecef_with_normals_latlon_km, points_ecef_with_normals_latlon_km
from . import regions


@dataclass(frozen=True)
class Shell:
    """A spherical shell sampling grid in ECEF.

    The intent is to treat the simulation point set (points + normals + lat/lon/meta)
    as a first-class object rather than passing raw arrays through the API.

    Typical usage:
        shell = Shell(earth, n_points=5000, shell_alt_km=0.0, analysis_shell_mode="full")
        shell = Shell(
            earth,
            shell_alt_km=200.0,
            analysis_shell_mode="symmetric",
            analysis_lat_step_deg=0.5,
            analysis_longitude_deg=0.0,
        )
        shell_roi = shell.mask_country("United States")
        out = run_simulation(..., shell=shell_roi)
    """

    earth: EarthConstants
    n_points: int | None = None
    shell_alt_km: float = 0.0
    analysis_shell_mode: str = "full"
    analysis_lat_step_deg: float | None = None
    analysis_longitude_deg: float | None = None

    # Generated fields
    points_ecef_km: np.ndarray = field(init=False, repr=False)
    normals_ecef: np.ndarray = field(init=False, repr=False)
    lat_deg: np.ndarray = field(init=False, repr=False)
    lon_deg: np.ndarray = field(init=False, repr=False)
    meta: dict[str, Any] = field(init=False, repr=False)

    def __post_init__(self):
        mode = str(self.analysis_shell_mode).strip().lower()
        if mode == "full":
            if self.n_points is None:
                raise ValueError("Shell(full): n_points is required")
            n_points_used = int(self.n_points)
            pts, n_hat, lat, lon, meta = points_ecef_with_normals_latlon_km(
                n_points_used,
                self.earth,
                shell_alt_km=float(self.shell_alt_km),
            )
            m = dict(meta) if isinstance(meta, dict) else {"meta": meta}
            m.update({
                "n_requested": n_points_used,
                "shell_alt_km": float(self.shell_alt_km),
                "analysis_shell_mode": "full",
            })
            object.__setattr__(self, "n_points", n_points_used)
            object.__setattr__(self, "analysis_shell_mode", "full")
            object.__setattr__(self, "analysis_lat_step_deg", None)
            object.__setattr__(self, "analysis_longitude_deg", None)
        elif mode == "symmetric":
            lat_step_deg = 0.5 if self.analysis_lat_step_deg is None else float(self.analysis_lat_step_deg)
            longitude_deg = 0.0 if self.analysis_longitude_deg is None else float(self.analysis_longitude_deg)
            pts, n_hat, lat, lon, meta = latitude_line_points_ecef_with_normals_latlon_km(
                self.earth,
                lat_step_deg=lat_step_deg,
                longitude_deg=longitude_deg,
                shell_alt_km=float(self.shell_alt_km),
            )
            m = dict(meta) if isinstance(meta, dict) else {"meta": meta}
            m.update({
                "n_requested": int(lat.size),
                "shell_alt_km": float(self.shell_alt_km),
                "analysis_shell_mode": "symmetric",
            })
            object.__setattr__(self, "n_points", int(lat.size))
            object.__setattr__(self, "analysis_shell_mode", "symmetric")
            object.__setattr__(self, "analysis_lat_step_deg", lat_step_deg)
            object.__setattr__(self, "analysis_longitude_deg", longitude_deg)
        else:
            raise ValueError("Shell: analysis_shell_mode must be 'full' or 'symmetric'")

        # Freeze dataclass, but still assign in __post_init__
        object.__setattr__(self, "points_ecef_km", np.asarray(pts, dtype=np.float64))
        object.__setattr__(self, "normals_ecef", np.asarray(n_hat, dtype=np.float64))
        object.__setattr__(self, "lat_deg", np.asarray(lat, dtype=np.float64))
        object.__setattr__(self, "lon_deg", np.asarray(lon, dtype=np.float64))
        object.__setattr__(self, "meta", m)

        self._validate()

    @classmethod
    def symmetric_latitude_shell(
        cls,
        earth: EarthConstants,
        *,
        lat_step_deg: float = 0.5,
        longitude_deg: float = 0.0,
        shell_alt_km: float = 0.0,
    ) -> "Shell":
        return cls(
            earth=earth,
            shell_alt_km=float(shell_alt_km),
            analysis_shell_mode="symmetric",
            analysis_lat_step_deg=float(lat_step_deg),
            analysis_longitude_deg=float(longitude_deg),
        )

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

        Unlike the 'full' and 'symmetric' modes, no point generation occurs —
        the caller supplies the exact points of interest. Useful for MILP target
        sets where each point is a named geographic target rather than a grid sample.

        Each point is placed on the spherical shell at earth.r_eq_km + shell_alt_km.
        Normals are outward radial unit vectors (same convention as 'full' mode).
        """
        lat = np.asarray(lat_deg, dtype=np.float64).ravel()
        lon = np.asarray(lon_deg, dtype=np.float64).ravel()
        if lat.shape != lon.shape:
            raise ValueError("Shell.from_points(): lat_deg and lon_deg must have the same length")
        if lat.size == 0:
            raise ValueError("Shell.from_points(): at least one point is required")

        lat_rad = np.deg2rad(lat)
        lon_rad = np.deg2rad(lon)
        cos_lat = np.cos(lat_rad)
        u = np.stack(
            [cos_lat * np.cos(lon_rad), cos_lat * np.sin(lon_rad), np.sin(lat_rad)],
            axis=1,
        )
        r_shell_km = float(earth.r_eq_km) + float(shell_alt_km)
        pts   = r_shell_km * u
        n_hat = u

        m: dict[str, Any] = {
            "n_requested": int(lat.size),
            "n_used": int(lat.size),
            "shell_alt_km": float(shell_alt_km),
            "r_shell_km": r_shell_km,
            "analysis_shell_mode": "predefined",
            "point_generation_method": "from_points",
        }
        if meta_extra:
            m.update(meta_extra)

        new = object.__new__(cls)
        object.__setattr__(new, "earth",                  earth)
        object.__setattr__(new, "n_points",               int(lat.size))
        object.__setattr__(new, "shell_alt_km",           float(shell_alt_km))
        object.__setattr__(new, "analysis_shell_mode",    "predefined")
        object.__setattr__(new, "analysis_lat_step_deg",  None)
        object.__setattr__(new, "analysis_longitude_deg", None)
        object.__setattr__(new, "points_ecef_km",         pts)
        object.__setattr__(new, "normals_ecef",           n_hat)
        object.__setattr__(new, "lat_deg",                lat)
        object.__setattr__(new, "lon_deg",                lon)
        object.__setattr__(new, "meta",                   m)
        new._validate()
        return new

    # Convenience aliases to match the language in your scripts
    @property
    def points_ecef(self) -> np.ndarray:
        return self.points_ecef_km

    @property
    def n_hat_ecef(self) -> np.ndarray:
        return self.normals_ecef

    @property
    def n_points_used(self) -> int:
        return int(self.points_ecef_km.shape[0])

    def _validate(self) -> None:
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

        # Construct a new shell without regenerating by bypassing __post_init__
        new = object.__new__(Shell)
        object.__setattr__(new, "earth", self.earth)
        object.__setattr__(new, "n_points", None if self.n_points is None else int(self.n_points))
        object.__setattr__(new, "shell_alt_km", float(self.shell_alt_km))
        object.__setattr__(new, "analysis_shell_mode", str(self.analysis_shell_mode))
        object.__setattr__(new, "analysis_lat_step_deg", self.analysis_lat_step_deg)
        object.__setattr__(new, "analysis_longitude_deg", self.analysis_longitude_deg)
        object.__setattr__(new, "points_ecef_km", self.points_ecef_km[mask].copy())
        object.__setattr__(new, "normals_ecef", self.normals_ecef[mask].copy())
        object.__setattr__(new, "lat_deg", self.lat_deg[mask].copy())
        object.__setattr__(new, "lon_deg", self.lon_deg[mask].copy())

        m = dict(self.meta)
        m.update({"n_used": int(np.count_nonzero(mask))})
        if meta_update:
            m.update(dict(meta_update))
        object.__setattr__(new, "meta", m)
        new._validate()
        return new

    def mask_latlon_box(
        self,
        *,
        lat_min_deg: float,
        lat_max_deg: float,
        lon_min_deg: float,
        lon_max_deg: float,
    ) -> "Shell":
        if str(self.analysis_shell_mode).strip().lower() == "symmetric":
            mask = regions.mask_lat_band(
                self.lat_deg,
                lat_min_deg=float(lat_min_deg),
                lat_max_deg=float(lat_max_deg),
            )
            return self.mask(mask, meta_update={
                "roi_mode": "lat_band_from_box",
                "lat_min_deg": float(min(lat_min_deg, lat_max_deg)),
                "lat_max_deg": float(max(lat_min_deg, lat_max_deg)),
                "source_lon_min_deg": float(lon_min_deg),
                "source_lon_max_deg": float(lon_max_deg),
            })

        mask = regions.mask_latlon_box(
            self.lat_deg,
            self.lon_deg,
            lat_min_deg=lat_min_deg,
            lat_max_deg=lat_max_deg,
            lon_min_deg=lon_min_deg,
            lon_max_deg=lon_max_deg,
        )
        return self.mask(mask, meta_update={
            "roi_mode": "latlon_box",
            "lat_min_deg": float(lat_min_deg),
            "lat_max_deg": float(lat_max_deg),
            "lon_min_deg": float(lon_min_deg),
            "lon_max_deg": float(lon_max_deg),
        })

    def mask_country(
        self,
        country: str | Sequence[str],
        *countries: str,
        bbox_prefilter: bool = True,
    ) -> "Shell":
        if countries:
            if isinstance(country, str):
                country_arg: str | Sequence[str] = [country, *countries]
            else:
                country_arg = [*country, *countries]
        else:
            country_arg = country
        if str(self.analysis_shell_mode).strip().lower() == "symmetric":
            if isinstance(country_arg, str):
                country_list = [country_arg.strip()] if country_arg.strip() else []
            else:
                country_list = [str(c).strip() for c in country_arg if str(c).strip()]
            if not country_list:
                raise ValueError("Shell.mask_country(): at least one country must be provided")
            bounds = [regions.country_latlon_bounds(c) for c in country_list]
            lat_min = min(float(b["lat_min_deg"]) for b in bounds)
            lat_max = max(float(b["lat_max_deg"]) for b in bounds)
            mask = regions.mask_lat_band(self.lat_deg, lat_min, lat_max)
            if len(country_list) == 1:
                roi_meta = {
                    "roi_mode": "country_lat_band",
                    **bounds[0],
                    "n_total": int(self.lat_deg.shape[0]),
                    "n_roi": int(mask.sum()),
                }
            else:
                roi_meta = {
                    "roi_mode": "multi_country_lat_band",
                    "countries": list(country_list),
                    "country_count": int(len(country_list)),
                    "lat_min_deg": lat_min,
                    "lat_max_deg": lat_max,
                    "n_total": int(self.lat_deg.shape[0]),
                    "n_roi": int(mask.sum()),
                    "country_components": bounds,
                }
            return self.mask(mask, meta_update=roi_meta)
        mask, roi_meta = regions.mask_country(
            self.lat_deg,
            self.lon_deg,
            country=country_arg,
            bbox_prefilter=bbox_prefilter,
        )
        return self.mask(mask, meta_update=roi_meta)
