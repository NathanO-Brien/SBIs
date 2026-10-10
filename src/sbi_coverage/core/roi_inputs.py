"""Combined region-of-interest mask construction from lat/lon boxes and
country names.
"""
import numpy as np

from .regions import mask_country


def _as_finite_float(value: object) -> float | None:
    """Return the value as float when it is a finite number, else None."""
    if isinstance(value, (int, float, np.number)):
        out = float(value)
        if np.isfinite(out):
            return out
    return None


def build_roi_mask_from_inputs(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    latlon_boxes: list[tuple[float, float, float, float]] | None = None,
    countries: list[str] | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Build a combined ROI mask from latitude/longitude boxes and country names.

    Inputs:
      - lat_deg: shell latitudes (shape Np,)
      - lon_deg: shell longitudes (shape Np,)
      - latlon_boxes: list of (lat_min, lat_max, lon_min, lon_max)
      - countries: list of country names supported by `regions.py`

    Returns:
      - roi_mask: boolean mask of shape (Np,), True where inside ROI
      - summary: dict with count, breakdown, and input-driven ROI bounds
        (`summary["roi_input_bounds"]` from authoritative country geometry
        and/or explicit box values, independent of shell discretization)
    """
    lats = np.asarray(lat_deg, dtype=np.float64)
    lons = np.asarray(lon_deg, dtype=np.float64)
    if lats.ndim != 1 or lons.ndim != 1 or lats.shape[0] != lons.shape[0]:
        raise ValueError("lat_deg and lon_deg must be 1D arrays with equal length")

    if latlon_boxes is None:
        latlon_boxes = []
    if countries is None:
        countries = []

    mask = np.zeros(lats.shape[0], dtype=bool)
    summary = {
        "total_points": len(mask),
        "boxes": {},
        "countries": {},
    }
    input_lat_mins: list[float] = []
    input_lat_maxs: list[float] = []
    input_lon_mins: list[float] = []
    input_lon_maxs: list[float] = []

    for (lat_min, lat_max, lon_min, lon_max) in latlon_boxes:
        box_mask = (lats >= lat_min) & (lats <= lat_max) & (lons >= lon_min) & (lons <= lon_max)
        mask |= box_mask
        summary["boxes"][(lat_min, lat_max, lon_min, lon_max)] = int(box_mask.sum())
        input_lat_mins.append(float(min(lat_min, lat_max)))
        input_lat_maxs.append(float(max(lat_min, lat_max)))
        input_lon_mins.append(float(min(lon_min, lon_max)))
        input_lon_maxs.append(float(max(lon_min, lon_max)))

    for country in countries:
        try:
            country_mask, country_meta = mask_country(lats, lons, country)
            mask |= country_mask
            summary["countries"][country] = int(country_mask.sum())
            summary.setdefault("country_meta", {})[country] = country_meta
            c_lat_min = _as_finite_float(country_meta.get("lat_min_deg"))
            c_lat_max = _as_finite_float(country_meta.get("lat_max_deg"))
            c_lon_min = _as_finite_float(country_meta.get("lon_min_deg"))
            c_lon_max = _as_finite_float(country_meta.get("lon_max_deg"))
            if c_lat_min is not None:
                input_lat_mins.append(c_lat_min)
            if c_lat_max is not None:
                input_lat_maxs.append(c_lat_max)
            if c_lon_min is not None:
                input_lon_mins.append(c_lon_min)
            if c_lon_max is not None:
                input_lon_maxs.append(c_lon_max)
        except Exception as e:
            summary["countries"][country] = f"ERROR: {e}"
            summary.setdefault("country_meta", {})[country] = {"error": str(e)}

    summary["roi_input_bounds"] = {
        "lat_min_deg": float(min(input_lat_mins)) if input_lat_mins else None,
        "lat_max_deg": float(max(input_lat_maxs)) if input_lat_maxs else None,
        "lon_min_deg": float(min(input_lon_mins)) if input_lon_mins else None,
        "lon_max_deg": float(max(input_lon_maxs)) if input_lon_maxs else None,
    }
    summary["roi_count"] = int(mask.sum())
    return mask, summary
