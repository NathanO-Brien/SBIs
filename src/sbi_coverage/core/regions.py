"""Region-of-interest masking: latitude bands, lat/lon boxes, and country
boundary polygons (Natural Earth via GeoPandas).
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# Optional heavy dependencies — only required for country masking
try:
    import geopandas as gpd
    from shapely.geometry import Point
    from shapely.prepared import prep
    _GEO_OK = True
except Exception:
    gpd = None
    Point = None
    prep = None
    _GEO_OK = False


def _normalize_lon_deg(lon_deg: np.ndarray) -> np.ndarray:
    """Normalize longitude to [-180, 180]."""
    return (lon_deg + 180.0) % 360.0 - 180.0


def mask_lat_band(lat_deg: np.ndarray, lat_min_deg: float, lat_max_deg: float) -> np.ndarray:
    """Inclusive latitude band mask."""
    lo = float(min(lat_min_deg, lat_max_deg))
    hi = float(max(lat_min_deg, lat_max_deg))
    return (lat_deg >= lo) & (lat_deg <= hi)


def mask_latlon_box(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat_min_deg: float,
    lat_max_deg: float,
    lon_min_deg: float,
    lon_max_deg: float,
) -> np.ndarray:
    """
    Inclusive lat/lon bounding box mask with dateline wrap handling.

    lon inputs are interpreted in degrees and normalized to [-180, 180].
    """
    lon = _normalize_lon_deg(np.asarray(lon_deg, dtype=np.float64))
    lat = np.asarray(lat_deg, dtype=np.float64)

    lat_lo = float(min(lat_min_deg, lat_max_deg))
    lat_hi = float(max(lat_min_deg, lat_max_deg))

    lon_lo = float(lon_min_deg)
    lon_hi = float(lon_max_deg)
    lon_lo = float(((lon_lo + 180.0) % 360.0) - 180.0)
    lon_hi = float(((lon_hi + 180.0) % 360.0) - 180.0)

    lat_ok = (lat >= lat_lo) & (lat <= lat_hi)

    # Handle wraparound: e.g., lon_lo=170, lon_hi=-170 means ">=170 OR <=-170"
    if lon_lo <= lon_hi:
        lon_ok = (lon >= lon_lo) & (lon <= lon_hi)
    else:
        lon_ok = (lon >= lon_lo) | (lon <= lon_hi)

    return lat_ok & lon_ok


# ----------------------------
# Country masking via GeoPandas
# ----------------------------

def _load_naturalearth_lowres():
    """
    Load Natural Earth low-res country boundaries in a GeoPandas>=1.0 compatible way.
    Source: NACIS-hosted Natural Earth ZIP (110m admin_0 countries).
    """
    if not _GEO_OK:
        raise ImportError(
            "Country ROI requires geopandas + shapely. Install with: pip install geopandas shapely pyogrio"
        )

    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    return gpd.read_file(url)


_COUNTRY_CACHE = {}

def country_polygon(country: str) -> tuple[object, dict]:
    """
    Return (geometry, meta) for a country lookup from Natural Earth admin_0 countries.

    Uses explicit columns that exist in the NACIS Natural Earth dataset:
      - NAME, ADMIN, NAME_LONG, ISO_A3

    Matching order:
      1) ISO_A3 exact (if 3 letters)
      2) NAME exact (case-insensitive)
      3) ADMIN exact
      4) NAME_LONG exact

    Raises a helpful error if not found.
    """
    key = country.strip()
    if not key:
        raise ValueError("country_polygon(): empty country string")

    cache_key = key.lower()
    if cache_key in _COUNTRY_CACHE:
        geom, meta = _COUNTRY_CACHE[cache_key]
        return geom, dict(meta)

    gdf = _load_naturalearth_lowres()

    required_cols = ["NAME", "ADMIN", "NAME_LONG", "ISO_A3"]
    missing = [c for c in required_cols if c not in gdf.columns]
    if missing:
        raise KeyError(
            f"Natural Earth dataset missing expected columns {missing}. "
            f"Available columns: {list(gdf.columns)}"
        )

    # 1) ISO_A3 exact match (best for unambiguous)
    if len(key) == 3 and key.isalpha():
        iso = key.upper()
        rows = gdf[gdf["ISO_A3"].astype(str).str.upper() == iso]
        if len(rows) > 0:
            row = rows.iloc[0]
            geom = row.geometry
            meta = {
                "country_query": country,
                "country_match": row["NAME"],
                "match_mode": "ISO_A3",
                "ISO_A3": row["ISO_A3"],
            }
            _COUNTRY_CACHE[cache_key] = (geom, meta)
            return geom, dict(meta)

    # normalize the query for name matches
    q = key.lower()

    def _exact_match(col: str):
        s = gdf[col].astype(str).str.strip().str.lower()
        return gdf[s == q]

    # 2) NAME exact
    rows = _exact_match("NAME")
    if len(rows) == 0:
        # 3) ADMIN exact
        rows = _exact_match("ADMIN")
    if len(rows) == 0:
        # 4) NAME_LONG exact
        rows = _exact_match("NAME_LONG")

    if len(rows) == 0:
        # helpful debugging output: show a few close candidates by substring in NAME
        s = gdf["NAME"].astype(str).str.strip().str.lower()
        candidates = gdf[s.str.contains(q, na=False)]["NAME"].head(10).tolist()
        raise ValueError(
            f"Country '{country}' not found by exact match in NAME/ADMIN/NAME_LONG or ISO_A3.\n"
            f"Try ISO3 (e.g., 'PRK') or check spelling.\n"
            f"Substring candidates in NAME: {candidates}"
        )

    row = rows.iloc[0]
    geom = row.geometry
    meta = {
        "country_query": country,
        "country_match": row["NAME"],
        "match_mode": "NAME/ADMIN/NAME_LONG exact",
        "ISO_A3": row["ISO_A3"],
    }

    _COUNTRY_CACHE[cache_key] = (geom, meta)
    return geom, dict(meta)


def country_latlon_bounds(country: str) -> dict[str, float | str]:
    """
    Return country bounding box in degrees using authoritative polygon geometry
    (not shell-grid sample points).
    """
    geom, meta = country_polygon(country)
    minx, miny, maxx, maxy = geom.bounds  # lon, lat
    return {
        **meta,
        "lon_min_deg": float(minx),
        "lon_max_deg": float(maxx),
        "lat_min_deg": float(miny),
        "lat_max_deg": float(maxy),
    }


def _normalize_country_inputs(country: str | Sequence[str]) -> list[str]:
    """Coerce a country name or sequence of names into a clean list."""
    if isinstance(country, str):
        out = [country]
    else:
        out = [str(c) for c in country]
    out = [c.strip() for c in out if str(c).strip()]
    if not out:
        raise ValueError("mask_country(): at least one country must be provided")
    return out


def _mask_single_country(
    lat: np.ndarray,
    lon: np.ndarray,
    country: str,
    *,
    bbox_prefilter: bool = True,
) -> tuple[np.ndarray, dict]:
    """Point-in-polygon mask for a single country. Returns (mask, roi_meta)."""
    geom, meta = country_polygon(country)

    # Bounding box prefilter to reduce point-in-polygon checks
    if bbox_prefilter:
        minx, miny, maxx, maxy = geom.bounds  # lon, lat
        pre = (lon >= minx) & (lon <= maxx) & (lat >= miny) & (lat <= maxy)
    else:
        pre = np.ones(lat.shape[0], dtype=bool)

    prepared = prep(geom)
    mask = np.zeros(lat.shape[0], dtype=bool)
    idx = np.where(pre)[0]

    for k in idx:
        if prepared.contains(Point(float(lon[k]), float(lat[k]))):
            mask[k] = True

    bounds = country_latlon_bounds(country)
    roi_meta = {
        "roi_mode": "country",
        **meta,
        "lat_min_deg": float(bounds["lat_min_deg"]),
        "lat_max_deg": float(bounds["lat_max_deg"]),
        "lon_min_deg": float(bounds["lon_min_deg"]),
        "lon_max_deg": float(bounds["lon_max_deg"]),
        "n_total": int(lat.shape[0]),
        "n_roi": int(mask.sum()),
        "bbox_prefilter": bool(bbox_prefilter),
    }
    return mask, roi_meta


def mask_country(
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    country: str | Sequence[str],
    *,
    bbox_prefilter: bool = True,
) -> tuple[np.ndarray, dict]:
    """
    Build a boolean mask selecting points inside one or more country boundary polygons.

    Returns (mask, roi_meta).

    Notes:
    - Uses Shapely point-in-polygon checks.
    - Applies a bounding-box prefilter by default for speed.
    - Assumes lat/lon in degrees, lon in [-180, 180].
    """
    lat = np.asarray(lat_deg, dtype=np.float64)
    lon = _normalize_lon_deg(np.asarray(lon_deg, dtype=np.float64))
    countries = _normalize_country_inputs(country)

    if len(countries) == 1:
        return _mask_single_country(lat, lon, countries[0], bbox_prefilter=bbox_prefilter)

    mask = np.zeros(lat.shape[0], dtype=bool)
    components: list[dict] = []

    for c in countries:
        submask, submeta = _mask_single_country(lat, lon, c, bbox_prefilter=bbox_prefilter)
        mask |= submask
        components.append(submeta)

    lat_min = min(float(m["lat_min_deg"]) for m in components)
    lat_max = max(float(m["lat_max_deg"]) for m in components)
    lon_min = min(float(m["lon_min_deg"]) for m in components)
    lon_max = max(float(m["lon_max_deg"]) for m in components)

    roi_meta = {
        "roi_mode": "multi_country",
        "countries": list(countries),
        "country_count": int(len(countries)),
        "lat_min_deg": lat_min,
        "lat_max_deg": lat_max,
        "lon_min_deg": lon_min,
        "lon_max_deg": lon_max,
        "n_total": int(lat.shape[0]),
        "n_roi": int(mask.sum()),
        "bbox_prefilter": bool(bbox_prefilter),
        "country_components": components,
    }
    return mask, roi_meta
