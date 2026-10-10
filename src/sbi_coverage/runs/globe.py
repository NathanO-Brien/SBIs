"""Interactive 3-D globe (Plotly) showing where target points are, in the ECI frame at epoch.

At the epoch the ECI and ECEF frames coincide (core/frames.py: ECEF = R_z(-w_E t) ECI with t = 0), so a point at
(lat, lon) sits at x = cos(lat)cos(lon), y = cos(lat)sin(lon), z = sin(lat). Targets are drawn at the intercept
altitude above the surface (true scale), with faint country borders for orientation. Earth radius = 1.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

EARTH_R_KM = 6378.137

_SPHERE_COLOR = "#0f2742"
_BORDER_COLOR = "rgba(190,205,228,0.20)"
_HIGHLIGHT_COLOR = "rgba(255,255,255,0.75)"

_BORDER_CACHE: dict[str, Any] = {}


def to_xyz(lat_deg, lon_deg, radius=1.0):
    lat, lon = np.deg2rad(np.asarray(lat_deg, dtype=float)), np.deg2rad(np.asarray(lon_deg, dtype=float))
    r = np.asarray(radius, dtype=float)
    return r * np.cos(lat) * np.cos(lon), r * np.cos(lat) * np.sin(lon), r * np.sin(lat)


def load_borders() -> dict[str, list[tuple[np.ndarray, np.ndarray]]] | None:
    """{country name: [(lon_deg, lat_deg) ring, ...]} from Natural Earth 110 m (downloaded once per process).

    Returns None when the boundary data cannot be loaded (offline / geopandas missing); callers then draw no borders.
    """
    if "borders" in _BORDER_CACHE:
        return _BORDER_CACHE["borders"]
    out = None
    try:
        from ..core import regions
        gdf = regions._load_naturalearth_lowres()
        out = {}
        for name, geom in zip(gdf["NAME"], gdf.geometry):
            polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else ([geom] if geom.geom_type == "Polygon" else [])
            out[str(name)] = [(np.asarray(p.exterior.xy[0]), np.asarray(p.exterior.xy[1])) for p in polys]
    except Exception:
        out = None
    _BORDER_CACHE["borders"] = out
    return out


def _line_arrays(rings, radius: float):
    xs, ys, zs = [], [], []
    for lon, lat in rings:
        x, y, z = to_xyz(lat, lon, radius)
        xs += [*x, np.nan]; ys += [*y, np.nan]; zs += [*z, np.nan]
    return xs, ys, zs


def build_globe_figure(points: Sequence[Sequence] | None, *, alt_km: float = 200.0, highlight_countries: Sequence[str] = (),
                       borders: dict | None = None, height: int = 380, title: str | None = None):
    """Plotly figure: Earth sphere, faint country borders, ECI axes and the targets as colored points.

    ``points`` is [[name, lat_deg, lon_deg], ...] (None/empty draws just the globe). Points are colored by latitude and
    sit at ``alt_km`` above the surface; the camera is centered on the targets. ``highlight_countries`` draws those
    borders brighter. ``borders`` defaults to load_borders().
    """
    import plotly.graph_objects as go

    traces: list[Any] = []
    # Earth
    u = np.linspace(0, 2 * np.pi, 72)
    v = np.linspace(0, np.pi, 37)
    xs = np.outer(np.cos(u), np.sin(v)); ys = np.outer(np.sin(u), np.sin(v)); zs = np.outer(np.ones_like(u), np.cos(v))
    traces.append(go.Surface(x=xs * 0.998, y=ys * 0.998, z=zs * 0.998, colorscale=[[0, _SPHERE_COLOR], [1, _SPHERE_COLOR]],
                             showscale=False, hoverinfo="skip", lighting=dict(ambient=0.8, diffuse=0.45, specular=0.05, roughness=0.9)))
    # borders
    b = borders if borders is not None else load_borders()
    hl = {h.lower() for h in highlight_countries}
    if b:
        base = [r for n, rings in b.items() if n.lower() not in hl for r in rings]
        x, y, z = _line_arrays(base, 1.001)
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=_BORDER_COLOR, width=0.8), hoverinfo="skip", showlegend=False))
        sel = [r for n, rings in b.items() if n.lower() in hl for r in rings]
        if sel:
            x, y, z = _line_arrays(sel, 1.002)
            traces.append(go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=_HIGHLIGHT_COLOR, width=2.5), hoverinfo="skip", showlegend=False))
    # ECI axes (X toward lon 0 at epoch = vernal equinox direction, Z = north)
    for lab, vec in (("X", (1.22, 0, 0)), ("Y", (0, 1.22, 0)), ("Z", (0, 0, 1.22))):
        traces.append(go.Scatter3d(x=[0, vec[0]], y=[0, vec[1]], z=[0, vec[2]], mode="lines+text", text=["", lab], textposition="top center",
                                   line=dict(color="rgba(150,160,175,0.55)", width=2), textfont=dict(color="rgba(170,180,195,0.9)", size=11),
                                   hoverinfo="skip", showlegend=False))
    eye = (1.45, 0.85, 0.6)
    rev = "empty"
    if points:
        pts = np.asarray([[p[1], p[2]] for p in points], dtype=float)
        lat, lon = pts[:, 0], pts[:, 1]
        r_pt = 1.0 + float(alt_km) / EARTH_R_KM
        x, y, z = to_xyz(lat, lon, r_pt)
        n = len(points)
        rev = f"{n}:{lat.mean():.2f}:{lon.mean():.2f}"   # camera re-centers when the target set changes
        size = 4.5 if n <= 50 else (3.5 if n <= 500 else 2.2)
        names = [str(p[0]) for p in points]
        hover = [f"{nm}<br>lat {la:.2f}°, lon {lo:.2f}°<br>alt {alt_km:.0f} km" for nm, la, lo in zip(names, lat, lon)]
        if n <= 150:   # stems from the surface to the point make the altitude visible
            sx, sy, sz = [], [], []
            x0, y0, z0 = to_xyz(lat, lon, 1.0)
            for a in range(n):
                sx += [x0[a], x[a], np.nan]; sy += [y0[a], y[a], np.nan]; sz += [z0[a], z[a], np.nan]
            traces.append(go.Scatter3d(x=sx, y=sy, z=sz, mode="lines", line=dict(color="rgba(255,255,255,0.45)", width=1), hoverinfo="skip", showlegend=False))
        lat_span = float(lat.max() - lat.min())
        marker = dict(size=size, color=lat if lat_span > 1e-9 else "#ffb020", opacity=0.95, line=dict(color="white", width=0.5))
        if lat_span > 1e-9:
            marker.update(colorscale="Turbo", cmin=float(lat.min()), cmax=float(lat.max()),
                          colorbar=dict(title=dict(text="Lat °", font=dict(size=10)), thickness=8, len=0.5, x=0.98, tickfont=dict(size=9)))
        traces.append(go.Scatter3d(x=x, y=y, z=z, mode="markers", marker=marker, text=hover, hoverinfo="text", showlegend=False))
        # camera over the targets' centroid
        ux, uy, uz = to_xyz(lat, lon, 1.0)
        c = np.array([ux.mean(), uy.mean(), uz.mean()])
        if np.linalg.norm(c) > 0.15:
            c = c / np.linalg.norm(c)
            # zoom in when the targets are clustered: angular radius of the set about its centroid
            rho = float(np.degrees(np.arccos(np.clip(np.min(ux * c[0] + uy * c[1] + uz * c[2]), -1, 1))))
            eye = tuple(float((1.12 + 0.43 * min(1.0, rho / 50.0)) * q) for q in c)
        else:
            eye = (1.45, 0.85, 0.6)
    fig = go.Figure(traces)
    ax = dict(visible=False, showbackground=False, range=[-1.12, 1.12])
    fig.update_layout(
        height=height, margin=dict(l=0, r=0, t=26 if title else 0, b=0), paper_bgcolor="rgba(0,0,0,0)",
        title=dict(text=title, x=0.02, y=0.98, font=dict(size=12)) if title else None,
        scene=dict(xaxis=ax, yaxis=ax, zaxis=ax, aspectmode="cube", bgcolor="rgba(0,0,0,0)",
                   camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]), up=dict(x=0, y=0, z=1))),
        showlegend=False, uirevision=rev,
    )
    return fig
