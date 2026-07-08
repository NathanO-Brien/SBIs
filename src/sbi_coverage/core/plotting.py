from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from .analysis_matrix import Analysis

try:
    import imageio
except Exception:  # pragma: no cover
    imageio = None

try:
    from . import regions as _regions
except Exception:  # pragma: no cover
    _regions = None


def plot_initial_orbits(
    elems0: Any,
    earth: Any,
    *,
    layer_specs: list[dict[str, Any]] | None = None,
    n_samples: int = 96,
    t0_s: float = 0.0,
    frame: str = "ECI",
    show_earth: bool = True,
    earth_texture_path: str | None = None,
    show_points: bool = True,
    show_orbits: bool = True,
    max_orbits: int | None = None,
    point_size: float = 10.0,
    line_alpha: float = 0.65,
    show_legend: bool = True,
    show_frame_axes: bool = True,
    frame_axes_scale: float = 0.35,
    view_elev_deg: float = 24.0,
    view_azim_deg: float = 36.0,
    title: str | None = None,
    save_path: str | None = None,
):
    """Plot epoch satellite states and full orbits, color-coded by layer."""
    del earth_texture_path
    a_km, e, inc, raan, argp, M0 = _extract_elements(elems0)
    ns = int(a_km.size)
    if ns == 0:
        raise ValueError("plot_initial_orbits(): no satellites found in elems0")

    ns_plot = min(ns, int(max_orbits)) if max_orbits is not None else ns
    idx = np.arange(ns_plot, dtype=np.int64)

    re_km = float(_get_attr_any(earth, ["r_eq_km", "Re_km", "Re", "R_earth_km"], default=6378.137))
    omega_earth = float(_get_attr_any(earth, ["omega_earth_rad_s", "omega_earth"], default=7.292115e-5))
    frame_u = str(frame).upper()
    if frame_u not in {"ECI", "ECEF"}:
        raise ValueError("plot_initial_orbits(): frame must be 'ECI' or 'ECEF'")

    layer_id = _build_layer_id(layer_specs=layer_specs, ns=ns)[:ns_plot]
    layer_labels = _build_layer_labels(layer_specs=layer_specs)
    layer_colors = _layer_color_map(layer_id)

    fig = plt.figure(figsize=(10.5, 9.0))
    ax = fig.add_subplot(111, projection="3d")
    if hasattr(ax, "computed_zorder"):
        ax.computed_zorder = False
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_zlabel("z (km)")
    ax.grid(False)
    ax.view_init(elev=float(view_elev_deg), azim=float(view_azim_deg))
    # Hide default pane fills so they do not look like clipping artifacts.
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    if show_earth:
        _plot_earth(ax, re_km=re_km)

    nu = np.linspace(0.0, 2.0 * np.pi, int(n_samples), endpoint=True)
    epoch_xyz = np.zeros((ns_plot, 3), dtype=np.float64)
    r_max_km = float(np.max(a_km[:ns_plot] * (1.0 + e[:ns_plot])))

    for k in idx:
        c = layer_colors[int(layer_id[k])]
        if show_orbits:
            xyz = _orbit_r_eci_from_elements(
                a_km=float(a_km[k]),
                e=float(e[k]),
                inc=float(inc[k]),
                raan=float(raan[k]),
                argp=float(argp[k]),
                nu=nu,
            )
            if frame_u == "ECEF":
                xyz = _eci_to_ecef_simple(xyz, t_s=float(t0_s), omega_earth_rad_s=omega_earth)
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=c, linewidth=0.9, alpha=float(line_alpha), zorder=10)

        nu0 = _true_anomaly_from_mean_anomaly(float(M0[k]), float(e[k]))
        p0 = _orbit_r_eci_from_elements(
            a_km=float(a_km[k]),
            e=float(e[k]),
            inc=float(inc[k]),
            raan=float(raan[k]),
            argp=float(argp[k]),
            nu=np.asarray([nu0], dtype=np.float64),
        )[0]
        if frame_u == "ECEF":
            p0 = _eci_to_ecef_simple(p0[None, :], t_s=float(t0_s), omega_earth_rad_s=omega_earth)[0]
        epoch_xyz[k, :] = p0

    if show_points:
        point_colors = [layer_colors[int(x)] for x in layer_id]
        ax.scatter(epoch_xyz[:, 0], epoch_xyz[:, 1], epoch_xyz[:, 2], s=float(point_size), c=point_colors, depthshade=True)

    _set_equal_3d(ax, re_km=re_km, points=epoch_xyz, r_max_km=r_max_km)
    axis_handles: list[Line2D] = []
    if show_frame_axes:
        axis_handles = _draw_frame_axes(ax, frame_name=frame_u, scale_km=float(frame_axes_scale) * re_km)

    if title is None:
        title = f"Initial Satellite States + Orbits ({frame_u})  Ns={ns_plot}/{ns}"
    ax.set_title(title)

    if show_legend:
        legend_handles = []
        for lid in sorted(set(int(x) for x in layer_id.tolist())):
            label = layer_labels.get(lid, f"Layer {lid + 1}")
            legend_handles.append(Patch(facecolor=layer_colors[lid], edgecolor="none", label=label))
        legend_handles.extend(axis_handles)
        if legend_handles:
            ax.legend(handles=legend_handles, loc="upper left", frameon=True)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig, ax


def render_coverage_video_mercator(
    ao: Analysis,
    *,
    elems0: Any | None = None,
    phys: Any | None = None,
    sim: Any | None = None,
    earth: Any | None = None,
    propagator: str = "Nominal_Propagator",
    stats_countries: list[str] | None = None,
    stats_ao: Analysis | None = None,
    fps: int = 60,
    output_path: str | os.PathLike[str] | None = None,
    highlight_roi_borders: bool = True,
    figsize: tuple[float, float] = (14.0, 7.0),
    dpi: int = 140,
    point_size: float = 6.0,
    cmap: str = "Blues",
    title_prefix: str = "Coverage Over Time",
    verbose: bool = True,
    progress_every_percent: int = 10,
    r_required: int = 1,
) -> dict[str, Any]:
    """Render an MP4 of shell-point coverage over time on an oval world projection.

    Notes
    -----
    - Each frame corresponds to one timestep in `ao.counts`.
    - Point color darkens with higher instantaneous coverage.
    - If `output_path` is None, a temporary `.mp4` is created.
    - MP4 encoding requires either `imageio-ffmpeg` or a working imageio ffmpeg plugin.
    - If `ao` was produced with an ROI-masked shell, only ROI shell points exist in the
      analysis object, so non-ROI points cannot be drawn from that input alone.
    """
    if not isinstance(ao, Analysis):
        raise TypeError("render_coverage_video_mercator(): ao must be an Analysis")
    if imageio is None:
        raise RuntimeError(
            "render_coverage_video_mercator(): imageio is not available. "
            "Install `imageio` and `imageio-ffmpeg` for MP4 export."
        )
    if int(fps) <= 0:
        raise ValueError("render_coverage_video_mercator(): fps must be >= 1")
    if int(progress_every_percent) <= 0 or int(progress_every_percent) > 100:
        raise ValueError("render_coverage_video_mercator(): progress_every_percent must be in [1, 100]")

    counts = np.asarray(ao.counts)
    if counts.ndim != 2:
        raise ValueError("render_coverage_video_mercator(): ao.counts must be 2D")

    Nt, Np = counts.shape
    lat_deg = np.asarray(ao.lat_deg, dtype=np.float64)
    lon_deg = np.asarray(ao.lon_deg, dtype=np.float64)
    if lat_deg.shape != (Np,) or lon_deg.shape != (Np,):
        raise ValueError("render_coverage_video_mercator(): lat/lon lengths must match counts.shape[1]")

    tmp_dir_obj = None
    if output_path is None:
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="sbi_coverage_video_")
        out_path = Path(tmp_dir_obj.name) / "coverage_animation.mp4"
        persistent = False
    else:
        out_path = Path(output_path).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        persistent = True

    x_proj, y_proj = _mollweide_project(lat_deg=lat_deg, lon_deg=lon_deg)
    vmin = int(np.min(counts)) if counts.size else 0
    vmax = int(np.max(counts)) if counts.size else 0
    if vmax <= 24 and vmin <= vmax:
        boundaries = np.arange(float(vmin) - 0.5, float(vmax) + 1.5, 1.0)
    else:
        boundaries = np.linspace(float(vmin), max(float(vmin) + 1.0, float(vmax)), 13)
    n_intervals = len(boundaries) - 1
    # Build a per-interval colormap so the 0-coverage bucket is neon pink and
    # the colorbar reflects this exactly.
    _base_colors = plt.get_cmap(cmap)(np.linspace(0.15, 1.0, n_intervals))
    if vmin == 0:
        _base_colors[0] = mcolors.to_rgba("#FF1493")
    cmap_obj = mcolors.ListedColormap(_base_colors)
    norm = mcolors.BoundaryNorm(boundaries=boundaries, ncolors=n_intervals, clip=True)
    if stats_countries:
        border_geoms = _extract_country_boundaries(stats_countries) if highlight_roi_borders else []
    else:
        border_geoms = _extract_roi_country_boundaries(getattr(ao, "meta", {})) if highlight_roi_borders else []
    # When a separate stats_ao is provided (e.g. predefined optimizer targets),
    # build country masks and per-frame stats from that shell so the video overlay
    # matches the same points reported by report(). Visual rendering still uses ao.
    _stats_counts = np.asarray(stats_ao.counts) if stats_ao is not None else counts
    _stats_lat    = np.asarray(stats_ao.lat_deg, dtype=np.float64) if stats_ao is not None else lat_deg
    _stats_lon    = np.asarray(stats_ao.lon_deg, dtype=np.float64) if stats_ao is not None else lon_deg
    country_stats = _prepare_country_stats(
        counts=_stats_counts,
        lat_deg=_stats_lat,
        lon_deg=_stats_lon,
        countries=stats_countries or [],
    )
    world_geoms = _extract_world_boundaries()

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(111, projection="mollweide")
    fig.subplots_adjust(left=0.06, right=0.92, top=0.92, bottom=0.10)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title_prefix)
    ax.grid(True, linewidth=0.75, color="#d3d3d3", alpha=0.85)

    _draw_mollweide_graticule(ax)
    _draw_mollweide_longitude_labels(ax)
    _plot_world_boundaries(ax, world_geoms)
    _plot_roi_boundaries(ax, border_geoms)

    scat = ax.scatter(
        x_proj,
        y_proj,
        c=counts[0].astype(np.float32, copy=False) if Nt > 0 else np.zeros(Np, dtype=np.float32),
        s=float(point_size),
        cmap=cmap_obj,
        norm=norm,
        linewidths=0.0,
        zorder=1.5,
    )
    cbar = fig.colorbar(scat, ax=ax, pad=0.02)
    _cbar_label = "SBIs In Range"
    if r_required > 1:
        _cbar_label += f"  (need ≥{r_required})"
    cbar.set_label(_cbar_label)
    _tick_step = max(1, int(np.ceil((vmax - vmin) / 20)))
    cbar.set_ticks(np.arange(vmin, vmax + 1, _tick_step))
    cbar.ax.yaxis.set_minor_locator(plt.NullLocator())

    time_text = ax.text(
        0.01,
        1.035,
        "",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )

    writer = None
    try:
        writer = imageio.get_writer(
            out_path,
            fps=int(fps),
            codec="libx264",
            format="FFMPEG",
            macro_block_size=None,
        )
    except Exception as exc:
        plt.close(fig)
        if tmp_dir_obj is not None:
            tmp_dir_obj.cleanup()
        raise RuntimeError(
            "render_coverage_video_mercator(): unable to create MP4 writer. "
            "Install `imageio-ffmpeg` or system `ffmpeg` to enable MP4 export."
        ) from exc

    try:
        progress_marks = sorted(
            set(
                max(1, int(np.ceil((p / 100.0) * float(Nt))))
                for p in range(int(progress_every_percent), 101, int(progress_every_percent))
            )
        )
        next_mark_idx = 0
        for k in range(Nt):
            frame_counts = counts[k].astype(np.float32, copy=False)
            scat.set_array(frame_counts)
            time_s = float(ao.times_s[k]) if k < len(ao.times_s) else float(k)
            overlay_lines = [f"t = {_format_hhmmss(time_s)}"]
            if country_stats:
                stats_frame = _stats_counts[k].astype(np.float32, copy=False) if k < len(_stats_counts) else frame_counts
                for item in country_stats:
                    _mask = item["mask"]
                    current_med = float(np.median(stats_frame[_mask]))
                    overlay_lines.append(
                        f"{item['label']}: min={item['min_count']}  median={current_med:.1f}"
                    )
            else:
                overlay_lines.append(
                    f"min={int(np.min(frame_counts))}  "
                    f"max={int(np.max(frame_counts))}  "
                    f"mean={float(np.mean(frame_counts)):.2f}"
                )
            time_text.set_text("\n".join(overlay_lines))
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(frame[:, :, :3])
            if verbose and next_mark_idx < len(progress_marks) and (k + 1) >= progress_marks[next_mark_idx]:
                pct = int(round(100.0 * float(k + 1) / float(max(Nt, 1))))
                print(
                    f"[video] rendered {pct:3d}% "
                    f"({k + 1}/{Nt} frames)"
                )
                next_mark_idx += 1
    finally:
        writer.close()
        plt.close(fig)

    file_size_bytes = int(out_path.stat().st_size) if out_path.exists() else 0
    duration_s = float(Nt) / float(fps)
    summary = {
        "output_path": str(out_path),
        "persistent_output": bool(persistent),
        "fps": int(fps),
        "frame_count": int(Nt),
        "video_duration_s": duration_s,
        "file_size_bytes": file_size_bytes,
        "file_size_mb": float(file_size_bytes) / (1024.0 * 1024.0),
        "counts_min": int(vmin),
        "counts_max": int(vmax),
        "roi_border_count": int(len(border_geoms)),
    }

    if verbose:
        roi_mode = _shell_roi_mode(getattr(ao, "meta", {}))
        if roi_mode in {"country", "multi_country", "latlon_box"}:
            print(
                "[video] note: Analysis contains ROI-masked shell points only. "
                "To color all global shell points, run the simulation on an unmasked shell."
            )
        print("\nCoverage video summary")
        print(f"  output_path: {summary['output_path']}")
        print(f"  persistent_output: {summary['persistent_output']}")
        print(f"  fps: {summary['fps']}")
        print(f"  frame_count: {summary['frame_count']}")
        print(f"  video_duration_s: {summary['video_duration_s']:.2f}")
        print(f"  file_size_mb: {summary['file_size_mb']:.2f}")
        print(f"  counts_min: {summary['counts_min']}")
        print(f"  counts_max: {summary['counts_max']}")
        print(f"  roi_border_count: {summary['roi_border_count']}")

    if tmp_dir_obj is not None:
        summary["_tempdir"] = tmp_dir_obj
    return summary


def _draw_frame_axes(ax: Any, *, frame_name: str, scale_km: float) -> list[Line2D]:
    L = float(max(scale_km, 1.0))
    # Keep axis colors distinct from layer palette (tab20).
    c_x = "#ff00ff"
    c_y = "#00e5ff"
    c_z = "#ffd400"
    ax.plot([0.0, L], [0.0, 0.0], [0.0, 0.0], color=c_x, linewidth=2.2, zorder=20)
    ax.plot([0.0, 0.0], [0.0, L], [0.0, 0.0], color=c_y, linewidth=2.2, zorder=20)
    ax.plot([0.0, 0.0], [0.0, 0.0], [0.0, L], color=c_z, linewidth=2.2, zorder=20)

    return [
        Line2D([0], [0], color=c_x, lw=2.2, label=f"X_{frame_name} axis"),
        Line2D([0], [0], color=c_y, lw=2.2, label=f"Y_{frame_name} axis"),
        Line2D([0], [0], color=c_z, lw=2.2, label=f"Z_{frame_name} axis"),
    ]


def _plot_earth(ax: Any, *, re_km: float) -> None:
    lon = np.linspace(-np.pi, np.pi, 96)
    lat = np.linspace(-0.5 * np.pi, 0.5 * np.pi, 64)
    lon_g, lat_g = np.meshgrid(lon, lat)

    x = re_km * np.cos(lat_g) * np.cos(lon_g)
    y = re_km * np.cos(lat_g) * np.sin(lon_g)
    z = re_km * np.sin(lat_g)

    ax.plot_surface(
        x,
        y,
        z,
        rstride=1,
        cstride=1,
        linewidth=0.0,
        antialiased=False,
        shade=False,
        color="lightgray",
        alpha=0.70,
        zorder=0,
    )


def _build_layer_id(*, layer_specs: list[dict[str, Any]] | None, ns: int) -> np.ndarray:
    if not layer_specs:
        return np.zeros(ns, dtype=np.int64)

    counts: list[int] = []
    for spec in layer_specs:
        if not isinstance(spec, dict):
            return np.zeros(ns, dtype=np.int64)
        n = _extract_layer_count(spec)
        if n is None or n <= 0:
            return np.zeros(ns, dtype=np.int64)
        counts.append(int(n))

    if sum(counts) != ns:
        return np.zeros(ns, dtype=np.int64)

    out = np.empty(ns, dtype=np.int64)
    start = 0
    for lid, n in enumerate(counts):
        out[start : start + n] = lid
        start += n
    return out


def _extract_layer_count(spec: dict[str, Any]) -> int | None:
    for key in ("n_total", "n_sats_total", "T", "n_satellites"):
        v = spec.get(key)
        if isinstance(v, (int, np.integer)):
            return int(v)
    n_planes = spec.get("n_planes")
    sats_per_plane = spec.get("sats_per_plane")
    if isinstance(n_planes, (int, np.integer)) and isinstance(sats_per_plane, (int, np.integer)):
        n = int(n_planes) * int(sats_per_plane)
        if n > 0:
            return n
    for key in ("n_s_est_with_margin", "n_s_est", "n_p_est"):
        v = spec.get(key)
        if isinstance(v, (int, np.integer)) and int(v) > 0:
            # wright_hex often provides estimates; prefer exact plane*sats above.
            if key == "n_p_est" and isinstance(sats_per_plane, (int, np.integer)):
                return int(v) * int(sats_per_plane)
            return int(v)
    return None


def _build_layer_labels(*, layer_specs: list[dict[str, Any]] | None) -> dict[int, str]:
    if not layer_specs:
        return {}
    out: dict[int, str] = {}
    used: dict[str, int] = {}
    for i, spec in enumerate(layer_specs):
        if not isinstance(spec, dict):
            out[i] = f"Layer {i + 1}"
            continue
        base = (
            spec.get("type")
            or spec.get("name")
            or spec.get("notation")
            or f"Layer {i + 1}"
        )
        label = str(base)
        n = used.get(label, 0)
        used[label] = n + 1
        out[i] = label if n == 0 else f"{label} #{n + 1}"
    return out


def _layer_color_map(layer_id: np.ndarray) -> dict[int, tuple[float, float, float, float]]:
    unique = sorted(set(int(x) for x in layer_id.tolist()))
    # Curated dark/high-contrast palette for visibility on light gray Earth + white background.
    palette = [
        "#1f77b4",  # dark blue
        "#ff7f0e",  # orange
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
        "#8c564b",  # brown
        "#e377c2",  # magenta
        "#7f7f00",  # olive
        "#17becf",  # teal (darker than light cyan)
        "#393b79",  # indigo
    ]
    return {lid: palette[i % len(palette)] for i, lid in enumerate(unique)}


def _get_attr_any(obj: Any, names: list[str], default: Any = None) -> Any:
    for n in names:
        if hasattr(obj, n):
            v = getattr(obj, n)
            if v is not None:
                return v
    return default


def _extract_elements(elems0: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if hasattr(elems0, "a_km") and hasattr(elems0, "e"):
        a_km = np.asarray(getattr(elems0, "a_km"), dtype=np.float64)
        e = np.asarray(getattr(elems0, "e"), dtype=np.float64)
        inc = np.asarray(_get_attr_any(elems0, ["i_rad", "i", "inc"]), dtype=np.float64)
        raan = np.asarray(_get_attr_any(elems0, ["raan_rad", "raan", "Omega"]), dtype=np.float64)
        argp = np.asarray(_get_attr_any(elems0, ["argp_rad", "argp", "omega"]), dtype=np.float64)
        M = np.asarray(_get_attr_any(elems0, ["M0_rad", "M0", "M_rad", "M"]), dtype=np.float64)
        if any(v is None for v in (inc, raan, argp, M)):
            raise ValueError("plot_initial_orbits(): missing angle arrays in elems0")
        inc, raan, argp, M = _ensure_radians(inc, raan, argp, M)
        return a_km.ravel(), e.ravel(), inc.ravel(), raan.ravel(), argp.ravel(), M.ravel()

    if isinstance(elems0, dict):
        a_km = np.asarray(elems0.get("a_km", elems0.get("a")), dtype=np.float64)
        e = np.asarray(elems0.get("e", elems0.get("ecc")), dtype=np.float64)
        inc = np.asarray(elems0.get("i_rad", elems0.get("i", elems0.get("inc"))), dtype=np.float64)
        raan = np.asarray(elems0.get("raan_rad", elems0.get("raan")), dtype=np.float64)
        argp = np.asarray(elems0.get("argp_rad", elems0.get("argp")), dtype=np.float64)
        M = np.asarray(elems0.get("M0_rad", elems0.get("M0", elems0.get("M"))), dtype=np.float64)
        inc, raan, argp, M = _ensure_radians(inc, raan, argp, M)
        return a_km.ravel(), e.ravel(), inc.ravel(), raan.ravel(), argp.ravel(), M.ravel()

    arr = np.asarray(elems0, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 6:
        raise ValueError("plot_initial_orbits(): elems0 format not recognized")
    inc, raan, argp, M = _ensure_radians(arr[:, 2], arr[:, 3], arr[:, 4], arr[:, 5])
    return arr[:, 0].ravel(), arr[:, 1].ravel(), inc.ravel(), raan.ravel(), argp.ravel(), M.ravel()


def _ensure_radians(*angles: np.ndarray) -> tuple[np.ndarray, ...]:
    out: list[np.ndarray] = []
    for a in angles:
        x = np.asarray(a, dtype=np.float64)
        if np.nanmax(np.abs(x)) > (2.0 * np.pi + 0.5):
            x = np.deg2rad(x)
        out.append(x)
    return tuple(out)


def _eci_to_ecef_simple(r_eci_km: np.ndarray, *, t_s: float, omega_earth_rad_s: float) -> np.ndarray:
    th = float(omega_earth_rad_s) * float(t_s)
    c = np.cos(th)
    s = np.sin(th)
    r = np.asarray(r_eci_km, dtype=np.float64)
    rot = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (rot @ r.T).T


def _orbit_r_eci_from_elements(
    *,
    a_km: float,
    e: float,
    inc: float,
    raan: float,
    argp: float,
    nu: np.ndarray,
) -> np.ndarray:
    nu = np.asarray(nu, dtype=np.float64)
    p = float(a_km) * (1.0 - float(e) * float(e))
    r = p / np.maximum(1.0 + float(e) * np.cos(nu), 1e-12)

    x_p = r * np.cos(nu)
    y_p = r * np.sin(nu)

    cO, sO = np.cos(raan), np.sin(raan)
    ci, si = np.cos(inc), np.sin(inc)
    cw, sw = np.cos(argp), np.sin(argp)

    R11 = cO * cw - sO * sw * ci
    R12 = -cO * sw - sO * cw * ci
    R21 = sO * cw + cO * sw * ci
    R22 = -sO * sw + cO * cw * ci
    R31 = sw * si
    R32 = cw * si

    x = R11 * x_p + R12 * y_p
    y = R21 * x_p + R22 * y_p
    z = R31 * x_p + R32 * y_p
    return np.stack([x, y, z], axis=1)


def _true_anomaly_from_mean_anomaly(M: float, e: float) -> float:
    M = float((M + np.pi) % (2.0 * np.pi) - np.pi)
    e = float(e)
    E = M if e < 0.8 else (np.pi if M >= 0.0 else -np.pi)
    for _ in range(32):
        f = E - e * np.sin(E) - M
        fp = 1.0 - e * np.cos(E)
        dE = -f / max(fp, 1e-12)
        E = E + dE
        if abs(dE) < 1e-12:
            break

    sin_nu = np.sqrt(max(1.0 - e * e, 1e-12)) * np.sin(E) / max(1.0 - e * np.cos(E), 1e-12)
    cos_nu = (np.cos(E) - e) / max(1.0 - e * np.cos(E), 1e-12)
    return float(np.arctan2(sin_nu, cos_nu))


def _set_equal_3d(ax: Any, *, re_km: float, points: np.ndarray, r_max_km: float | None = None) -> None:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        lim = 1.5 * re_km
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)
        return

    xyz_min = np.min(pts, axis=0)
    xyz_max = np.max(pts, axis=0)
    center = 0.5 * (xyz_min + xyz_max)
    span = float(np.max(xyz_max - xyz_min))
    if r_max_km is not None:
        span = max(span, 2.4 * float(r_max_km))
    span = max(span, 2.8 * re_km)
    lim = 0.55 * span
    ax.set_xlim(center[0] - lim, center[0] + lim)
    ax.set_ylim(center[1] - lim, center[1] + lim)
    ax.set_zlim(center[2] - lim, center[2] + lim)


def _mercator_project(*, lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = ((np.asarray(lon_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat = np.clip(np.asarray(lat_deg, dtype=np.float64), -85.0, 85.0)
    x = np.deg2rad(lon)
    lat_rad = np.deg2rad(lat)
    y = np.log(np.tan((np.pi / 4.0) + (lat_rad / 2.0)))
    return x, y


def _mollweide_project(*, lat_deg: np.ndarray, lon_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lon = ((np.asarray(lon_deg, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat = np.clip(np.asarray(lat_deg, dtype=np.float64), -89.999, 89.999)
    return np.deg2rad(lon), np.deg2rad(lat)


def _format_hhmmss(t_s: float) -> str:
    total = max(0, int(round(float(t_s))))
    hh = total // 3600
    mm = (total % 3600) // 60
    ss = total % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def _draw_mollweide_graticule(ax: Any) -> None:
    xtick_deg = np.arange(-150, 151, 30)
    xtick_rad = np.deg2rad(xtick_deg)
    ax.set_xticks(xtick_rad)
    ax.set_xticklabels([])
    ytick_deg = np.array([-60, -30, 0, 30, 60], dtype=np.float64)
    ax.set_yticks(np.deg2rad(ytick_deg))
    ax.set_yticklabels([f"{int(x):d}" for x in ytick_deg])
    ax.tick_params(axis="x", pad=10, length=0)
    ax.tick_params(axis="y", pad=6)
    for lab in ax.get_yticklabels():
        lab.set_fontsize(8)
        lab.set_color("#444444")


def _draw_mollweide_longitude_labels(ax: Any) -> None:
    xtick_deg = np.arange(-150, 151, 30)
    for deg in xtick_deg:
        x = 0.5 + 0.5 * (float(deg) / 180.0)
        ax.text(
            float(x),
            -0.060,
            f"{int(deg):d}",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#444444",
            clip_on=False,
            zorder=5,
        )


def _shell_roi_mode(meta: Any) -> str | None:
    if not isinstance(meta, dict):
        return None
    inputs = meta.get("inputs")
    if isinstance(inputs, dict):
        shell = inputs.get("shell")
        if isinstance(shell, dict):
            shell_meta = shell.get("meta")
            if isinstance(shell_meta, dict):
                mode = shell_meta.get("roi_mode")
                if isinstance(mode, str):
                    return mode
    return None


def _extract_world_boundaries() -> list[Any]:
    if _regions is None:
        return []
    try:
        gdf = _regions._load_naturalearth_lowres()
    except Exception:
        return []
    out = []
    for _, row in gdf.iterrows():
        geom = getattr(row, "geometry", None)
        if geom is not None:
            out.append(geom)
    return out


def _extract_roi_country_boundaries(meta: Any) -> list[Any]:
    if _regions is None or not isinstance(meta, dict):
        return []
    countries: list[str] = []
    inputs = meta.get("inputs")
    if isinstance(inputs, dict):
        shell = inputs.get("shell")
        if isinstance(shell, dict):
            shell_meta = shell.get("meta")
            if isinstance(shell_meta, dict):
                if isinstance(shell_meta.get("countries"), list):
                    countries.extend([str(x) for x in shell_meta["countries"] if str(x)])
                elif isinstance(shell_meta.get("country_match"), str):
                    countries.append(str(shell_meta["country_match"]))
                elif isinstance(shell_meta.get("country_query"), str):
                    countries.append(str(shell_meta["country_query"]))
    out = []
    seen = set()
    for country in countries:
        if country in seen:
            continue
        seen.add(country)
        try:
            geom, _ = _regions.country_polygon(country)
            out.append((country, geom))
        except Exception:
            continue
    return out


def _extract_country_boundaries(countries: list[str]) -> list[Any]:
    if _regions is None:
        return []
    out = []
    seen = set()
    for country in countries:
        key = str(country).strip()
        if not key or key.lower() in seen:
            continue
        seen.add(key.lower())
        try:
            geom, _ = _regions.country_polygon(key)
            out.append((key, geom))
        except Exception:
            continue
    return out


def _plot_roi_boundaries(ax: Any, border_geoms: list[Any]) -> None:
    for country, geom in border_geoms:
        for line_lon, line_lat in _iter_geometry_lonlat_lines(geom):
            for seg_lon, seg_lat in _split_dateline_segments(line_lon, line_lat):
                x, y = _mollweide_project(lat_deg=np.asarray(seg_lat), lon_deg=np.asarray(seg_lon))
                ax.plot(x, y, color="#cc1f1a", linewidth=1.2, alpha=0.98, zorder=2.6, label=country)


def _iter_geometry_lonlat_lines(geom: Any) -> list[tuple[np.ndarray, np.ndarray]]:
    out: list[tuple[np.ndarray, np.ndarray]] = []
    if geom is None:
        return out
    gtype = getattr(geom, "geom_type", "")
    if gtype == "Polygon":
        xs, ys = geom.exterior.xy
        out.append((np.asarray(xs), np.asarray(ys)))
    elif gtype == "MultiPolygon":
        for poly in geom.geoms:
            xs, ys = poly.exterior.xy
            out.append((np.asarray(xs), np.asarray(ys)))
    return out


def _plot_world_boundaries(ax: Any, world_geoms: list[Any]) -> None:
    for geom in world_geoms:
        for line_lon, line_lat in _iter_geometry_lonlat_lines(geom):
            for seg_lon, seg_lat in _split_dateline_segments(line_lon, line_lat):
                x, y = _mollweide_project(lat_deg=np.asarray(seg_lat), lon_deg=np.asarray(seg_lon))
                ax.plot(x, y, color="#000000", linewidth=0.55, alpha=0.82, zorder=2)


def _split_dateline_segments(lon: np.ndarray, lat: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    lon_arr = ((np.asarray(lon, dtype=np.float64) + 180.0) % 360.0) - 180.0
    lat_arr = np.asarray(lat, dtype=np.float64)
    if lon_arr.size <= 1:
        return [(lon_arr, lat_arr)]

    breaks = np.where(np.abs(np.diff(lon_arr)) > 180.0)[0]
    if breaks.size == 0:
        return [(lon_arr, lat_arr)]

    segments: list[tuple[np.ndarray, np.ndarray]] = []
    start = 0
    for b in breaks:
        end = int(b + 1)
        if end - start >= 2:
            segments.append((lon_arr[start:end], lat_arr[start:end]))
        start = end
    if lon_arr.size - start >= 2:
        segments.append((lon_arr[start:], lat_arr[start:]))
    return segments


def _prepare_country_stats(
    *,
    counts: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    countries: list[str],
) -> list[dict[str, Any]]:
    if _regions is None:
        return []

    out: list[dict[str, Any]] = []
    for country in countries:
        label = str(country).strip()
        if not label:
            continue
        try:
            mask, meta = _regions.mask_country(lat_deg, lon_deg, label)
        except Exception:
            continue
        if not np.any(mask):
            continue
        match_label = str(meta.get("country_match", meta.get("country_query", label)))
        min_count = int(np.min(counts[:, mask]))
        out.append({
            "label": match_label,
            "mask": mask,
            "min_count": min_count,
        })
    return out
