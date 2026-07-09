"""
Plot salvo size vs required satellites from sweep optimization results.

Reads JSON result files produced by sweep_optimize.py from:

    results/{country_slug}/burnout-vel-{v}km-s/intercept-window-{w}s/
        intercept-alt-{h}km/max-accel-{g}g/orbit-alt-{alt}km/
        interceptors-per-sat-{n}/salvo-size-{s}/{timestamp}_result.json

One figure is produced per interceptors-per-sat value.
Within each figure, series are keyed by (orbit_alt_km, v_bo_km_s):

  • Colour family → orbit altitude  (one hue family per altitude, light→dark)
  • Shade          → burnout velocity (lighter = slower, darker = faster)

Hover the mouse over a data point to see a tooltip with detailed metadata.
Black-filled markers indicate runs that were time-limited or user-interrupted.
"""
from __future__ import annotations

import colorsys
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RESULTS_ROOT: Path = Path(__file__).resolve().parents[1] / "results"
COUNTRY_NAME: str  = "North Korea"   # Display name; folder slug is derived automatically
SHOW_PLOT: bool    = True


# ---------------------------------------------------------------------------
# Colour palette
# ---------------------------------------------------------------------------
# One base colour per altitude rank.  Within each altitude, burnout velocities
# are shaded from light (low v_bo) to dark (high v_bo).
_ALT_BASE_COLOURS: list[str] = [
    "#1f77b4",   # blue   — altitude rank 0
    "#ff7f0e",   # orange — altitude rank 1
    "#2ca02c",   # green  — altitude rank 2
    "#d62728",   # red    — altitude rank 3
    "#9467bd",   # purple — altitude rank 4
    "#8c564b",   # brown  — altitude rank 5
    "#e377c2",   # pink   — altitude rank 6
    "#7f7f7f",   # grey   — altitude rank 7
]


def _build_colour_map(
    sorted_alts: list[float],
    sorted_vbos: list[float],
) -> dict[tuple[float, float], tuple[float, float, float]]:
    """Return {(alt_km, vbo_km_s): rgb_tuple}.

    Each altitude gets one base hue.  Burnout velocities are mapped from
    lightness 0.72 (low v_bo, pale) down to 0.25 (high v_bo, dark) so that
    faster missiles are always the darkest line in their colour family.
    """
    colour_map: dict[tuple[float, float], tuple[float, float, float]] = {}
    n_vel = len(sorted_vbos)
    for alt_idx, alt in enumerate(sorted_alts):
        base_rgb = mcolors.to_rgb(_ALT_BASE_COLOURS[alt_idx % len(_ALT_BASE_COLOURS)])
        h, _l, s = colorsys.rgb_to_hls(*base_rgb)
        for vel_idx, vbo in enumerate(sorted_vbos):
            if n_vel == 1:
                lightness = 0.45
            else:
                # Light (slow) → dark (fast)
                lightness = 0.72 - 0.47 * (vel_idx / (n_vel - 1))
            r, g, b = colorsys.hls_to_rgb(h, max(0.10, min(0.92, lightness)), s)
            colour_map[(alt, vbo)] = (r, g, b)
    return colour_map


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResultPoint:
    v_bo_km_s:            float
    orbit_alt_km:         float   # rounded actual altitude parsed from folder name
    t_window_s:           float
    intercept_alt_km:     float
    max_accel_g:          float
    interceptors_per_sat: int
    salvo_size:           int
    n_satellites:         int
    obj_bound:            Optional[float]
    status_str:           str
    mip_gap:              float
    json_path:            Path


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------
def _parse_float_from_path(path: Path, pattern: str) -> float:
    """Search every path component for *pattern* and return the first match."""
    for part in path.parts:
        m = re.search(pattern, part)
        if m:
            return float(m.group(1))
    raise ValueError(f"Pattern {pattern!r} not found in path: {path}")


def _parse_int_from_path(path: Path, pattern: str) -> int:
    return int(round(_parse_float_from_path(path, pattern)))


def _latest_json_in_dir(folder: Path) -> Optional[Path]:
    jsons = sorted(folder.glob("*_result.json"), key=lambda p: p.stat().st_mtime)
    return jsons[-1] if jsons else None


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def _collect_country_results(country_slug: str) -> list[ResultPoint]:
    country_dir = RESULTS_ROOT / country_slug
    if not country_dir.is_dir():
        raise FileNotFoundError(
            f"Country results folder not found: {country_dir}\n"
            f"Expected structure: {RESULTS_ROOT}/{{country_slug}}/burnout-vel-*/…"
        )

    points: list[ResultPoint] = []

    # Every leaf folder is a salvo-size-* directory; pick the latest JSON inside.
    for salvo_dir in sorted(country_dir.rglob("salvo-size-*")):
        if not salvo_dir.is_dir():
            continue
        json_path = _latest_json_in_dir(salvo_dir)
        if json_path is None:
            continue

        try:
            # Support both full S3-style names and local abbreviated names
            v_bo          = _parse_float_from_path(json_path, r"(?:burnout-vel-|vbo-)([\d.]+)km-s")
            t_window      = _parse_float_from_path(json_path, r"(?:intercept-window-|tw-)([\d.]+)s")
            intercept_alt = _parse_float_from_path(json_path, r"(?:intercept-alt-|ia-)([\d.]+)km")
            max_accel     = _parse_float_from_path(json_path, r"(?:max-accel-|ag-)([\d.]+)g")
            orbit_alt     = _parse_float_from_path(json_path, r"(?:orbit-alt-|oa-)([\d.]+)km")
            n_int         = _parse_int_from_path(json_path,   r"(?:interceptors-per-sat-|int-)(\d+)")
            salvo_size    = _parse_int_from_path(json_path,   r"(?:salvo-size-|sal-)(\d+)")
        except ValueError as exc:
            print(f"  [warn] skipping {json_path}: {exc}")
            continue

        try:
            with json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] could not read {json_path}: {exc}")
            continue

        n_sat_raw = data.get("n_satellites")
        if n_sat_raw is None:
            print(f"  [skip] no solution (n_satellites=null): {json_path}")
            continue

        run_cfg       = data.get("run_config", {})
        obj_bound_raw = run_cfg.get("obj_bound")
        mip_gap_raw   = data.get("mip_gap")

        points.append(ResultPoint(
            v_bo_km_s            = v_bo,
            orbit_alt_km         = orbit_alt,
            t_window_s           = t_window,
            intercept_alt_km     = intercept_alt,
            max_accel_g          = max_accel,
            interceptors_per_sat = n_int,
            salvo_size           = salvo_size,
            n_satellites         = int(n_sat_raw),
            obj_bound            = float(obj_bound_raw) if obj_bound_raw is not None else None,
            status_str           = str(data.get("status_str", "Unknown")),
            mip_gap              = float(mip_gap_raw) if mip_gap_raw is not None else 1.0,
            json_path            = json_path,
        ))

    return points


def _dedup_best(points: list[ResultPoint]) -> list[ResultPoint]:
    """If multiple runs share the same (alt, v_bo, n_int, salvo), keep the best."""
    best: dict[tuple, ResultPoint] = {}
    for pt in points:
        key = (pt.orbit_alt_km, pt.v_bo_km_s, pt.interceptors_per_sat, pt.salvo_size)
        if key not in best or pt.n_satellites < best[key].n_satellites:
            best[key] = pt
    return list(best.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    country_slug = COUNTRY_NAME.lower().replace(" ", "-")
    print(f"Loading results for '{COUNTRY_NAME}' from {RESULTS_ROOT / country_slug} …")

    points = _collect_country_results(country_slug)
    if not points:
        raise ValueError(f"No *_result.json files found under {RESULTS_ROOT / country_slug}")

    points = _dedup_best(points)
    print(f"Loaded {len(points)} result point(s) after dedup.\n")

    # Sorted unique dimensions
    sorted_alts  = sorted({pt.orbit_alt_km         for pt in points})
    sorted_vbos  = sorted({pt.v_bo_km_s            for pt in points})
    sorted_n_int = sorted({pt.interceptors_per_sat  for pt in points})

    colour_map = _build_colour_map(sorted_alts, sorted_vbos)

    # Human-readable altitude list for the figure title
    alt_label = ", ".join(f"{a:.0f} km" for a in sorted_alts)

    figures: list[plt.Figure] = []

    # ------------------------------------------------------------------
    # One figure per interceptors-per-sat value
    # ------------------------------------------------------------------
    for n_int in sorted_n_int:
        pts_fig = [pt for pt in points if pt.interceptors_per_sat == n_int]
        if not pts_fig:
            continue

        fig, ax = plt.subplots(figsize=(11.0, 6.5))
        ax.set_facecolor("#ffffff")
        hover_points: list[tuple[float, float, str]] = []

        # Series sorted by altitude first, then burnout velocity
        series_keys = sorted(
            {(pt.orbit_alt_km, pt.v_bo_km_s) for pt in pts_fig},
            key=lambda k: (k[0], k[1]),
        )

        for alt, vbo in series_keys:
            series_pts = sorted(
                [pt for pt in pts_fig if pt.orbit_alt_km == alt and pt.v_bo_km_s == vbo],
                key=lambda pt: pt.salvo_size,
            )
            if not series_pts:
                continue

            rgb   = colour_map[(alt, vbo)]
            label = f"{alt:.0f} km  ·  {vbo:.1f} km/s"

            x = [pt.salvo_size   for pt in series_pts]
            y = [pt.n_satellites for pt in series_pts]

            ax.plot(x, y, marker="o", linewidth=2.0, markersize=6.0, color=rgb, label=label)

            # Black-filled markers for time-limited / interrupted runs
            limited = [
                pt for pt in series_pts
                if "interrupt" in pt.status_str.lower() or "time" in pt.status_str.lower()
            ]
            if limited:
                ax.scatter(
                    [pt.salvo_size   for pt in limited],
                    [pt.n_satellites for pt in limited],
                    s          = 60,
                    facecolors = "#000000",
                    edgecolors = rgb,
                    linewidths = 1.8,
                    zorder     = 4,
                )

            for pt in series_pts:
                bound_str = f"{pt.obj_bound:.0f}" if pt.obj_bound is not None else "—"
                hover_points.append((
                    float(pt.salvo_size),
                    float(pt.n_satellites),
                    (f"{label}\n"
                     f"Salvo = {pt.salvo_size}\n"
                     f"Satellites = {pt.n_satellites}\n"
                     f"Bound ≤ {bound_str}\n"
                     f"MIP gap = {pt.mip_gap * 100:.1f}%\n"
                     f"Status = {pt.status_str}"),
                ))

        int_word  = "Interceptors" if n_int != 1 else "Interceptor"
        int_label = f"{n_int} {int_word} Per Satellite"
        ax.set_title(
            f"{COUNTRY_NAME}: Salvo Size vs Required Satellites\n"
            f"{int_label}  ·  Orbit altitude: {alt_label}"
        )
        ax.set_xlabel("Salvo Size")
        ax.set_ylabel("Required Satellites")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        ax.legend(frameon=True, fontsize=8, title="Altitude  ·  v_bo")

        # Hover annotation
        annot = ax.annotate(
            "",
            xy=(0.0, 0.0),
            xytext=(12, 12),
            textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "white", "ec": "black", "alpha": 0.95},
            fontsize=8,
        )
        annot.set_visible(False)

        hover_xy = (
            np.array([(hx, hy) for hx, hy, _ in hover_points], dtype=np.float64)
            if hover_points else np.empty((0, 2))
        )

        def _on_move(
            event: object,
            *,
            _ax   = ax,
            _fig  = fig,
            _ann  = annot,
            _hxy  = hover_xy,
            _hpts = hover_points,
        ) -> None:
            if event is None or getattr(event, "inaxes", None) is not _ax:
                if _ann.get_visible():
                    _ann.set_visible(False)
                    _fig.canvas.draw_idle()
                return
            if _hxy.size == 0:
                return
            xdata = getattr(event, "xdata", None)
            ydata = getattr(event, "ydata", None)
            if xdata is None or ydata is None:
                return

            dx    = _hxy[:, 0] - float(xdata)
            dy    = _hxy[:, 1] - float(ydata)
            dist2 = dx * dx + dy * dy
            best  = int(np.argmin(dist2))

            xspan = max(_ax.get_xlim()[1] - _ax.get_xlim()[0], 1.0)
            yspan = max(_ax.get_ylim()[1] - _ax.get_ylim()[0], 1.0)
            if (dx[best] / xspan) ** 2 + (dy[best] / yspan) ** 2 > 0.0012:
                if _ann.get_visible():
                    _ann.set_visible(False)
                    _fig.canvas.draw_idle()
                return

            _ann.xy = (_hxy[best, 0], _hxy[best, 1])
            _ann.set_text(_hpts[best][2])
            _ann.set_visible(True)
            _fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", _on_move)
        fig.tight_layout()
        figures.append(fig)

    # ------------------------------------------------------------------
    # Console summary table
    # ------------------------------------------------------------------
    print(f"\nCountry folder : {RESULTS_ROOT / country_slug}")
    print("\nSeries summary tables:")
    for n_int in sorted_n_int:
        pts_group = sorted(
            [pt for pt in points if pt.interceptors_per_sat == n_int],
            key=lambda pt: (pt.orbit_alt_km, pt.v_bo_km_s, pt.salvo_size),
        )
        if not pts_group:
            continue

        int_word  = "Interceptors" if n_int != 1 else "Interceptor"
        int_label = f"{n_int} {int_word} Per Satellite"
        print(f"\n{'=' * 84}")
        print(f"  {int_label}")
        print(f"{'=' * 84}")
        header = (
            f"  {'Alt km':>7}  {'v_bo km/s':>10}  {'Salvo':>7}  "
            f"{'Satellites':>11}  {'Bound':>8}  {'MIP gap':>8}  {'Status'}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for pt in pts_group:
            bound_str = f"{pt.obj_bound:.0f}" if pt.obj_bound is not None else "—"
            mip_pct   = f"{pt.mip_gap * 100:.2f}%"
            print(
                f"  {pt.orbit_alt_km:>7.0f}  {pt.v_bo_km_s:>10.1f}  {pt.salvo_size:>7}  "
                f"{pt.n_satellites:>11}  {bound_str:>8}  {mip_pct:>8}  {pt.status_str}"
            )

    if SHOW_PLOT and figures:
        plt.show()
    elif figures:
        plt.close("all")


if __name__ == "__main__":
    main()
