"""
cloud_plot_best_results.py

Plot salvo size vs required satellites from sweep results stored in S3.

Mirrors plot_best_results_by_country.py exactly — same colour scheme, hover
tooltips, console summary table, and deduplication logic — but pulls JSON
result files directly from S3 instead of a local results/ folder.

All JSON files are downloaded into memory (no local write).  Use --save-local
to also mirror them into a local results/ folder compatible with
plot_best_results_by_country.py.

Usage:
    python scripts/cloud_plot_best_results.py --country China
    python scripts/cloud_plot_best_results.py --country China --filter-alt 400 --filter-interceptors 1
    python scripts/cloud_plot_best_results.py --country China --save-local
"""
from __future__ import annotations

import argparse
import colorsys
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import boto3
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# AWS / S3 configuration
# ---------------------------------------------------------------------------
S3_BUCKET:  str = "sbi-optimization-runs"
AWS_REGION: str = "us-east-1"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
COUNTRY_NAME: str  = "Iran"   # Default; override with --country
SHOW_PLOT:    bool = True

# Local results root used when --save-local is passed.
LOCAL_RESULTS_ROOT: Path = Path("results")


# ---------------------------------------------------------------------------
# Colour palette (identical to plot_best_results_by_country.py)
# ---------------------------------------------------------------------------
_ALT_BASE_COLOURS: list[str] = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
]


def _build_colour_map(
    sorted_alts: list[float],
    sorted_vbos: list[float],
) -> dict[tuple[float, float], tuple[float, float, float]]:
    colour_map: dict[tuple[float, float], tuple[float, float, float]] = {}
    n_vel = len(sorted_vbos)
    for alt_idx, alt in enumerate(sorted_alts):
        base_rgb = mcolors.to_rgb(_ALT_BASE_COLOURS[alt_idx % len(_ALT_BASE_COLOURS)])
        h, _l, s = colorsys.rgb_to_hls(*base_rgb)
        for vel_idx, vbo in enumerate(sorted_vbos):
            if n_vel == 1:
                lightness = 0.45
            else:
                lightness = 0.72 - 0.47 * (vel_idx / (n_vel - 1))
            r, g, b = colorsys.hls_to_rgb(h, max(0.10, min(0.92, lightness)), s)
            colour_map[(alt, vbo)] = (r, g, b)
    return colour_map


# ---------------------------------------------------------------------------
# Data model (identical to plot_best_results_by_country.py)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResultPoint:
    v_bo_km_s:            float
    orbit_alt_km:         float
    t_window_s:           float
    intercept_alt_km:     float
    max_accel_g:          float
    interceptors_per_sat: int
    salvo_size:           int
    n_satellites:         int
    obj_bound:            Optional[float]
    status_str:           str
    mip_gap:              float
    s3_key:               str   # source S3 key (replaces json_path)


# ---------------------------------------------------------------------------
# Path / key helpers (identical regex patterns to plot_best_results_by_country.py)
# ---------------------------------------------------------------------------
def _parse_float_from_key(key: str, pattern: str) -> float:
    m = re.search(pattern, key)
    if not m:
        raise ValueError(f"Pattern {pattern!r} not found in S3 key: {key}")
    return float(m.group(1))


def _parse_int_from_key(key: str, pattern: str) -> int:
    return int(round(_parse_float_from_key(key, pattern)))


# ---------------------------------------------------------------------------
# S3 data collection
# ---------------------------------------------------------------------------
def _collect_s3_results(
    s3,
    country_slug: str,
    filter_alt: Optional[float],
    filter_vbo: Optional[float],
    filter_interceptors: Optional[int],
    save_local: bool,
) -> list[ResultPoint]:
    paginator    = s3.get_paginator("list_objects_v2")
    broad_prefix = f"country/{country_slug}/"
    points: list[ResultPoint] = []
    n_checked = 0
    n_skipped = 0

    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=broad_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]

            # Only result JSON sidecars (not _config.json)
            if not key.endswith(".json") or key.endswith("_config.json"):
                continue

            # Optional pre-download filters (key-string only — no fetch needed)
            if filter_alt is not None:
                alt_int = int(round(filter_alt))
                if f"orbit-alt-{alt_int}km" not in key:
                    n_skipped += 1
                    continue
            if filter_vbo is not None:
                if f"burnout-vel-{filter_vbo:.1f}km-s" not in key:
                    n_skipped += 1
                    continue
            if filter_interceptors is not None:
                if f"interceptors-per-sat-{filter_interceptors}" not in key:
                    n_skipped += 1
                    continue

            # Parse dimensions from the key before fetching
            try:
                v_bo          = _parse_float_from_key(key, r"burnout-vel-([\d.]+)km-s")
                t_window      = _parse_float_from_key(key, r"intercept-window-([\d.]+)s")
                intercept_alt = _parse_float_from_key(key, r"intercept-alt-([\d.]+)km")
                max_accel     = _parse_float_from_key(key, r"max-accel-([\d.]+)g")
                orbit_alt     = _parse_float_from_key(key, r"orbit-alt-([\d.]+)km")
                n_int         = _parse_int_from_key(key,   r"interceptors-per-sat-(\d+)")
                salvo_size    = _parse_int_from_key(key,   r"salvo-size-(\d+)")
            except ValueError as exc:
                print(f"  [warn] skipping {key}: {exc}")
                continue

            # Fetch and parse JSON into memory
            try:
                resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
                data = json.loads(resp["Body"].read().decode("utf-8"))
            except Exception as exc:
                print(f"  [warn] could not read {key}: {exc}")
                continue

            n_checked += 1

            n_sat_raw = data.get("n_satellites")
            if n_sat_raw is None:
                print(f"  [skip] no solution (n_satellites=null): {key}")
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
                s3_key               = key,
            ))

            # Optionally mirror to local results/ folder
            if save_local:
                local_path = LOCAL_RESULTS_ROOT / key
                local_path.parent.mkdir(parents=True, exist_ok=True)
                with local_path.open("w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2)

    print(f"  Checked {n_checked} JSON(s), skipped {n_skipped} by filter, loaded {len(points)} result point(s).")
    return points


def _dedup_best(points: list[ResultPoint]) -> list[ResultPoint]:
    """If multiple runs share (alt, v_bo, n_int, salvo), keep the one with fewest satellites."""
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
    parser = argparse.ArgumentParser(
        description="Plot sweep results for a country pulled from S3."
    )
    parser.add_argument(
        "--country",
        default=COUNTRY_NAME,
        help=f"Country name (default: {COUNTRY_NAME})",
    )
    parser.add_argument(
        "--filter-alt",
        type=float,
        default=None,
        metavar="KM",
        help="Only include results at this orbit altitude (km).",
    )
    parser.add_argument(
        "--filter-vbo",
        type=float,
        default=None,
        metavar="KM_S",
        help="Only include results at this burnout velocity (km/s).",
    )
    parser.add_argument(
        "--filter-interceptors",
        type=int,
        default=None,
        metavar="N",
        help="Only include results with this interceptors-per-sat value.",
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        help=f"Mirror downloaded JSONs into {LOCAL_RESULTS_ROOT}/ for use with plot_best_results_by_country.py.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the interactive plot (console summary only).",
    )
    args = parser.parse_args()

    country_slug = args.country.lower().replace(" ", "-")
    print(f"Loading results for '{args.country}' from s3://{S3_BUCKET}/country/{country_slug}/ ...")

    if args.filter_alt is not None:
        print(f"  Filter: orbit-alt ≈ {args.filter_alt:.0f} km")
    if args.filter_vbo is not None:
        print(f"  Filter: burnout-vel = {args.filter_vbo:.1f} km/s")
    if args.filter_interceptors is not None:
        print(f"  Filter: interceptors-per-sat = {args.filter_interceptors}")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    points = _collect_s3_results(
        s3,
        country_slug=country_slug,
        filter_alt=args.filter_alt,
        filter_vbo=args.filter_vbo,
        filter_interceptors=args.filter_interceptors,
        save_local=args.save_local,
    )
    if not points:
        raise ValueError(
            f"No results found for '{args.country}' in s3://{S3_BUCKET}/country/{country_slug}/. "
            "Check filters or run cloud_sweep.py first."
        )

    points = _dedup_best(points)
    print(f"After dedup: {len(points)} result point(s).\n")

    sorted_alts  = sorted({pt.orbit_alt_km         for pt in points})
    sorted_vbos  = sorted({pt.v_bo_km_s            for pt in points})
    sorted_n_int = sorted({pt.interceptors_per_sat  for pt in points})

    colour_map = _build_colour_map(sorted_alts, sorted_vbos)
    alt_label  = ", ".join(f"{a:.0f} km" for a in sorted_alts)

    figures: list[plt.Figure] = []

    for n_int in sorted_n_int:
        pts_fig = [pt for pt in points if pt.interceptors_per_sat == n_int]
        if not pts_fig:
            continue

        fig, ax = plt.subplots(figsize=(11.0, 6.5))
        ax.set_facecolor("#ffffff")
        hover_points: list[tuple[float, float, str]] = []

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
            f"{args.country}: Salvo Size vs Required Satellites\n"
            f"{int_label}  ·  Orbit altitude: {alt_label}"
        )
        ax.set_xlabel("Salvo Size")
        ax.set_ylabel("Required Satellites")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        ax.legend(frameon=True, fontsize=8, title="Altitude  ·  v_bo")

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
    print(f"\nS3 source : s3://{S3_BUCKET}/country/{country_slug}/")
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

    show = SHOW_PLOT and not args.no_plot
    if show and figures:
        plt.show()
    elif figures:
        plt.close("all")


if __name__ == "__main__":
    main()
