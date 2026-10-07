"""
cloud_plot_principal_results.py

Plot salvo size vs required satellites for PRINCIPAL runs only (run_config.
is_principal_set == true), pulled directly from S3.

Unlike cloud_plot_best_results.py (which is scoped to one --country and reads
a few dimensions off the S3 key), this script:
  - Scans every result across all targets (country AND prespecified mode) in
    one pass, since is_principal_set can only be read from inside the JSON
    body -- there's no way to pre-filter on it from the S3 key alone.
  - Exposes every run input as an optional filter, edited as module constants
    at the top of this file (not CLI flags) -- salvo size, interceptors/sat,
    v_bo, target altitude, intercept altitude, detection/decision/burnout
    time, a_g, country, and a regex over run_name (e.g. r"^(NK|RU|CN|IR)" to
    match an acronym prefix). Each filter is a list; an empty list means "do
    not discriminate on this dimension" -- leaving everything blank plots
    every principal run found.
  - Groups surviving runs into one figure per SCENARIO -- every input
    dimension held fixed except v_bo_km_s (color) and salvo_size (x-axis).
    Two runs only ever share a plot if every other input matches, so the
    script never overlays results that aren't actually comparable. If your
    filters leave more than one value on some other dimension (e.g. two
    different detection_time_s), that dimension splits into separate
    figures automatically rather than being silently mixed into one.

Usage:
    python scripts/cloud_plot_principal_results.py
    python scripts/cloud_plot_principal_results.py --no-plot
    python scripts/cloud_plot_principal_results.py --save-local
"""
from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import boto3
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# AWS / S3 configuration
# ---------------------------------------------------------------------------
S3_BUCKET:  str = "sbi-optimization-runs"
AWS_REGION: str = "us-east-1"

SHOW_PLOT: bool = True
LOCAL_RESULTS_ROOT: Path = Path("results")   # used only with --save-local

# ---------------------------------------------------------------------------
# Filters -- edit these to control what gets plotted.
#
# Every filter below is a list. An empty list means "do not discriminate on
# this dimension" -- every value passes. A non-empty list keeps only runs
# whose value is IN that list, so you can select multiple values at once
# (e.g. FILTER_SALVO_SIZES = [10, 50] keeps both salvo sizes, each still
# split correctly across figures/series as appropriate).
#
# All of these are compared against run_config fields in the stored JSON,
# not against the S3 key -- so they work identically for country-mode and
# prespecified-mode runs.
# ---------------------------------------------------------------------------
FILTER_SALVO_SIZES:             list[int]   = []
FILTER_N_INTERCEPTORS_PER_SAT:  list[int]   = [1]
FILTER_V_BO_KM_S:                list[float] = [6.0]
FILTER_TARGET_ALTITUDE_KM:      list[float] = []
FILTER_INTERCEPT_ALT_KM:        list[float] = []
FILTER_DETECTION_TIME_S:        list[float] = [15.0]
FILTER_DECISION_TIME_S:         list[float] = [0.0]
FILTER_BURNOUT_TIME_S:          list[float] = [170.0]   # target_missile_burnout_time_s
FILTER_A_G:                     list[float] = [10.0]
FILTER_TARGET_MODE:             list[str]   = []   # "country" / "prespecified"
FILTER_COUNTRIES:               list[str]   = []   # country-mode only, e.g. ["China", "Iran"]

# Regex applied to run_name with re.search (case-insensitive). None/"" = no
# filter. Example: r"^(NK|RU|CN|IR)" to keep only run names tagged with one
# of those country acronyms.
FILTER_RUN_NAME_REGEX: Optional[str] = r"NK"

_FLOAT_TOL = 1e-6


def _matches_float(value: float, allowed: list[float]) -> bool:
    if not allowed:
        return True
    return any(math.isclose(value, a, abs_tol=_FLOAT_TOL) for a in allowed)


def _matches_int(value: int, allowed: list[int]) -> bool:
    if not allowed:
        return True
    return value in allowed


def _matches_str(value: str, allowed: list[str]) -> bool:
    if not allowed:
        return True
    value_norm = value.strip().lower()
    return any(value_norm == a.strip().lower() for a in allowed)


# ---------------------------------------------------------------------------
# Color palette -- one qualitative color per v_bo value within a figure.
# (Scenario dimensions like target/altitude/interceptors are figure-level,
# not color-level, so we only need one free color dimension here.)
# ---------------------------------------------------------------------------
_V_BO_COLOURS: list[str] = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
]


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunPoint:
    # --- scenario dimensions (figure-grouping key) ---
    target_mode:            str
    target_label:           str     # e.g. "China" or "Sohae Satellite Launching Station... (+5 more)"
    n_interceptors_per_sat: int
    target_altitude_km:     float
    intercept_alt_km:       float
    detection_time_s:       float
    decision_time_s:        float
    burnout_time_s:         float
    a_g:                    float
    # --- series dimension (color) ---
    v_bo_km_s:               float
    # --- x-axis ---
    salvo_size:              int
    # --- result ---
    n_satellites:            int
    obj_bound:               Optional[float]
    status_str:              str
    mip_gap:                 float
    # --- metadata, informational only ---
    run_name:                Optional[str]
    actual_alt_km:            float
    s3_key:                   str

    @property
    def scenario_key(self) -> tuple:
        return (
            self.target_mode,
            self.target_label,
            self.n_interceptors_per_sat,
            self.target_altitude_km,
            self.intercept_alt_km,
            self.detection_time_s,
            self.decision_time_s,
            self.burnout_time_s,
            self.a_g,
        )

    @property
    def scenario_label(self) -> str:
        return (
            f"{self.target_label}  |  {self.n_interceptors_per_sat} interceptor(s)/sat  |  "
            f"target alt {self.target_altitude_km:.0f} km  |  intercept alt {self.intercept_alt_km:.0f} km\n"
            f"detection {self.detection_time_s:.0f}s  decision {self.decision_time_s:.0f}s  "
            f"burnout {self.burnout_time_s:.0f}s  a_g {self.a_g:.1f}"
        )


def _target_label(run_cfg: dict) -> str:
    mode = str(run_cfg.get("target_mode", "country")).strip().lower()
    if mode == "prespecified":
        pts = run_cfg.get("target_points") or []
        if not pts:
            return "prespecified (unknown)"
        first_name = str(pts[0][0]).strip()
        return first_name if len(pts) == 1 else f"{first_name} (+{len(pts) - 1} more, {len(pts)} pts)"
    country_raw = run_cfg.get("target_country")
    if isinstance(country_raw, list):
        return "+".join(str(c).strip() for c in country_raw)
    return str(country_raw).strip() if country_raw else "country (unknown)"


def _country_for_filter(run_cfg: dict) -> Optional[str]:
    """Return the country name for FILTER_COUNTRIES matching, or None (prespecified mode)."""
    if str(run_cfg.get("target_mode", "country")).strip().lower() != "country":
        return None
    country_raw = run_cfg.get("target_country")
    if isinstance(country_raw, list):
        return "+".join(str(c).strip() for c in country_raw)
    return str(country_raw).strip() if country_raw else None


# ---------------------------------------------------------------------------
# S3 data collection
# ---------------------------------------------------------------------------
def _collect_principal_points(s3, save_local: bool) -> tuple[list[RunPoint], int, int]:
    paginator = s3.get_paginator("list_objects_v2")
    points: list[RunPoint] = []
    n_checked = 0
    n_non_principal = 0

    run_name_re = re.compile(FILTER_RUN_NAME_REGEX, re.IGNORECASE) if FILTER_RUN_NAME_REGEX else None

    for prefix in ("country/", "prespecified/"):
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if not key.endswith(".json") or key.endswith("_config.json"):
                    continue

                try:
                    resp = s3.get_object(Bucket=S3_BUCKET, Key=key)
                    data = json.loads(resp["Body"].read().decode("utf-8"))
                except Exception as exc:
                    print(f"  [warn] could not read {key}: {exc}")
                    continue
                n_checked += 1

                run_cfg = data.get("run_config", {})
                if run_cfg.get("is_principal_set") is not True:
                    n_non_principal += 1
                    continue

                n_sat_raw = data.get("n_satellites")
                if n_sat_raw is None:
                    print(f"  [skip] principal run with no solution (n_satellites=null): {key}")
                    continue

                run_name = run_cfg.get("run_name")
                if run_name_re is not None and not run_name_re.search(str(run_name or "")):
                    continue

                target_mode = str(run_cfg.get("target_mode", "country")).strip().lower()
                if not _matches_str(target_mode, FILTER_TARGET_MODE):
                    continue
                country_val = _country_for_filter(run_cfg)
                if FILTER_COUNTRIES and (country_val is None or not _matches_str(country_val, FILTER_COUNTRIES)):
                    continue

                try:
                    salvo_size    = int(run_cfg["salvo_size"])
                    n_int         = int(run_cfg["n_interceptors_per_sat"])
                    v_bo          = float(run_cfg["v_bo_km_s"])
                    target_alt    = float(run_cfg["target_altitude_km"])
                    intercept_alt = float(run_cfg["intercept_alt_km"])
                    detect_s      = float(run_cfg["detection_time_s"])
                    decide_s      = float(run_cfg["decision_time_s"])
                    burnout_s     = float(run_cfg["target_missile_burnout_time_s"])
                    a_g           = float(run_cfg["a_g"])
                    actual_alt    = float(run_cfg.get("actual_alt_km", target_alt))
                except (KeyError, TypeError, ValueError) as exc:
                    print(f"  [warn] {key}: missing/invalid run_config field ({exc}), skipping")
                    continue

                if not _matches_int(salvo_size, FILTER_SALVO_SIZES):
                    continue
                if not _matches_int(n_int, FILTER_N_INTERCEPTORS_PER_SAT):
                    continue
                if not _matches_float(v_bo, FILTER_V_BO_KM_S):
                    continue
                if not _matches_float(target_alt, FILTER_TARGET_ALTITUDE_KM):
                    continue
                if not _matches_float(intercept_alt, FILTER_INTERCEPT_ALT_KM):
                    continue
                if not _matches_float(detect_s, FILTER_DETECTION_TIME_S):
                    continue
                if not _matches_float(decide_s, FILTER_DECISION_TIME_S):
                    continue
                if not _matches_float(burnout_s, FILTER_BURNOUT_TIME_S):
                    continue
                if not _matches_float(a_g, FILTER_A_G):
                    continue

                obj_bound_raw = data.get("obj_bound", run_cfg.get("obj_bound"))
                mip_gap_raw   = data.get("mip_gap")

                points.append(RunPoint(
                    target_mode            = target_mode,
                    target_label           = _target_label(run_cfg),
                    n_interceptors_per_sat = n_int,
                    target_altitude_km     = target_alt,
                    intercept_alt_km       = intercept_alt,
                    detection_time_s       = detect_s,
                    decision_time_s        = decide_s,
                    burnout_time_s         = burnout_s,
                    a_g                    = a_g,
                    v_bo_km_s              = v_bo,
                    salvo_size             = salvo_size,
                    n_satellites           = int(n_sat_raw),
                    obj_bound              = float(obj_bound_raw) if obj_bound_raw is not None else None,
                    status_str             = str(data.get("status_str", run_cfg.get("result_status", "Unknown"))),
                    mip_gap                = float(mip_gap_raw) if mip_gap_raw is not None else 1.0,
                    run_name               = str(run_name) if run_name is not None else None,
                    actual_alt_km          = actual_alt,
                    s3_key                 = key,
                ))

                if save_local:
                    local_path = LOCAL_RESULTS_ROOT / key
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    with local_path.open("w", encoding="utf-8") as fh:
                        json.dump(data, fh, indent=2)

    return points, n_checked, n_non_principal


def _dedup_best(points: list[RunPoint]) -> list[RunPoint]:
    """If multiple runs share the full (scenario, v_bo, salvo) key, keep the fewest-satellite one."""
    best: dict[tuple, RunPoint] = {}
    for pt in points:
        key = (*pt.scenario_key, pt.v_bo_km_s, pt.salvo_size)
        if key not in best or pt.n_satellites < best[key].n_satellites:
            best[key] = pt
    return list(best.values())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot principal-set (is_principal_set=true) sweep results pulled from S3."
    )
    parser.add_argument(
        "--save-local",
        action="store_true",
        help=f"Mirror downloaded JSONs into {LOCAL_RESULTS_ROOT}/.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the interactive plots (console summary only).",
    )
    args = parser.parse_args()

    active_filters = {
        "salvo_size":             FILTER_SALVO_SIZES,
        "n_interceptors_per_sat": FILTER_N_INTERCEPTORS_PER_SAT,
        "v_bo_km_s":              FILTER_V_BO_KM_S,
        "target_altitude_km":     FILTER_TARGET_ALTITUDE_KM,
        "intercept_alt_km":       FILTER_INTERCEPT_ALT_KM,
        "detection_time_s":       FILTER_DETECTION_TIME_S,
        "decision_time_s":        FILTER_DECISION_TIME_S,
        "burnout_time_s":         FILTER_BURNOUT_TIME_S,
        "a_g":                    FILTER_A_G,
        "target_mode":            FILTER_TARGET_MODE,
        "countries":              FILTER_COUNTRIES,
        "run_name_regex":         FILTER_RUN_NAME_REGEX,
    }
    print(f"Scanning s3://{S3_BUCKET}/{{country,prespecified}}/ for principal-set results ...")
    for name, val in active_filters.items():
        if val:
            print(f"  Filter: {name} = {val}")

    s3 = boto3.client("s3", region_name=AWS_REGION)
    points, n_checked, n_non_principal = _collect_principal_points(s3, save_local=args.save_local)
    print(
        f"  Checked {n_checked} result JSON(s), "
        f"{n_non_principal} were not principal-set, "
        f"{len(points)} matched all filters."
    )
    if not points:
        raise ValueError(
            "No principal-set results matched the current filters. "
            "Check FILTER_* constants at the top of this script, or confirm "
            "is_principal_set=true runs exist in S3."
        )

    points = _dedup_best(points)
    print(f"After dedup: {len(points)} result point(s).\n")

    # ------------------------------------------------------------------
    # Group into one figure per scenario (everything held fixed except
    # v_bo_km_s, which colors series, and salvo_size, which is the x-axis).
    # ------------------------------------------------------------------
    scenario_keys = sorted({pt.scenario_key for pt in points})
    print(f"{len(scenario_keys)} distinct scenario(s) -> {len(scenario_keys)} figure(s).")
    if len(scenario_keys) > 1:
        print(
            "  (More than one scenario survived your filters -- each one gets its own figure "
            "so results that aren't directly comparable never share a plot. Narrow the FILTER_* "
            "constants if you expected only one.)"
        )

    figures: list[plt.Figure] = []

    for scenario_key in scenario_keys:
        pts_fig = [pt for pt in points if pt.scenario_key == scenario_key]
        scenario_label = pts_fig[0].scenario_label

        sorted_vbos = sorted({pt.v_bo_km_s for pt in pts_fig})
        colour_map = {
            vbo: _V_BO_COLOURS[i % len(_V_BO_COLOURS)] for i, vbo in enumerate(sorted_vbos)
        }

        fig, ax = plt.subplots(figsize=(11.0, 6.5))
        ax.set_facecolor("#ffffff")
        hover_points: list[tuple[float, float, str]] = []

        for vbo in sorted_vbos:
            series_pts = sorted(
                [pt for pt in pts_fig if pt.v_bo_km_s == vbo],
                key=lambda pt: pt.salvo_size,
            )
            if not series_pts:
                continue

            colour = colour_map[vbo]
            label = f"v_bo = {vbo:.1f} km/s"

            x = [pt.salvo_size   for pt in series_pts]
            y = [pt.n_satellites for pt in series_pts]
            ax.plot(x, y, marker="o", linewidth=2.0, markersize=6.0, color=colour, label=label)

            limited = [
                pt for pt in series_pts
                if "interrupt" in pt.status_str.lower() or "time" in pt.status_str.lower()
            ]
            if limited:
                ax.scatter(
                    [pt.salvo_size   for pt in limited],
                    [pt.n_satellites for pt in limited],
                    s=60, facecolors="#000000", edgecolors=colour, linewidths=1.8, zorder=4,
                )

            for pt in series_pts:
                bound_str = f"{pt.obj_bound:.0f}" if pt.obj_bound is not None else "-"
                hover_points.append((
                    float(pt.salvo_size),
                    float(pt.n_satellites),
                    (f"{pt.target_label}\n"
                     f"run_name = {pt.run_name}\n"
                     f"v_bo = {pt.v_bo_km_s:.1f} km/s   salvo = {pt.salvo_size}   "
                     f"interceptors/sat = {pt.n_interceptors_per_sat}\n"
                     f"target alt = {pt.target_altitude_km:.0f} km   actual alt = {pt.actual_alt_km:.0f} km   "
                     f"intercept alt = {pt.intercept_alt_km:.0f} km\n"
                     f"detection = {pt.detection_time_s:.0f}s   decision = {pt.decision_time_s:.0f}s   "
                     f"burnout = {pt.burnout_time_s:.0f}s   a_g = {pt.a_g:.1f}\n"
                     f"Satellites = {pt.n_satellites}   Bound <= {bound_str}\n"
                     f"MIP gap = {pt.mip_gap * 100:.1f}%   Status = {pt.status_str}"),
                ))

        ax.set_title(f"Principal set: Salvo Size vs Required Satellites\n{scenario_label}", fontsize=10)
        ax.set_xlabel("Salvo Size")
        ax.set_ylabel("Required Satellites")
        ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.45)
        ax.legend(frameon=True, fontsize=8, title="Burnout velocity")

        annot = ax.annotate(
            "", xy=(0.0, 0.0), xytext=(12, 12), textcoords="offset points",
            bbox={"boxstyle": "round", "fc": "white", "ec": "black", "alpha": 0.95}, fontsize=8,
        )
        annot.set_visible(False)

        hover_xy = (
            np.array([(hx, hy) for hx, hy, _ in hover_points], dtype=np.float64)
            if hover_points else np.empty((0, 2))
        )

        def _on_move(
            event: object, *, _ax=ax, _fig=fig, _ann=annot, _hxy=hover_xy, _hpts=hover_points,
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
    # Console summary tables (raw values), one per scenario.
    # ------------------------------------------------------------------
    print(f"\nS3 source : s3://{S3_BUCKET}/{{country,prespecified}}/\n")
    for scenario_key in scenario_keys:
        pts_group = sorted(
            [pt for pt in points if pt.scenario_key == scenario_key],
            key=lambda pt: (pt.v_bo_km_s, pt.salvo_size),
        )
        if not pts_group:
            continue
        print(f"{'=' * 96}")
        print(f"  {pts_group[0].scenario_label.replace(chr(10), '  |  ')}")
        print(f"{'=' * 96}")
        header = (
            f"  {'v_bo km/s':>10}  {'Salvo':>7}  {'Satellites':>11}  "
            f"{'Bound':>8}  {'MIP gap':>8}  {'Status':<18}  {'run_name'}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))
        for pt in pts_group:
            bound_str = f"{pt.obj_bound:.0f}" if pt.obj_bound is not None else "-"
            mip_pct   = f"{pt.mip_gap * 100:.2f}%"
            print(
                f"  {pt.v_bo_km_s:>10.1f}  {pt.salvo_size:>7}  {pt.n_satellites:>11}  "
                f"{bound_str:>8}  {mip_pct:>8}  {pt.status_str:<18}  {pt.run_name}"
            )
        print()

    show = SHOW_PLOT and not args.no_plot
    if show and figures:
        plt.show()
    elif figures:
        plt.close("all")


if __name__ == "__main__":
    main()
