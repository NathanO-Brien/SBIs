"""
coverage_from_sclp.py

Post-optimization coverage analysis for a constellation selected by lee_test2.py.

Reads the CSV + JSON sidecar produced by lee_test2.py, reconstructs the orbital
elements and all simulation parameters, and runs a coverage simulation with
optional orbit plotting and video generation — no layer definitions required.

Usage:
    python scripts/coverage_from_sclp.py                             # auto-finds newest result
    python scripts/coverage_from_sclp.py results/sclp_result_<ts>.csv
"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sbi_coverage.core.config import EarthConstants, SimConfig, CoverageConfig, coverage_config_from_sidecar
from sbi_coverage.core.elements import OrbitalElements, SatellitePhysical
from sbi_coverage.core.shell import Shell
from sbi_coverage.core.simulate import run_simulation
from sbi_coverage.core.plotting import plot_initial_orbits, render_coverage_video_mercator
from sbi_coverage.core.output_report import report


# ---------------------------------------------------------------------------
# Input — path to the SCLP result CSV produced by lee_test2.py.
# Leave as None to auto-select the newest result in RESULTS_DIR, or pass
# the path as a command-line argument.
# ---------------------------------------------------------------------------
RESULT_CSV: Path | None = Path(r"results\china\burnout-vel-4.0km-s\intercept-window-170s\intercept-alt-200km\max-accel-10.0g\orbit-alt-262km\interceptors-per-sat-1\salvo-size-1\20260612_195842_result.csv")
RESULTS_DIR: Path = Path(__file__).resolve().parents[1] / "results"

# ---------------------------------------------------------------------------
# Optional reporting/rendering countries override
# Leave as None to use the countries stored in the JSON sidecar from lee_test2.py.
# Set to a tuple/list to override those values for this replay.
# ---------------------------------------------------------------------------
REPORT_COUNTRIES_OVERRIDE: tuple[str, ...] | None = None

# ---------------------------------------------------------------------------
# Simulation overrides
# Leave as None to inherit the exact settings from the JSON sidecar (i.e.,
# the same dt and horizon that were used during optimization).  Override to
# run a richer coverage analysis — e.g. a longer horizon or finer timestep.
# ---------------------------------------------------------------------------
SIM_HORIZON_OVERRIDE_S: float | None = None
SIM_DT_OVERRIDE_S:      float | None = None

# ---------------------------------------------------------------------------
# Shell settings
#
# SHELL_MODE controls the analysis grid:
#   "predefined"  — only the exact target points from the JSON (same as the
#                   optimizer shell; gives binary covered/not covered per target)
#   "full"        — dense Fibonacci sphere, best for global coverage heatmaps
#
# ---------------------------------------------------------------------------
SHELL_MODE:            str                     = "predefined"
SHELL_N_POINTS:        int                     = 15000
RENDER_SHELL_N_POINTS: int                     = 15000

# ---------------------------------------------------------------------------
# Output flags
# ---------------------------------------------------------------------------
PLOT_INITIAL_ORBITS:       bool       = False
RENDER_COVERAGE_VIDEO:     bool       = False
COVERAGE_VIDEO_OUTPUT_PATH: str | None = None   # None = auto-named in docs/videos/


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_csv_path() -> Path:
    if RESULT_CSV is not None:
        return Path(RESULT_CSV)
    # Sweep results: results/**/*_result.csv  (nested subdirs, timestamp prefix)
    # Legacy lee_test2 results: results/sclp_result_*.csv  (flat)
    candidates = sorted(
        RESULTS_DIR.rglob("*_result.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"No *_result.csv found under {RESULTS_DIR}. "
            "Run sweep_optimize.py first, or set RESULT_CSV above."
        )
    print(f"Auto-selected newest result: {candidates[0].relative_to(RESULTS_DIR)}")
    return candidates[0]


def _load_orb_csv(csv_path: Path) -> np.ndarray:
    """Return (n_sats, 6) float64: a_km | e | i_deg | raan_deg | argp_deg | arg_lat_deg."""
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            rows.append([
                float(row["a_km"]),
                float(row["e"]),
                float(row["i_deg"]),
                float(row["raan_deg"]),
                float(row["argp_deg"]),
                float(row["arg_lat_deg"]),
            ])
    if not rows:
        raise ValueError(f"CSV contains no data rows: {csv_path}")
    return np.array(rows, dtype=np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv_path  = _resolve_csv_path()
    json_path = csv_path.with_suffix(".json")
    if not json_path.exists():
        raise FileNotFoundError(
            f"JSON sidecar not found: {json_path}\n"
            "Re-run lee_test2.py to regenerate it alongside the CSV."
        )

    with json_path.open() as f:
        cfg = json.load(f)

    orb_arr  = _load_orb_csv(csv_path)
    n_sats   = orb_arr.shape[0]

    rc = cfg.get("run_config", {})
    r_required          = int(rc.get("r_required", 1))
    salvo_size          = int(rc.get("salvo_size", 1))
    n_interceptors_per_sat = int(rc.get("n_interceptors_per_sat", 1))

    print(f"Loaded {n_sats} satellites from {csv_path.name}")
    print(f"  Status  : {cfg.get('status_str', '?')}")
    print(f"  MIP gap : {cfg.get('mip_gap', 0.0) * 100:.4f}%")
    print(f"  Salvo   : {salvo_size} KVs  |  {n_interceptors_per_sat} KV/sat  |  r_required = {r_required} SBIs")
    report_countries = (
        [str(country).strip() for country in REPORT_COUNTRIES_OVERRIDE if str(country).strip()]
        if REPORT_COUNTRIES_OVERRIDE is not None
        else [str(country).strip() for country in cfg.get("countries", []) if str(country).strip()]
    )
    print(f"  Countries: {', '.join(report_countries) if report_countries else '(none)'}")

    # ------------------------------------------------------------------
    # Reconstruct config objects
    # ------------------------------------------------------------------
    earth = EarthConstants()

    cov = coverage_config_from_sidecar(cfg["cov"], earth)

    s = cfg["sim"]
    sim = SimConfig(
        horizon_s=             SIM_HORIZON_OVERRIDE_S if SIM_HORIZON_OVERRIDE_S is not None
                               else float(s["horizon_s"]),
        dt_s=                  SIM_DT_OVERRIDE_S if SIM_DT_OVERRIDE_S is not None
                               else float(s["dt_s"]),
        use_j2=                bool(s.get("use_j2",  False)),
        # use_j2 = True,
        use_drag=              bool(s.get("use_drag", False)),
        analysis_matrix_dtype= str(s.get("analysis_matrix_dtype", "uint8")),
    )

    propagator = str(cfg.get("propagator", "Nominal_Propagator"))
    bc_kg_m2   = float(cfg.get("bc_kg_m2", 80.0))

    # ------------------------------------------------------------------
    # Reconstruct orbital elements
    # arg_lat_deg = argp_deg + M0_deg  =>  M0_deg = arg_lat_deg - argp_deg
    # ------------------------------------------------------------------
    a_km     = orb_arr[:, 0]
    e        = orb_arr[:, 1]
    i_rad    = np.deg2rad(orb_arr[:, 2])
    raan_rad = np.deg2rad(orb_arr[:, 3])
    argp_rad = np.deg2rad(orb_arr[:, 4])
    M0_rad   = np.deg2rad(orb_arr[:, 5] - orb_arr[:, 4])

    elems0 = OrbitalElements(
        a_km=a_km, e=e, i_rad=i_rad,
        raan_rad=raan_rad, argp_rad=argp_rad, M0_rad=M0_rad,
    )
    phys = SatellitePhysical(bc_kg_m2=np.full(n_sats, bc_kg_m2, dtype=np.float64))

    # layer_specs: single entry — all selected satellites belong to one constellation
    layer_specs = [{"type": "SCLP-selected (RGT)", "n_satellites": n_sats}]

    # ------------------------------------------------------------------
    # Shell
    # ------------------------------------------------------------------
    targets = cfg.get("targets", [])

    if SHELL_MODE == "predefined":
        lats   = np.array([t["lat_deg"] for t in targets], dtype=np.float64)
        lons   = np.array([t["lon_deg"] for t in targets], dtype=np.float64)
        labels = [t.get("name", f"Target {i}") for i, t in enumerate(targets)]
        shell  = Shell.from_points(
            earth,
            lat_deg=lats,
            lon_deg=lons,
            shell_alt_km=cov.intercept_alt_km,
            meta_extra={"labels": labels},
        )

    elif SHELL_MODE == "full":
        shell = Shell(
            earth,
            n_points=SHELL_N_POINTS,
            shell_alt_km=cov.intercept_alt_km,
            analysis_shell_mode="full",
        )

    else:
        raise ValueError("SHELL_MODE must be either 'predefined' or 'full'")

    # ------------------------------------------------------------------
    # Orbit plot
    # ------------------------------------------------------------------
    if PLOT_INITIAL_ORBITS:
        plot_initial_orbits(
            elems0,
            earth,
            layer_specs=layer_specs,
            frame="ECEF",
            show_points=True,
            show_orbits=True,
        )
        plt.show()

    # ------------------------------------------------------------------
    # Simulation
    # ------------------------------------------------------------------
    print(
        f"\nRunning coverage simulation  "
        f"({n_sats} sats, horizon={sim.horizon_s / 3600:.2f} h, dt={sim.dt_s:.6f} s) ..."
    )
    t0 = time.perf_counter()
    ao = run_simulation(
        elems0, phys, sim, cov, earth, shell, layer_specs,
        propagator=propagator,
    )
    t1 = time.perf_counter()

    report(ao, print_meta=True)
    print(f"run_simulation runtime: {t1 - t0:.3f} s")

    # ------------------------------------------------------------------
    # Coverage video
    # ------------------------------------------------------------------
    if RENDER_COVERAGE_VIDEO:
        if SHELL_MODE == "full":
            ao_video = ao
        else:
            render_shell = Shell(
                earth,
                n_points=RENDER_SHELL_N_POINTS,
                shell_alt_km=cov.intercept_alt_km,
                analysis_shell_mode="full",
            )
            print(
                f"\nRunning full-shell simulation for rendering  "
                f"({n_sats} sats, n_points={render_shell.n_points_used}, horizon={sim.horizon_s / 3600:.2f} h, dt={sim.dt_s:.6f} s) ..."
            )
            t_render0 = time.perf_counter()
            ao_video = run_simulation(
                elems0, phys, sim, cov, earth, render_shell, layer_specs,
                propagator=propagator,
            )
            t_render1 = time.perf_counter()
            print(f"full-shell render simulation runtime: {t_render1 - t_render0:.3f} s")

        video_path = COVERAGE_VIDEO_OUTPUT_PATH
        if video_path is None:
            ts        = datetime.now().strftime("%Y%m%d_%H%M%S")
            video_dir = Path(__file__).resolve().parents[1] / "docs" / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = str(video_dir / f"coverage_sclp_{ts}.mp4")

        # When rendering with a denser shell than the optimizer used, pass the
        # optimizer-target ao as stats_ao so the text overlay reports min/median
        # over the same points that report() uses — not the denser render grid.
        _stats_ao = ao if (SHELL_MODE != "full") else None
        video_summary = render_coverage_video_mercator(
            ao_video,
            elems0=elems0,
            phys=phys,
            sim=sim,
            earth=earth,
            propagator=propagator,
            stats_countries=report_countries,
            stats_ao=_stats_ao,
            fps=10,
            output_path=video_path,
            highlight_roi_borders=True,
            verbose=True,
            r_required=r_required,
        )
        print(f"Coverage video: {video_summary['output_path']}")


if __name__ == "__main__":
    main()
