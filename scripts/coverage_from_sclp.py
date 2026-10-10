"""
coverage_from_sclp.py

Post-optimization coverage analysis of a stored run, with optional orbit plot and coverage video.

Reads the run's CSV + JSON sidecar (local, or fetched from the flat S3 runs/ prefix by name/URI),
reconstructs the orbital elements and all simulation parameters, and runs a coverage simulation --
no layer definitions required.

Usage:
    python scripts/coverage_from_sclp.py                       # newest result in results/
    python scripts/coverage_from_sclp.py <run name | path to _result.csv | s3://.../runs/<name>_result.csv>

Output switches (plot / video, shell mode, overrides) are the module constants below.
NOTE: the video renderer still highlights the ``countries`` stored in the sidecar and does not yet
draw prespecified target points on the map (planned update).
"""
from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt

from sbi_coverage.core.output_report import report
from sbi_coverage.core.plotting import plot_initial_orbits, render_coverage_video
from sbi_coverage.core.shell import Shell
from sbi_coverage.core.simulate import run_simulation
from sbi_coverage.runs.replay import PROPAGATOR, elements_from_orb, load_run, predefined_shell, replay_configs

# ---------------------------------------------------------------------------
# Input: run name, local _result.csv path, or S3 URI/key. None = newest result in RESULTS_DIR.
# A command-line argument overrides this.
# ---------------------------------------------------------------------------
RESULT: str | None = None
RESULTS_DIR: Path = Path(__file__).resolve().parents[1] / "results"   # flat folder; S3 runs are downloaded here

# Country names to highlight/report on (None = use the countries stored in the sidecar).
REPORT_COUNTRIES_OVERRIDE: tuple[str, ...] | None = None

# Simulation overrides (None = exactly the stored settings), e.g. a longer horizon or finer step.
SIM_HORIZON_OVERRIDE_S: float | None = None
SIM_DT_OVERRIDE_S:      float | None = None

# Analysis grid: "predefined" = exactly the stored target points; "full" = dense Fibonacci sphere
# (best for global coverage heatmaps).
SHELL_MODE:            str = "predefined"
SHELL_N_POINTS:        int = 15000
RENDER_SHELL_N_POINTS: int = 15000

# Output flags
PLOT_INITIAL_ORBITS:        bool = False
RENDER_COVERAGE_VIDEO:      bool = False
COVERAGE_VIDEO_OUTPUT_PATH: str | None = None   # None = auto-named in docs/videos/


def main() -> None:
    run = load_run(sys.argv[1] if len(sys.argv) > 1 else RESULT, RESULTS_DIR)
    sc, orb = run["sidecar"], run["orb"]
    n_sats = orb.shape[0]
    rc, res = sc["run_config"], sc["result"]
    r_required = int(rc.get("r_required", 1))

    print(f"Loaded {n_sats} satellites from {run['csv_path'].name}")
    print(f"  Status  : {res.get('status', '?')}")
    print(f"  MIP gap : {(res.get('mip_gap_achieved') or 0.0) * 100:.4f}%")
    print(f"  Salvo   : {rc.get('salvo_size', '?')}  |  {rc.get('n_interceptors_per_sat', '?')} interceptor(s)/sat"
          f"  |  r_required = {r_required} satellites")
    report_countries = (
        [str(c).strip() for c in REPORT_COUNTRIES_OVERRIDE if str(c).strip()] if REPORT_COUNTRIES_OVERRIDE is not None
        else [str(c).strip() for c in sc.get("countries", []) if str(c).strip()]
    )
    print(f"  Countries: {', '.join(report_countries) if report_countries else '(none)'}")

    earth, cov, sim = replay_configs(sc, analysis_dtype="uint16", horizon_s=SIM_HORIZON_OVERRIDE_S, dt_s=SIM_DT_OVERRIDE_S)
    elems0 = elements_from_orb(orb)
    layer_specs = [{"type": "SCLP-selected (RGT)", "n_satellites": n_sats}]

    if SHELL_MODE == "predefined":
        shell = predefined_shell(sc, earth, cov)
    elif SHELL_MODE == "full":
        shell = Shell(earth, n_points=SHELL_N_POINTS, shell_alt_km=cov.intercept_alt_km, analysis_shell_mode="full")
    else:
        raise ValueError("SHELL_MODE must be either 'predefined' or 'full'")

    if PLOT_INITIAL_ORBITS:
        plot_initial_orbits(elems0, earth, layer_specs=layer_specs, frame="ECEF", show_points=True, show_orbits=True)
        plt.show()

    print(f"\nRunning coverage simulation  ({n_sats} sats, horizon={sim.horizon_s / 3600:.2f} h, dt={sim.dt_s:.6f} s) ...")
    t0 = time.perf_counter()
    ao = run_simulation(elems0, sim, cov, earth, shell, layer_specs, propagator=PROPAGATOR)
    report(ao, print_meta=True)
    print(f"run_simulation runtime: {time.perf_counter() - t0:.3f} s")

    if RENDER_COVERAGE_VIDEO:
        if SHELL_MODE == "full":
            ao_video = ao
        else:
            render_shell = Shell(earth, n_points=RENDER_SHELL_N_POINTS, shell_alt_km=cov.intercept_alt_km,
                                 analysis_shell_mode="full")
            print(f"\nRunning full-shell simulation for rendering  ({n_sats} sats, n_points={render_shell.n_points_used}) ...")
            t_render0 = time.perf_counter()
            ao_video = run_simulation(elems0, sim, cov, earth, render_shell, layer_specs, propagator=PROPAGATOR)
            print(f"full-shell render simulation runtime: {time.perf_counter() - t_render0:.3f} s")

        video_path = COVERAGE_VIDEO_OUTPUT_PATH
        if video_path is None:
            video_dir = Path(__file__).resolve().parents[1] / "docs" / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            video_path = str(video_dir / f"coverage_{run['csv_path'].stem.removesuffix('_result')}_{datetime.now():%H%M%S}.mp4")

        # With a denser render grid, pass the optimizer-target analysis as stats_ao so the text overlay
        # reports min/median over the same points report() uses.
        video_summary = render_coverage_video(
            ao_video, stats_countries=report_countries, stats_ao=ao if SHELL_MODE != "full" else None, fps=10,
            output_path=video_path, highlight_roi_borders=True, verbose=True, r_required=r_required,
        )
        print(f"Coverage video: {video_summary['output_path']}")


if __name__ == "__main__":
    main()
