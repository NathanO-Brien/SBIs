"""
cloud_coverage_from_sclp.py

Post-optimization coverage analysis for a constellation stored in S3.

Mirrors coverage_from_sclp.py exactly — reconstructs orbital elements and
simulation parameters from an S3 result, then runs a coverage simulation with
optional orbit plotting and video generation.

Files are cached locally in tmp/s3_cache/ so repeated runs skip re-downloading.

Usage:
    # Hardcode RESULT_S3_PREFIX below, then:
    python scripts/cloud_coverage_from_sclp.py

    # Or pass the S3 key prefix on the command line:
    python scripts/cloud_coverage_from_sclp.py --prefix "country/china/.../salvo-size-10/sweep_20260623_041454_china_alt400km_s10_i1_v10"
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import boto3

from sbi_coverage.core.config import EarthConstants, SimConfig, CoverageConfig
from sbi_coverage.core.elements import OrbitalElements, SatellitePhysical
from sbi_coverage.core.shell import Shell
from sbi_coverage.core.simulate import run_simulation
from sbi_coverage.core.plotting import plot_initial_orbits, render_coverage_video_mercator
from sbi_coverage.core.output_report import report


# ---------------------------------------------------------------------------
# AWS / S3 configuration
# ---------------------------------------------------------------------------
S3_BUCKET:  str = "sbi-optimization-runs"
AWS_REGION: str = "us-east-1"

# ---------------------------------------------------------------------------
# Input — S3 key prefix for the result, without file extension.
# Example:
#   "country/china/target-points-273/intercept-alt-200km/orbit-alt-404km/
#    burnout-vel-10.0km-s/intercept-window-170s/max-accel-10.0g/
#    interceptors-per-sat-1/doctrine-1x/salvo-size-10/
#    sweep_20260623_041454_china_alt400km_s10_i1_v10"
#
# Pass via --prefix CLI arg or hardcode here.
# ---------------------------------------------------------------------------
RESULT_S3_PREFIX: str | None = None

# Local cache directory — files are downloaded here and reused on repeat runs.
CACHE_DIR: Path = Path(__file__).resolve().parents[1] / "tmp" / "s3_cache"

# ---------------------------------------------------------------------------
# Optional reporting/rendering countries override.
# None → use the countries stored in the JSON sidecar.
# ---------------------------------------------------------------------------
REPORT_COUNTRIES_OVERRIDE: tuple[str, ...] | None = None

# ---------------------------------------------------------------------------
# Simulation overrides.
# None → inherit exact settings from the JSON sidecar.
# ---------------------------------------------------------------------------
SIM_HORIZON_OVERRIDE_S: float | None = None
SIM_DT_OVERRIDE_S:      float | None = None

# ---------------------------------------------------------------------------
# Shell settings
#   "predefined" — exact target points from the JSON (same as optimizer shell)
#   "full"       — dense Fibonacci sphere, best for global coverage heatmaps
# ---------------------------------------------------------------------------
SHELL_MODE:            str = "predefined"
SHELL_N_POINTS:        int = 15000
RENDER_SHELL_N_POINTS: int = 15000

# ---------------------------------------------------------------------------
# Output flags
# ---------------------------------------------------------------------------
PLOT_INITIAL_ORBITS:        bool      = False
RENDER_COVERAGE_VIDEO:      bool      = False
COVERAGE_VIDEO_OUTPUT_PATH: str | None = None   # None = auto-named in docs/videos/


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _cache_path(prefix: str, suffix: str) -> Path:
    """Return a local path for caching an S3 object.

    Uses a short hash + the job name to stay well under Windows MAX_PATH (260 chars).
    """
    import hashlib
    job_name  = prefix.rstrip("/").split("/")[-1]
    short_hash = hashlib.md5(prefix.encode()).hexdigest()[:8]
    return CACHE_DIR / f"{short_hash}_{job_name}{suffix}"


def _download_if_needed(s3, key: str, local: Path) -> Path:
    """Download key from S3 to local path unless the file already exists."""
    local.parent.mkdir(parents=True, exist_ok=True)
    if local.exists():
        print(f"  (cached) {local.name}")
        return local
    print(f"  Downloading s3://{S3_BUCKET}/{key} ...")
    s3.download_file(S3_BUCKET, key, str(local))
    return local


def _fetch_result(s3, prefix: str) -> tuple[Path, Path]:
    """Download .csv and .json for the given prefix. Returns (csv_path, json_path)."""
    job_name = prefix.rstrip("/").split("/")[-1]
    csv_key  = f"{prefix}.csv"
    json_key = f"{prefix}.json"

    csv_local  = _cache_path(prefix, ".csv")
    json_local = _cache_path(prefix, ".json")

    _download_if_needed(s3, csv_key,  csv_local)
    _download_if_needed(s3, json_key, json_local)

    return csv_local, json_local


# ---------------------------------------------------------------------------
# CSV loader (identical to coverage_from_sclp.py)
# ---------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(
        description="Run coverage analysis on an S3 SCLP result."
    )
    parser.add_argument(
        "--prefix",
        default=None,
        metavar="S3_KEY_PREFIX",
        help=(
            "S3 key prefix for the result, without file extension. "
            "Falls back to RESULT_S3_PREFIX constant if omitted."
        ),
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-download files even if cached locally.",
    )
    args = parser.parse_args()

    prefix = args.prefix or RESULT_S3_PREFIX
    if prefix is None:
        raise ValueError(
            "No S3 prefix specified. Pass --prefix or set RESULT_S3_PREFIX in the script."
        )
    prefix = prefix.rstrip("/")

    s3 = boto3.client("s3", region_name=AWS_REGION)

    # Optionally bust the cache
    if args.no_cache:
        for suffix in (".csv", ".json"):
            p = _cache_path(prefix, suffix)
            if p.exists():
                p.unlink()

    print(f"Fetching result from s3://{S3_BUCKET}/{prefix}.*")
    csv_path, json_path = _fetch_result(s3, prefix)

    with json_path.open() as f:
        cfg = json.load(f)

    orb_arr = _load_orb_csv(csv_path)
    n_sats  = orb_arr.shape[0]

    rc = cfg.get("run_config", {})
    r_required             = int(rc.get("r_required", 1))
    salvo_size             = int(rc.get("salvo_size", 1))
    n_interceptors_per_sat = int(rc.get("n_interceptors_per_sat", 1))

    print(f"\nLoaded {n_sats} satellites from {csv_path.name}")
    print(f"  Status  : {cfg.get('status_str', '?')}")
    print(f"  MIP gap : {cfg.get('mip_gap', 0.0) * 100:.4f}%")
    print(f"  Salvo   : {salvo_size} KVs  |  {n_interceptors_per_sat} KV/sat  |  r_required = {r_required} SBIs")

    report_countries = (
        [str(c).strip() for c in REPORT_COUNTRIES_OVERRIDE if str(c).strip()]
        if REPORT_COUNTRIES_OVERRIDE is not None
        else [str(c).strip() for c in cfg.get("countries", []) if str(c).strip()]
    )
    print(f"  Countries: {', '.join(report_countries) if report_countries else '(none)'}")

    # ------------------------------------------------------------------
    # Reconstruct config objects
    # ------------------------------------------------------------------
    earth = EarthConstants()

    c = cfg["cov"]
    cov = CoverageConfig(
        earth=earth,
        T_window_s=       float(c["T_window_s"]),
        v_bo_km_s=        float(c["v_bo_km_s"]),
        a_g=              float(c["a_g"]),
        intercept_alt_km= float(c["intercept_alt_km"]),
        min_elev_deg=     float(c["min_elev_deg"]),
    )

    s = cfg["sim"]
    sim = SimConfig(
        horizon_s=             SIM_HORIZON_OVERRIDE_S if SIM_HORIZON_OVERRIDE_S is not None
                               else float(s["horizon_s"]),
        dt_s=                  SIM_DT_OVERRIDE_S if SIM_DT_OVERRIDE_S is not None
                               else float(s["dt_s"]),
        use_j2=                bool(s.get("use_j2",  False)),
        use_drag=              bool(s.get("use_drag", False)),
        # Force uint16 regardless of what the original optimizer run used.
        # The stored value is often "uint8" (max 255), fine for single-satellite
        # seed sims but silently overflows when re-simulating all selected
        # satellites together if any point is covered by >255 at once.
        analysis_matrix_dtype= "uint16",
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
        raise ValueError("SHELL_MODE must be 'predefined' or 'full'")

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
                f"({n_sats} sats, n_points={render_shell.n_points_used}, "
                f"horizon={sim.horizon_s / 3600:.2f} h, dt={sim.dt_s:.6f} s) ..."
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
