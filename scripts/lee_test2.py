"""
lee_test2.py

Run ONE local SCLP optimization from the constants below (the quick way to try settings before sending a
run to the cloud). It calls the same pipeline as the AWS worker (sbi_coverage.Lee.run_pipeline.run_sclp),
so a local result and a cloud result with the same settings are directly comparable and get the same
run name (apart from date/sequence).

Orbit model: Lee et al. (2020) exact secular J2. Seed families are N_D = ND_REPEAT synchronized
(N_P, inclination, altitude) tuples that share one repeat period T_r, anchored top-down from the target
latitudes (Lee/seed_families.py), so altitude is an output, not an input.

Output (flat, no subfolders) in RESULTS_DIR, named by sbi_coverage.runs.names, e.g.
TEST-NKp-s005-d15-v6-g10-b170-k0-i1-261009a-6dff_result.csv / _result.json / _gurobi.log

Usage:
    python scripts/lee_test2.py
Browse cloud results with:  streamlit run scripts/sbi_gui.py
"""
from __future__ import annotations

from pathlib import Path

from sbi_coverage.Lee.run_pipeline import run_sclp
from sbi_coverage.core.target_points import load_target_points_csv
from sbi_coverage.runs.names import scenario_from_path

# ---------------------------------------------------------------------------
# Targets: a list of [name, lat_deg, lon_deg] points, or a CSV with a header row containing some
# case-insensitive variant of Location/Lat/Lon (see sbi_coverage.core.target_points).
# Country mode (TARGET_MODE = "country") is deprecated.
# ---------------------------------------------------------------------------
TARGET_MODE: str = "prespecified"
TARGET_POINTS: list[tuple[str, float, float]] = [("NK: Yongbyon (nuclear)", 39.79, 125.66)]
TARGET_POINTS_CSV: str | Path | None = None
TARGET_COUNTRY: str | list[str] = "North Korea"   # country mode only
TARGET_SHELL_N_POINTS: int = 15000                # country mode only

# Run-name metadata. SCENARIO None = inferred from the CSV path (a NK/CN/... folder) or the country.
SCENARIO: str | None = None
LABEL: str | None = None            # free-text note stored in the JSON
IS_PRINCIPAL_SET: bool = False      # lowercase p in the run name
IS_TEST_RUN: bool = True            # TEST- prefix; hidden by default in the run browser

# ---------------------------------------------------------------------------
# Engagement. Timing: DETECTION_TIME_S and DECISION_TIME_S eat into TARGET_MISSILE_BURNOUT_TIME_S before
# any of it is available to the interceptor's own flight (see CoverageConfig).
# ---------------------------------------------------------------------------
SALVO_SIZE: int = 2
N_INTERCEPTORS_PER_SAT: int = 1
BURNOUT_VELOCITY_KM_S: float = 4.0
MAX_ACCEL_G: float = 10.0
DETECTION_TIME_S: float = 0.0
DECISION_TIME_S: float = 0.0
TARGET_MISSILE_BURNOUT_TIME_S: float = 170.0
INTERCEPT_ALT_KM: float = 200.0
MIN_ELEV_DEG: float = 0.0

# ---------------------------------------------------------------------------
# Seed families and seed selection (defaults live in sbi_coverage.runs.config.CONFIG_DEFAULTS)
# ---------------------------------------------------------------------------
ND_REPEAT: int = 3
ALTITUDE_MIN_KM: float = 200.0          # upper bound = INTERCEPT_ALT_KM + R_max (derived, not set here)
ANCHOR_LAT_MARGIN_DEG: float = 0.0      # anchor inclination = highest |target latitude| - margin
RAAN_STEP_DEG: float = 0.25
DT_TARGET_S: float = 120.0              # nominal step; the actual dt = T_r / L
MARGIN_MULTIPLIER: float = 5.0          # greedy build until worst-target access >= this x r_required
TRIM_MARGIN_MULTIPLIER: float = 2.0     # then trim down to this x r_required
MAX_SEEDS: int = 200                    # safety cap on seeds passed to the MILP
SEED_WORKERS: int = 8

# ---------------------------------------------------------------------------
# Solver (not part of run identity; recorded in the JSON)
# ---------------------------------------------------------------------------
MIP_GAP: float = 0.001
TIME_LIMIT_S: float = 45
STOP_AT_GAP: float | None = None        # e.g. 0.999999 stops once any non-trivial bound exists

RESULTS_DIR: Path = Path("results")


def build_config() -> dict:
    """The run configuration for the constants above (keys as in sbi_coverage.runs.config)."""
    cfg: dict = {
        "target_mode": TARGET_MODE, "scenario": SCENARIO, "label": LABEL,
        "is_principal_set": IS_PRINCIPAL_SET, "is_test_run": IS_TEST_RUN,
        "salvo_size": SALVO_SIZE, "n_interceptors_per_sat": N_INTERCEPTORS_PER_SAT,
        "burnout_velocity_km_s": BURNOUT_VELOCITY_KM_S, "max_accel_g": MAX_ACCEL_G,
        "detection_time_s": DETECTION_TIME_S, "decision_time_s": DECISION_TIME_S,
        "target_missile_burnout_time_s": TARGET_MISSILE_BURNOUT_TIME_S,
        "intercept_alt_km": INTERCEPT_ALT_KM, "min_elev_deg": MIN_ELEV_DEG,
        "nd_repeat": ND_REPEAT, "altitude_min_km": ALTITUDE_MIN_KM, "anchor_lat_margin_deg": ANCHOR_LAT_MARGIN_DEG,
        "raan_step_deg": RAAN_STEP_DEG, "dt_target_s": DT_TARGET_S, "margin_multiplier": MARGIN_MULTIPLIER,
        "trim_margin_multiplier": TRIM_MARGIN_MULTIPLIER, "max_seeds": MAX_SEEDS, "seed_workers": SEED_WORKERS,
        "mip_gap": MIP_GAP, "time_limit_s": TIME_LIMIT_S, "stop_at_gap": STOP_AT_GAP,
    }
    if TARGET_MODE.strip().lower() == "prespecified":
        cfg["target_points"] = (load_target_points_csv(TARGET_POINTS_CSV) if TARGET_POINTS_CSV
                                else [list(p) for p in TARGET_POINTS])
        if SCENARIO is None and TARGET_POINTS_CSV:
            cfg["scenario"] = scenario_from_path(TARGET_POINTS_CSV)
    else:
        cfg["target_country"] = TARGET_COUNTRY
        cfg["target_shell_n_points"] = TARGET_SHELL_N_POINTS
    return cfg


def main() -> None:
    summary = run_sclp(build_config(), out_dir=RESULTS_DIR)
    print(f"\nDone: {summary['n_satellites']} satellites, status {summary['status']}  ->  {summary['csv_path']}")


if __name__ == "__main__":
    main()
