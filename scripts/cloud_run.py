"""
cloud_run.py -- submit ONE optimization to AWS Batch.

Builds a run configuration (keys and defaults: sbi_coverage.runs.config), uploads it to S3, and submits an
AWS Batch job that runs scripts/batch_worker.py in the Docker image. Nothing is computed locally.

Targets (prespecified mode only; country mode is deprecated):
    --target-points-csv FILE   CSV with Location/Lat/Lon columns (e.g. data/targets/pacific_blob1.csv)
    --target-points "Name,lat,lon" ...   one or more inline points

Examples:
    python scripts/cloud_run.py --target-points-csv "<...>/NK/Prespecified_10_ICBMS_NK_High_Priority.csv" \\
        --salvo 5 --interceptors 1 --burnout-velocity 6 --detection-time 15 --principal-set
    python scripts/cloud_run.py --target-points-csv targets.csv --salvo 5 --time-limit 120 --test-run
    python scripts/cloud_run.py ... --skip-existing     # do nothing if an equivalent finished run is in S3
    python scripts/cloud_run.py ... --dry-run           # print the run identity and config, submit nothing

Results land flat in s3://sbi-optimization-runs/runs/<run name>_*; browse them with
streamlit run scripts/sbi_gui.py. Tolerances you do not pass use CONFIG_DEFAULTS, except the time
limit (default 4 h here).
"""
from __future__ import annotations

import argparse
import datetime
import json
import sys

from sbi_coverage.core.target_points import load_target_points_csv
from sbi_coverage.runs.cloud import S3_CONFIG_PREFIX, clients, s3_run_exists, sanitize_job_name, submit_job, upload_config
from sbi_coverage.runs.config import CONFIG_DEFAULTS, S3_BUCKET, normalize_config, settings_from_config
from sbi_coverage.runs.names import scenario_from_path, settings_key

DEFAULT_TIME_LIMIT_S = 14400.0   # 4 h


def _parser() -> argparse.ArgumentParser:
    d = CONFIG_DEFAULTS
    p = argparse.ArgumentParser(description="Submit one SBI optimization to AWS Batch.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    t = p.add_argument_group("targets")
    t.add_argument("--target-points-csv", help="CSV of target points (Location/Lat/Lon)")
    t.add_argument("--target-points", nargs="+", metavar="NAME,LAT,LON", help="inline target points")
    t.add_argument("--country", help="DEPRECATED country mode: target a country's shell points")
    t.add_argument("--n-points", type=int, default=d["target_shell_n_points"], help="country mode shell resolution")
    t.add_argument("--scenario", help="run-name scenario code, e.g. NK (inferred from the CSV path/country if omitted)")
    e = p.add_argument_group("engagement")
    e.add_argument("--salvo", type=int, default=1, help="salvo size (missiles to intercept)")
    e.add_argument("--interceptors", type=int, default=1, help="interceptors per satellite")
    e.add_argument("--burnout-velocity", "--vbo", dest="burnout_velocity", type=float, default=4.0, help="km/s")
    e.add_argument("--max-accel", "--a-g", dest="max_accel", type=float, default=d["max_accel_g"], help="g")
    e.add_argument("--detection-time", type=float, default=d["detection_time_s"], help="s")
    e.add_argument("--decision-time", type=float, default=d["decision_time_s"], help="s")
    e.add_argument("--burnout-time", type=float, default=d["target_missile_burnout_time_s"], help="target burnout, s")
    e.add_argument("--intercept-alt", type=float, default=d["intercept_alt_km"], help="km")
    e.add_argument("--min-elev", type=float, default=d["min_elev_deg"], help="deg")
    o = p.add_argument_group("seed families / selection")
    o.add_argument("--nd-repeat", type=int, default=d["nd_repeat"])
    o.add_argument("--altitude-min", type=float, default=d["altitude_min_km"], help="km; upper bound = intercept alt + R_max")
    o.add_argument("--anchor-lat-margin", type=float, default=d["anchor_lat_margin_deg"], help="deg")
    o.add_argument("--raan-step", type=float, default=d["raan_step_deg"], help="deg")
    o.add_argument("--dt", type=float, default=d["dt_target_s"], help="nominal time step, s")
    o.add_argument("--margin-multiplier", type=float, default=d["margin_multiplier"])
    o.add_argument("--trim-margin-multiplier", type=float, default=d["trim_margin_multiplier"])
    o.add_argument("--max-seeds", type=int, default=d["max_seeds"])
    o.add_argument("--seed-workers", type=int, default=d["seed_workers"])
    s = p.add_argument_group("solver (not part of run identity)")
    s.add_argument("--mip-gap", type=float, default=d["mip_gap"])
    s.add_argument("--time-limit", type=float, default=DEFAULT_TIME_LIMIT_S, help="s")
    s.add_argument("--stop-at-gap", type=float, default=None)
    m = p.add_argument_group("bookkeeping")
    m.add_argument("--label", "--run-name", dest="label", default=None, help="free-text label stored in the JSON")
    m.add_argument("--principal-set", action="store_true", help="mark as a principal research set (p in the name)")
    m.add_argument("--test-run", action="store_true", help="TEST- name prefix; hidden by default in the run browser")
    m.add_argument("--job-name", default=None, help="custom AWS Batch job name")
    m.add_argument("--skip-existing", action="store_true", help="skip if an equivalent finished run exists in S3")
    m.add_argument("--dry-run", action="store_true", help="print identity/config and exit")
    return p


def build_config(a: argparse.Namespace) -> dict:
    """Run configuration (runs.config keys) from parsed arguments."""
    cfg: dict = {
        "scenario": a.scenario, "label": a.label, "is_principal_set": a.principal_set, "is_test_run": a.test_run,
        "salvo_size": a.salvo, "n_interceptors_per_sat": a.interceptors, "burnout_velocity_km_s": a.burnout_velocity,
        "max_accel_g": a.max_accel, "detection_time_s": a.detection_time, "decision_time_s": a.decision_time,
        "target_missile_burnout_time_s": a.burnout_time, "intercept_alt_km": a.intercept_alt, "min_elev_deg": a.min_elev,
        "nd_repeat": a.nd_repeat, "altitude_min_km": a.altitude_min, "anchor_lat_margin_deg": a.anchor_lat_margin,
        "raan_step_deg": a.raan_step, "dt_target_s": a.dt, "margin_multiplier": a.margin_multiplier,
        "trim_margin_multiplier": a.trim_margin_multiplier, "max_seeds": a.max_seeds, "seed_workers": a.seed_workers,
        "mip_gap": a.mip_gap, "time_limit_s": a.time_limit, "stop_at_gap": a.stop_at_gap,
    }
    if a.country:
        cfg.update(target_mode="country", target_country=a.country, target_shell_n_points=a.n_points)
    elif a.target_points_csv:
        cfg.update(target_mode="prespecified", target_points=load_target_points_csv(a.target_points_csv))
        cfg["scenario"] = a.scenario or scenario_from_path(a.target_points_csv)
    elif a.target_points:
        pts = []
        for entry in a.target_points:
            parts = entry.split(",")
            if len(parts) != 3:
                raise SystemExit(f"--target-points entries must be 'Name,lat,lon', got: {entry!r}")
            pts.append([parts[0].strip(), float(parts[1]), float(parts[2])])
        cfg.update(target_mode="prespecified", target_points=pts)
    else:
        raise SystemExit("give --target-points-csv or --target-points (or the deprecated --country)")
    return cfg


def main() -> None:
    args = _parser().parse_args()
    cfg = normalize_config(build_config(args))
    key = settings_key(settings_from_config(cfg))
    slug = (cfg["scenario"] or ("country" if cfg["target_mode"] == "country" else "ps")).lower()
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    cfg["job_name"] = sanitize_job_name(args.job_name or f"run_{ts}_{slug}_s{cfg['salvo_size']}_i{cfg['n_interceptors_per_sat']}")

    print(f"Run identity : {key}   (date and sequence letter are added by the worker)")
    print(f"Job name     : {cfg['job_name']}")
    if args.dry_run:
        print(json.dumps({k: v for k, v in cfg.items() if k != "target_points"}, indent=2, default=str))
        print(f"target_points: {len(cfg.get('target_points') or [])} points")
        return

    s3, batch = clients()
    if args.skip_existing:
        existing = s3_run_exists(s3, cfg)
        if existing:
            print(f"SKIP: equivalent finished run(s) already in S3: {', '.join(existing)}")
            return

    config_key = f"{S3_CONFIG_PREFIX}/{cfg['job_name']}.json"
    uri = upload_config(s3, cfg, config_key)
    print(f"Config uploaded to {uri}")
    job_id = submit_job(batch, job_name=cfg["job_name"], s3_config_uri=uri)
    print(f"\nJob submitted.\n  Job ID : {job_id}\n  Status : aws batch describe-jobs --jobs {job_id} --query \"jobs[0].status\"")
    print(f"  Output : s3://{S3_BUCKET}/runs/   (browse: streamlit run scripts/sbi_gui.py)")


if __name__ == "__main__":
    sys.exit(main())
