"""Building, validating and submitting many runs at once (the logic behind the GUI's Launch tab).

Everything here is pure Python plus injected boto3 clients, so it is testable without a browser or AWS.
Errors that a user can fix are raised as ``InputProblems`` carrying plain-language guidance.
"""
from __future__ import annotations

import datetime
import itertools
import math
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from ..core.config import CoverageConfig, EarthConstants
from ..core.engagement import interceptor_range_km
from .cloud import S3_CONFIG_PREFIX, sanitize_job_name, upload_config, submit_job
from .config import CONFIG_DEFAULTS, JOB_QUEUE, S3_BUCKET, normalize_config, settings_from_config
from .names import _num, decode_run_name, scenario_code, scenario_from_path, settings_key

MAX_COMBINATIONS = 5000            # hard cap on one submission
CONFIRM_ABOVE = 50                 # ask for explicit confirmation above this many jobs
MAX_LIST_ITEMS = 500


class InputProblems(ValueError):
    """User-fixable input errors. ``problems`` is a list of plain-language messages."""

    def __init__(self, problems: Sequence[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


# --------------------------------------------------------------------------- number lists
_RANGE_RE = re.compile(r"^\s*(?P<a>[\d.]+)\s*(?:-|\.\.|to)\s*(?P<b>[\d.]+)\s*(?:[:/]|step)\s*(?P<s>[\d.]+)\s*$", re.I)


def parse_number_list(text: str, *, name: str, kind: type = float, minimum: float | None = None,
                      maximum: float | None = None, max_items: int = MAX_LIST_ITEMS) -> list:
    """Parse '1, 2, 5' and/or ranges like '10-100:10' into a de-duplicated list.

    Values must be encodable in a run name (non-negative, at most 3 decimals; integers for ``kind=int``).
    Raises InputProblems with an example of the accepted syntax.
    """
    hint = f"Enter {name} as numbers separated by commas, e.g. '5, 10, 20', or a range 'start-end:step', e.g. '10-100:10'."
    if text is None or not str(text).strip():
        raise InputProblems([f"{name}: nothing entered. {hint}"])
    out: list = []
    for tok in re.split(r"[,;\n]+", str(text)):
        tok = tok.strip()
        if not tok:
            continue
        m = _RANGE_RE.match(tok)
        try:
            if m:
                a, b, s = float(m["a"]), float(m["b"]), float(m["s"])
                if s <= 0 or b < a:
                    raise ValueError
                n = int(math.floor((b - a) / s + 1e-9)) + 1
                if n > max_items:
                    raise InputProblems([f"{name}: range '{tok}' expands to {n} values (limit {max_items})."])
                vals = [a + i * s for i in range(n)]
            else:
                vals = [float(tok)]
        except ValueError:
            raise InputProblems([f"{name}: cannot read '{tok}'. {hint}"]) from None
        for v in vals:
            if not math.isfinite(v) or v < 0:
                raise InputProblems([f"{name}: {v} is not allowed (must be a non-negative number)."])
            if minimum is not None and v < minimum or maximum is not None and v > maximum:
                lim = f"between {minimum} and {maximum}" if minimum is not None and maximum is not None else (
                    f">= {minimum}" if minimum is not None else f"<= {maximum}")
                raise InputProblems([f"{name}: {v:g} is out of range; values must be {lim}."])
            if kind is int:
                if abs(v - round(v)) > 1e-9:
                    raise InputProblems([f"{name}: {v:g} must be a whole number."])
                v = int(round(v))
            else:
                try:
                    _num(v, name)
                except ValueError:
                    raise InputProblems([f"{name}: {v!r} has more than 3 decimals, which run names cannot encode."]) from None
                v = round(float(v), 6)
            if v not in out:
                out.append(v)
    if not out:
        raise InputProblems([f"{name}: no values found. {hint}"])
    if len(out) > max_items:
        raise InputProblems([f"{name}: {len(out)} values entered (limit {max_items})."])
    return out


GRID_FIELDS: dict[str, dict[str, Any]] = {
    "salvo_size": dict(label="Salvo size(s)", kind=int, minimum=1, maximum=100000, default="5, 10, 20",
                       help="Number of missiles to intercept at once. Comma list or range, e.g. 10-100:10."),
    "n_interceptors_per_sat": dict(label="Interceptors per satellite", kind=int, minimum=1, maximum=1000, default="1",
                                   help="Interceptors each satellite carries; requirement per target = ceil(salvo / this)."),
    "burnout_velocity_km_s": dict(label="Interceptor burnout velocity (km/s)", kind=float, minimum=0.1, maximum=20.0, default="6",
                                  help="Typical values 4-8 km/s."),
    "detection_time_s": dict(label="Detection time (s)", kind=float, minimum=0, maximum=3600, default="15",
                             help="Seconds from launch until the missile is detected."),
    "decision_time_s": dict(label="Decision time (s)", kind=float, minimum=0, maximum=3600, default="0",
                            help="Seconds from detection until the engage decision."),
    "target_missile_burnout_time_s": dict(label="Target missile burnout time (s)", kind=float, minimum=1, maximum=3600, default="170",
                                          help="Time from launch to burnout; detection + decision must be shorter."),
    "max_accel_g": dict(label="Interceptor max acceleration (g)", kind=float, minimum=0.1, maximum=1000, default="10",
                        help="Boost acceleration in multiples of g0."),
}

ADVANCED_FIELDS: dict[str, dict[str, Any]] = {
    "intercept_alt_km": dict(label="Intercept altitude (km)", kind=float, minimum=0.0, maximum=2000.0,
                             help="Altitude of the target-point shell where interception must occur."),
    "min_elev_deg": dict(label="Min elevation (deg)", kind=float, minimum=-90.0, maximum=90.0, help="Minimum satellite elevation above the target's horizon."),
    "nd_repeat": dict(label="Repeat days N_D", kind=int, minimum=1, maximum=30, help="Shared repeat period in days; 3 is the validated setting."),
    "altitude_min_km": dict(label="Min satellite altitude (km)", kind=float, minimum=100.0, maximum=2000.0,
                            help="Lower altitude bound; the upper bound is intercept altitude + interceptor range."),
    "anchor_lat_margin_deg": dict(label="Anchor latitude margin (deg)", kind=float, minimum=0.0, maximum=30.0,
                                  help="Highest seed inclination = highest target latitude minus this."),
    "raan_step_deg": dict(label="RAAN step (deg)", kind=float, minimum=0.01, maximum=10.0, help="Spacing of seed planes within a family; smaller = more seeds."),
    "dt_target_s": dict(label="Nominal time step (s)", kind=float, minimum=10.0, maximum=600.0, help="Actual step is T_r / L."),
    "margin_multiplier": dict(label="Seed build margin (x requirement)", kind=float, minimum=1.0, maximum=50.0, help="Seeds are added until every target has this multiple of the requirement."),
    "trim_margin_multiplier": dict(label="Seed trim margin (x requirement)", kind=float, minimum=1.0, maximum=50.0, help="Seeds are then trimmed down to this multiple."),
    "max_seeds": dict(label="Max seeds to MILP", kind=int, minimum=1, maximum=5000, help="Safety cap on the optimization size."),
    "mip_gap": dict(label="MIP gap target", kind=float, minimum=0.0, maximum=1.0, help="Stop when the solver proves this relative gap (0.01 = 1%)."),
    "time_limit_s": dict(label="Solver time limit per run (s)", kind=float, minimum=10.0, maximum=172800.0, help="Wall-clock limit for each run's MILP solve."),
}


# --------------------------------------------------------------------------- targets
def validate_scenario(code: str) -> str:
    code = (code or "").strip().upper()
    if not re.fullmatch(r"[A-Z][A-Z0-9]{1,4}", code):
        raise InputProblems(["Scenario code must be 2-5 capital letters/digits starting with a letter (e.g. NK, CN, IR). "
                             "It prefixes every run name."])
    return code


def default_scenario(*, filename: str | None = None, countries: Sequence[str] = ()) -> str:
    """Suggested scenario code from an uploaded file name or the chosen countries."""
    if countries:
        try:
            return scenario_code({"target_country": list(countries) if len(countries) > 1 else countries[0]})
        except ValueError:
            return "CT"
    if filename:
        s = scenario_from_path(filename)
        if s:
            return s
        stem = re.sub(r"[^A-Za-z]", "", filename.rsplit(".", 1)[0]).upper()[:3]
        return stem if len(stem) >= 2 else "PS"
    return "PS"


def target_summary(points: list[list]) -> dict[str, float]:
    lats = [p[1] for p in points]
    return {"n": len(points), "lat_min": min(lats), "lat_max": max(lats), "lat_span": max(lats) - min(lats)}


def country_names() -> list[str] | None:
    """Selectable country names (Natural Earth), or None if the boundary data cannot be loaded."""
    try:
        from ..core import regions
        gdf = regions._load_naturalearth_lowres()
        return sorted(str(n) for n in gdf["NAME"].dropna().unique())
    except Exception:
        return None


# --------------------------------------------------------------------------- grid -> configs
@dataclass
class GridPlan:
    configs: list[dict]
    table: list[dict]
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def build_grid_plan(*, target: dict, grid: dict[str, list], fixed: dict[str, Any], scenario: str, label: str | None,
                    principal: bool, test: bool) -> GridPlan:
    """Expand the grid into normalized run configurations and a preview table.

    ``target`` is {"mode": "prespecified", "points": [...]} or {"mode": "country", "countries": [...],
    "n_points": int}. ``grid`` maps GRID_FIELDS keys to value lists; ``fixed`` holds advanced settings.
    Raises InputProblems if the grid is empty/too large; invalid combinations are reported in
    ``plan.problems`` (and excluded) rather than aborting.
    """
    n = math.prod(len(v) for v in grid.values())
    if n == 0:
        raise InputProblems(["The grid is empty."])
    if n > MAX_COMBINATIONS:
        raise InputProblems([f"{n} combinations exceeds the limit of {MAX_COMBINATIONS} per submission. Narrow some lists or split into several submissions."])
    scenario = validate_scenario(scenario)
    base: dict[str, Any] = {**fixed, "scenario": scenario, "label": label or None,
                            "is_principal_set": bool(principal), "is_test_run": bool(test)}
    if target["mode"] == "prespecified":
        base.update(target_mode="prespecified", target_points=target["points"])
    else:
        base.update(target_mode="country", target_country=list(target["countries"]) if len(target["countries"]) > 1
                    else target["countries"][0], target_shell_n_points=int(target["n_points"]))
    keys = list(grid)
    configs, table, problems, warnings = [], [], [], []
    earth = EarthConstants()
    for combo in itertools.product(*[grid[k] for k in keys]):
        row = dict(zip(keys, combo))
        try:
            cfg = normalize_config({**base, **row})
            CoverageConfig(earth=earth, detection_time_s=cfg["detection_time_s"], decision_time_s=cfg["decision_time_s"],
                           target_missile_burnout_time_s=cfg["target_missile_burnout_time_s"],
                           burnout_velocity_km_s=cfg["burnout_velocity_km_s"], max_accel_g=cfg["max_accel_g"],
                           intercept_alt_km=cfg["intercept_alt_km"], min_elev_deg=cfg["min_elev_deg"])
            t_eng = cfg["target_missile_burnout_time_s"] - cfg["detection_time_s"] - cfg["decision_time_s"]
            rmax = interceptor_range_km(t_eng, cfg["burnout_velocity_km_s"], cfg["max_accel_g"], earth.g0_m_s2)
            settings = settings_from_config(cfg)
            key = settings_key(settings)
        except ValueError as exc:
            problems.append(f"{', '.join(f'{k}={v:g}' for k, v in row.items())}: {exc}")
            continue
        if rmax < 100.0:
            warnings.append(f"Interceptor range is only {rmax:.0f} km for {', '.join(f'{k}={v:g}' for k, v in row.items())}; the run will likely be infeasible.")
        configs.append(cfg)
        table.append({**row, "engagement_time_s": t_eng, "r_max_km": round(rmax, 1),
                      "r_required": int(math.ceil(cfg["salvo_size"] / cfg["n_interceptors_per_sat"])), "identity": key})
    return GridPlan(configs, table, problems, sorted(set(warnings))[:5])


def existing_identities(listing: dict[str, dict]) -> dict[str, list[str]]:
    """identity key -> names of FINISHED runs (result.json present) in an S3 listing (see s3_index.list_run_files)."""
    out: dict[str, list[str]] = {}
    for name, files in listing.items():
        if "result.json" in files:
            try:
                out.setdefault(decode_run_name(name)["key"], []).append(name)
            except ValueError:
                pass
    return out


# --------------------------------------------------------------------------- submit
def submit_runs(configs: list[dict], *, s3, batch, bucket: str = S3_BUCKET, progress: Callable[[int, int], None] | None = None,
                max_workers: int = 6) -> list[dict]:
    """Upload each config and submit one AWS Batch job per config. Returns one result dict per config.

    Failures are captured per job (``status == 'FAILED_TO_SUBMIT'`` with ``error``) so one bad job never aborts the batch.
    """
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"gui_{ts}"
    results: list[dict | None] = [None] * len(configs)
    done = [0]

    def one(i: int) -> None:
        cfg = dict(configs[i])
        cfg["job_name"] = sanitize_job_name(
            f"{tag}_{i:04d}_{(cfg.get('scenario') or 'ps').lower()}_s{cfg['salvo_size']}_i{cfg['n_interceptors_per_sat']}"
            f"_v{cfg['burnout_velocity_km_s']:g}_d{cfg['detection_time_s']:g}")
        res = {"index": i, "job_name": cfg["job_name"], "identity": settings_key(settings_from_config(cfg))}
        try:
            uri = upload_config(s3, cfg, f"{S3_CONFIG_PREFIX}/{tag}/{cfg['job_name']}.json", bucket)
            res.update(job_id=submit_job(batch, job_name=cfg["job_name"], s3_config_uri=uri, bucket=bucket),
                       status="SUBMITTED", config_uri=uri)
        except Exception as exc:
            res.update(status="FAILED_TO_SUBMIT", error=f"{type(exc).__name__}: {exc}"[:300])
        results[i] = res
        done[0] += 1
        if progress:
            progress(done[0], len(configs))

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        list(ex.map(one, range(len(configs))))
    return [r for r in results if r]


# --------------------------------------------------------------------------- AWS status
def check_aws(bucket: str = S3_BUCKET, queue: str = JOB_QUEUE) -> list[str]:
    """Problems (with fix guidance) preventing the GUI from using AWS; empty list = all good."""
    problems: list[str] = []
    try:
        import boto3
        from botocore.exceptions import ClientError, NoCredentialsError, NoRegionError, EndpointConnectionError
    except ImportError:
        return ["boto3 is not installed. Run: pip install -e \".[cloud]\""]
    try:
        boto3.client("sts").get_caller_identity()
    except NoCredentialsError:
        return ["No AWS credentials found. Run `aws configure` (or set AWS_PROFILE / AWS_ACCESS_KEY_ID), then reload this page."]
    except EndpointConnectionError:
        return ["Cannot reach AWS. Check your internet connection / VPN."]
    except (ClientError, NoRegionError) as exc:
        return [f"AWS credentials were rejected: {exc}. Re-run `aws configure` or refresh your SSO login."]
    try:
        boto3.client("s3").head_bucket(Bucket=bucket)
    except ClientError as exc:
        problems.append(f"Cannot access bucket s3://{bucket} ({exc.response.get('Error', {}).get('Code', 'error')}). Results cannot be read.")
    try:
        boto3.client("batch", region_name="us-east-1").list_jobs(jobQueue=queue, maxResults=1)
    except ClientError as exc:
        problems.append(f"Cannot use Batch queue '{queue}' ({exc.response.get('Error', {}).get('Code', 'error')}). Runs cannot be launched or tracked.")
    return problems


JOB_STATUSES = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING", "SUCCEEDED", "FAILED")


def list_recent_jobs(batch, queue: str = JOB_QUEUE, statuses: Iterable[str] = JOB_STATUSES, per_status: int = 200) -> list[dict]:
    """Recent AWS Batch jobs in the queue (newest first), one dict per job."""
    jobs: list[dict] = []
    for st in statuses:
        got = 0
        for page in batch.get_paginator("list_jobs").paginate(jobQueue=queue, jobStatus=st):
            for j in page.get("jobSummaryList", []):
                jobs.append({
                    "job_name": j.get("jobName"), "status": j.get("status"), "job_id": j.get("jobId"),
                    "created": _ms(j.get("createdAt")), "started": _ms(j.get("startedAt")), "stopped": _ms(j.get("stoppedAt")),
                    "reason": j.get("statusReason") or (j.get("container") or {}).get("reason") or "",
                })
                got += 1
            if got >= per_status:
                break
    jobs.sort(key=lambda r: r["created"] or datetime.datetime.min.replace(tzinfo=datetime.timezone.utc), reverse=True)
    return jobs


def _ms(v) -> datetime.datetime | None:
    return datetime.datetime.fromtimestamp(v / 1000.0, tz=datetime.timezone.utc) if v else None
