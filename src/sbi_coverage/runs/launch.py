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
    ``progress(done, total)`` is always called from the calling thread (never a worker), so UI callbacks are safe.
    """
    from concurrent.futures import as_completed
    ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    tag = f"gui_{ts}"
    results: list[dict | None] = [None] * len(configs)

    def one(i: int) -> dict:
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
        return res

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(one, i): i for i in range(len(configs))}
        done = 0
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
            done += 1
            if progress:
                progress(done, len(configs))
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


def country_targets(countries: Sequence[str], n_points: int, intercept_alt_km: float = 200.0) -> list[list]:
    """Target points ([[name, lat, lon], ...]) that country mode would use: the Fibonacci globe points inside the countries.

    Raises InputProblems when the boundary data is unavailable, a name is unknown, or no point falls inside.
    """
    from ..core.config import EarthConstants
    from ..core.shell import Shell
    if not countries:
        raise InputProblems(["Pick at least one country."])
    try:
        sh = Shell(EarthConstants(), n_points=int(n_points), shell_alt_km=float(intercept_alt_km)).mask_country(list(countries))
    except ImportError:
        raise InputProblems(["Country masking needs geopandas (pip install geopandas shapely pyogrio) and internet access for boundary data."]) from None
    except Exception as exc:
        raise InputProblems([f"Could not mask the globe to {', '.join(countries)}: {exc}"]) from None
    if sh.n_points_used == 0:
        raise InputProblems([f"No globe points fall inside {', '.join(countries)} with {int(n_points)} points. Increase 'Globe points'."])
    label = ", ".join(countries)
    return [[f"{label} target {i}", float(la), float(lo)] for i, (la, lo) in enumerate(zip(sh.lat_deg, sh.lon_deg))]


# =============================================================================== job control
ACTIVE_STATUSES = ("SUBMITTED", "PENDING", "RUNNABLE", "STARTING", "RUNNING")


def terminate_jobs(batch, job_ids: Sequence[str], reason: str = "Cancelled from the SBI GUI", max_workers: int = 6) -> list[dict]:
    """Cancel/terminate AWS Batch jobs (terminate_job handles every state: queued jobs are cancelled, running jobs stopped).

    Returns one dict per job id: {job_id, ok, error}. Failures never abort the rest.
    """
    def one(job_id: str) -> dict:
        try:
            batch.terminate_job(jobId=job_id, reason=reason[:200])
            return {"job_id": job_id, "ok": True, "error": ""}
        except Exception as exc:
            return {"job_id": job_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        return list(ex.map(one, list(job_ids)))


def check_guardrails(*, n_jobs: int, hours: float, active_jobs: int, max_jobs: int, max_hours: float, max_active: int) -> list[str]:
    """Reasons a submission is blocked by the user's safety limits (empty list = allowed)."""
    out = []
    if n_jobs > max_jobs:
        out.append(f"{n_jobs} jobs exceeds your per-submission limit of {max_jobs}. Raise 'Max jobs per submission' in the sidebar (Safety limits) or split the grid.")
    if hours > max_hours:
        out.append(f"Worst-case {hours:,.0f} compute-hours exceeds your limit of {max_hours:,.0f} h. Lower the solver time limit, submit fewer runs, or raise the limit in the sidebar.")
    if active_jobs + n_jobs > max_active:
        out.append(f"{active_jobs} job(s) are already queued/running; adding {n_jobs} would exceed your limit of {max_active} active jobs. "
                   "Wait for jobs to finish, cancel some, or raise 'Max active jobs' in the sidebar.")
    return out


# =============================================================================== clone a past run into the launch form
def _fmt(v) -> str:
    return f"{int(v)}" if float(v).is_integer() else f"{v:g}"


def form_from_sidecar(sidecar: dict[str, Any]) -> dict[str, Any]:
    """Launch-form values that reproduce a stored run (normalized sidecar in; see runs.index.normalize_sidecar).

    Returns {mode, points|countries+n_points, scenario, label, principal, test, grid (field -> text), advanced (field -> value)}.
    Raises InputProblems if the sidecar lacks the information needed (for example a very old file without targets).
    """
    rc = sidecar.get("run_config") or {}
    sf, sv = rc.get("seed_filter") or {}, rc.get("solver") or {}
    problems = []
    targets = sidecar.get("targets") or []
    if not targets:
        problems.append("This run has no stored targets, so it cannot be cloned.")
    need = {k: rc.get(k) for k in ("salvo_size", "n_interceptors_per_sat", "burnout_velocity_km_s")}
    missing = [k for k, v in need.items() if v is None]
    if missing:
        problems.append(f"This run lacks {', '.join(missing)}; it was written by an older version and cannot be cloned.")
    if problems:
        raise InputProblems(problems)
    country_mode = rc.get("target_mode") == "country"
    grid = {k: _fmt(rc[k]) for k in GRID_FIELDS if rc.get(k) is not None}
    adv_src = {
        "intercept_alt_km": rc.get("intercept_alt_km"), "min_elev_deg": rc.get("min_elev_deg"), "nd_repeat": rc.get("nd_repeat"),
        "altitude_min_km": (rc.get("altitude_bounds_km") or [None])[0], "anchor_lat_margin_deg": rc.get("anchor_lat_margin_deg"),
        "raan_step_deg": sf.get("raan_step_deg"), "dt_target_s": rc.get("dt_target_s"),
        "margin_multiplier": sf.get("margin_multiplier"), "trim_margin_multiplier": sf.get("trim_margin_multiplier"),
        "max_seeds": sf.get("max_seeds"), "mip_gap": sv.get("mip_gap_target"), "time_limit_s": sv.get("time_limit_s"),
    }
    out: dict[str, Any] = {
        "mode": "country" if country_mode else "prespecified",
        "scenario": rc.get("scenario") or "PS", "label": f"clone of {sidecar.get('run_name') or 'run'}",
        "principal": bool(rc.get("is_principal_set")), "test": bool(rc.get("is_test_run", True)),
        "grid": grid, "advanced": {k: v for k, v in adv_src.items() if v is not None},
    }
    if country_mode:
        c = sidecar.get("countries") or ([rc["target_country"]] if isinstance(rc.get("target_country"), str) else rc.get("target_country") or [])
        out.update(countries=list(c), n_points=int(rc.get("target_shell_n_points") or 15000))
    else:
        out["points"] = [[t.get("name", ""), t["lat_deg"], t["lon_deg"]] for t in targets]
    return out


def points_to_csv_text(points: Sequence[Sequence]) -> str:
    """CSV text (Name,Lat,Lon) for the paste box; names containing commas are quoted."""
    import csv
    import io
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["Name", "Lat", "Lon"])
    for p in points:
        w.writerow([p[0], f"{p[1]:.6f}", f"{p[2]:.6f}"])
    return buf.getvalue()


# =============================================================================== pre-launch orbit plan preview
def preview_orbit_plan(points: Sequence[Sequence], configs: Sequence[dict]) -> list[dict]:
    """Run the (solver-free) seed-family planner for the smallest and largest interceptor range in ``configs``.

    The planner decides the anchor inclination from the highest target latitude, the shared-repeat-period family
    ladder, the altitudes and the number of seeds. Returns one dict per case: {case, r_max_km, families (list of dicts),
    n_seeds, anchor_alt_km, anchor_inc_deg, alt_bounds_km, L, T_r_s, est_profile_mb, warnings, error}.
    """
    from ..Lee.seed_families import plan_seed_families
    earth = EarthConstants()
    scored = []
    for c in configs:
        cov = CoverageConfig(earth=earth, detection_time_s=c["detection_time_s"], decision_time_s=c["decision_time_s"],
                             target_missile_burnout_time_s=c["target_missile_burnout_time_s"], burnout_velocity_km_s=c["burnout_velocity_km_s"],
                             max_accel_g=c["max_accel_g"], intercept_alt_km=c["intercept_alt_km"], min_elev_deg=c["min_elev_deg"])
        scored.append((cov.max_range_km, c, cov))
    if not scored:
        return []
    scored.sort(key=lambda t: t[0])
    cases = [("shortest interceptor range", scored[0])]
    if scored[-1][0] - scored[0][0] > 1.0:
        cases.append(("longest interceptor range", scored[-1]))
    out = []
    for label, (rmax, c, cov) in cases:
        res: dict[str, Any] = {"case": label, "r_max_km": float(rmax), "families": [], "warnings": [], "error": None}
        try:
            plan = plan_seed_families(
                [list(p) for p in points], earth=earth, N_D=c["nd_repeat"], alt_min_km=c["altitude_min_km"],
                anchor_lat_margin_deg=c["anchor_lat_margin_deg"], useful_reach_deg=c["useful_reach_deg"], cov=cov,
                allow_reducible=c["allow_reducible"], dt_target_s=c["dt_target_s"])
        except Exception as exc:
            res["error"] = str(exc)
            res["warnings"].append("No orbit family can serve these targets with this interceptor range; the run would fail. "
                                   "Increase the interceptor range (burnout velocity, acceleration or timing) or revise the targets.")
            out.append(res)
            continue
        fams = plan["families"]
        n_raan = [int(math.ceil(f["raan_interval_deg"] / c["raan_step_deg"] - 1e-9)) for f in fams]
        n_seeds = int(sum(n_raan))
        res.update(
            families=[{"N_P:N_D": f"{f['N_P']}:{f['N_D']}", "inclination_deg": round(f["inc_deg"], 2), "altitude_km": round(f["alt_km"], 1),
                       "reach_deg": round(f["reach_deg"], 2), "seeds": n} for f, n in zip(fams, n_raan)],
            n_seeds=n_seeds, anchor_alt_km=float(plan["anchor"]["alt_km"]), anchor_inc_deg=float(plan["anchor"]["inc_deg"]),
            alt_bounds_km=[float(x) for x in plan["alt_bounds_km"]], L=int(plan["L"]), T_r_s=float(plan["T_r_s"]),
            est_profile_mb=n_seeds * int(plan["L"]) * len(points) / 1e6,
        )
        if len(fams) == 1:
            res["warnings"].append("Only one orbit family reaches these targets: little diversity for the optimizer, expect more satellites.")
        if res["est_profile_mb"] > 20000:
            res["warnings"].append(f"Seed simulation would hold about {res['est_profile_mb'] / 1000:.0f} GB of profiles; the run may run out of memory. "
                                   "Use fewer targets, a larger RAAN step, or fewer globe points.")
        if n_seeds > 20000:
            res["warnings"].append(f"{n_seeds} seeds is very large; seed simulation will be slow. Consider a larger RAAN step.")
        out.append(res)
    return out
