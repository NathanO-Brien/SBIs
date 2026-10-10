"""SBI optimizer GUI: launch many runs on AWS, watch the jobs, and pull/plot/export results.

    streamlit run scripts/sbi_gui.py

The GUI works only with the cloud: runs are launched as AWS Batch jobs and results are read straight from
s3://sbi-optimization-runs/runs/. It never reads your local results folder (exports are written to a folder
you choose, default ~/Downloads/SBI_exports). Needs AWS credentials (`aws configure`).

Tabs
----
 Launch runs   targets (CSV upload / paste, or countries + globe points) -> labels -> grid of settings -> review;
               a sticky side panel shows the targets on an interactive 3-D ECI globe and the run summary/submit
 Jobs          live status of the Batch queue and of what you submitted in this session
 Results       filter the runs in S3, inspect one run, compare many runs, export CSV/JSON; the side panel shows
               the selected run's targets on the globe
"""
from __future__ import annotations

import datetime
import sys
import traceback
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi_coverage.core.target_points import TargetFileError, parse_target_points_text  # noqa: E402
from sbi_coverage.runs import export as exp  # noqa: E402
from sbi_coverage.runs import globe  # noqa: E402
from sbi_coverage.runs import launch as lc  # noqa: E402
from sbi_coverage.runs import s3_index  # noqa: E402
from sbi_coverage.runs.config import CONFIG_DEFAULTS, JOB_QUEUE, S3_BUCKET, S3_RUNS_PREFIX  # noqa: E402

st.set_page_config(page_title="SBI optimizer", page_icon="🛰️", layout="wide", initial_sidebar_state="expanded")

DEFAULT_TIME_LIMIT_S = 14400.0
FILTER_COLS = {
    "scenario": "Scenario", "salvo_size": "Salvo", "detection_time_s": "Detection [s]", "decision_time_s": "Decision [s]",
    "burnout_velocity_km_s": "Burnout vel [km/s]", "max_accel_g": "Max accel [g]",
    "target_missile_burnout_time_s": "Burnout time [s]", "n_interceptors_per_sat": "Interceptors/sat",
}
CURVE_BY = {k: v for k, v in FILTER_COLS.items() if k not in ("scenario", "salvo_size")}
AXES = {**{k: v for k, v in FILTER_COLS.items() if k != "scenario"},
        "n_satellites": "Satellites required", "mip_gap_achieved": "MIP gap", "time_optimization_s": "Solve time [s]",
        "anchor_alt_km": "Anchor altitude [km]", "n_targets": "Targets", "r_required": "r required"}
TABLE_COLS = ["name", "date_iso", "scenario", "principal", "test", "salvo_size", "detection_time_s", "decision_time_s",
              "burnout_velocity_km_s", "max_accel_g", "target_missile_burnout_time_s", "n_interceptors_per_sat",
              "n_satellites", "result_status", "mip_gap_achieved", "anchor_alt_km", "n_targets", "families", "hash"]
STATUS_ICON = {"SUCCEEDED": "🟢", "RUNNING": "🔵", "STARTING": "🟡", "RUNNABLE": "🟡", "SUBMITTED": "⚪", "PENDING": "⚪", "FAILED": "🔴"}

# =============================================================================== look and feel
CSS = """
<style>
html, body, [class*="css"], .stMarkdown, button, input, textarea {
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif; }
.block-container { padding-top: 1.2rem; padding-bottom: 3rem; max-width: 1600px; }
[data-testid="stAppDeployButton"] { display: none; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }
.hero { display: flex; align-items: baseline; gap: .75rem; margin: 0 0 .6rem 0; }
.hero .t { font-size: 1.7rem; font-weight: 700; }
.hero .s { opacity: .62; font-size: .98rem; }
.step { display: flex; align-items: center; gap: .6rem; margin: .1rem 0 .1rem 0; }
.step .badge { width: 1.55rem; height: 1.55rem; border-radius: 50%; background: #2f6fed; color: #fff; display: inline-flex;
  align-items: center; justify-content: center; font-size: .8rem; font-weight: 700; flex: none; }
.step .title { font-size: 1.08rem; font-weight: 650; }
.step-cap { opacity: .62; font-size: .86rem; margin: 0 0 .7rem 2.15rem; line-height: 1.35; }
.panel-title { font-size: .78rem; font-weight: 650; letter-spacing: .06em; text-transform: uppercase; opacity: .65; margin-bottom: .3rem; }
div[data-testid="stMetric"] { border: 1px solid rgba(128,128,128,.28); border-radius: 12px; padding: .55rem .8rem; }
div[data-testid="stMetricLabel"] p { font-size: .72rem; opacity: .7; text-transform: uppercase; letter-spacing: .05em; }
div[data-testid="stMetricValue"] { font-size: 1.45rem; }
.stTabs [data-baseweb="tab-list"] { gap: .25rem; }
.stTabs [data-baseweb="tab"] { font-size: .98rem; padding: .55rem 1rem; }
div[data-testid="stColumn"]:has(.sticky-marker) { position: sticky; top: 3.2rem; align-self: flex-start; }
.pill { display: inline-block; padding: .12rem .6rem; border-radius: 999px; font-size: .78rem; font-weight: 600; margin-right: .35rem; }
.pill.ok { background: rgba(34,170,90,.16); color: #1f9d55; } .pill.bad { background: rgba(220,60,60,.15); color: #d64545; }
.pill.warn { background: rgba(230,160,20,.18); color: #b7791f; } .pill.mute { background: rgba(128,128,128,.18); }
.check { font-size: .9rem; line-height: 1.7; }
button[kind="primary"] { border-radius: 10px; font-weight: 650; }
div[data-testid="stDataFrame"] { border-radius: 10px; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def step_header(n: int, title: str, caption: str = "") -> None:
    st.markdown(f'<div class="step"><span class="badge">{n}</span><span class="title">{title}</span></div>', unsafe_allow_html=True)
    if caption:
        st.markdown(f'<div class="step-cap">{caption}</div>', unsafe_allow_html=True)


def panel_title(text: str) -> None:
    st.markdown(f'<div class="panel-title">{text}</div>', unsafe_allow_html=True)


def pill(text: str, kind: str = "mute") -> str:
    return f'<span class="pill {kind}">{text}</span>'


@contextmanager
def guard(what: str):
    """Show any unexpected error as a friendly message (details in an expander) instead of crashing the page."""
    try:
        yield
    except lc.InputProblems as exc:
        for p in exc.problems:
            st.error(p)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Something went wrong while {what}: {type(exc).__name__}: {exc}")
        with st.expander("Technical details"):
            st.code(traceback.format_exc())


# =============================================================================== cached cloud access
@st.cache_resource
def _clients():
    import boto3
    return boto3.client("s3"), boto3.client("batch", region_name="us-east-1")


@st.cache_data(ttl=60, show_spinner=False)
def _aws_problems() -> list[str]:
    return lc.check_aws()


@st.cache_data(ttl=300, show_spinner="Reading runs from S3 ...")
def _load_index(token: int):
    s3, _ = _clients()
    listing = s3_index.list_run_files(s3)
    return s3_index.build_index(s3, listing=listing), lc.existing_identities(listing)


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_sidecar(key: str):
    s3, _ = _clients()
    return s3_index.normalize_sidecar(s3_index.read_json(s3, S3_BUCKET, key))


@st.cache_data(ttl=30, show_spinner="Reading the Batch queue ...")
def _jobs(statuses: tuple[str, ...], token: float):
    _, batch = _clients()
    return lc.list_recent_jobs(batch, statuses=statuses)


@st.cache_data(ttl=3600, show_spinner=False)
def _countries():
    return lc.country_names()


@st.cache_data(ttl=3600, show_spinner="Placing globe points inside the selected countries ...")
def _country_points(countries: tuple[str, ...], n_points: int, alt_km: float):
    return lc.country_targets(countries, n_points, alt_km)


@st.cache_resource(show_spinner=False)
def _borders():
    return globe.load_borders()


def show_globe(points, *, alt_km: float, highlight=(), height: int = 360, title: str | None = None, key: str = "globe") -> None:
    """Interactive 3-D ECI globe with the targets (or just the globe when there are none)."""
    borders = _borders()
    fig = globe.build_globe_figure(points, alt_km=alt_km, highlight_countries=highlight, borders=borders or {}, height=height, title=title)
    st.plotly_chart(fig, width="stretch", key=key, config={"displayModeBar": False, "scrollZoom": True})
    if borders is None:
        st.caption("Country borders unavailable (offline or geopandas missing).")


def export_buttons(label: str, data: bytes, filename: str, mime: str, key: str) -> None:
    """Download button plus a 'save to export folder' button for the same data."""
    c1, c2 = st.columns(2)
    c1.download_button(f"Download {label}", data, filename, mime, key=f"dl_{key}", icon=":material/download:", width="stretch")
    if c2.button(f"Save {label} to export folder", key=f"sv_{key}", icon=":material/folder_open:", width="stretch"):
        folder, err = exp.validate_export_dir(st.session_state.get("export_dir", str(exp.DEFAULT_EXPORT_DIR)))
        if err:
            st.error(err)
        else:
            st.success(f"Saved: {exp.save_export(folder, filename, data)}")


# =============================================================================== sidebar and AWS gate
with st.sidebar:
    st.markdown("### 🛰️ SBI optimizer")
    aws_problems = _aws_problems()
    if aws_problems:
        st.markdown(pill("AWS not ready", "bad"), unsafe_allow_html=True)
    else:
        st.markdown(pill("AWS connected", "ok"), unsafe_allow_html=True)
    st.caption(f"**Results** `s3://{S3_BUCKET}/{S3_RUNS_PREFIX}`  \n**Batch queue** `{JOB_QUEUE}`")
    st.text_input("Export folder", value=str(exp.DEFAULT_EXPORT_DIR), key="export_dir",
                  help="'Save to export folder' buttons write CSV/JSON here. The browser 'Download' buttons use your normal download folder.")
    if st.button("Re-check AWS connection", icon=":material/refresh:", width="stretch"):
        _aws_problems.clear()
        st.rerun()
    st.divider()
    st.caption("Cloud only: this app never reads a local results folder.")

st.markdown('<div class="hero"><span class="t">SBI constellation optimizer</span>'
            '<span class="s">Launch, monitor and compare space-based-interceptor runs on AWS</span></div>', unsafe_allow_html=True)
if aws_problems:
    for p in aws_problems:
        st.error(p)
    st.info("Fix the problem above, then use 'Re-check AWS connection' in the sidebar.")
    st.stop()

tab_launch, tab_jobs, tab_results = st.tabs([":material/rocket_launch: Launch runs", ":material/monitoring: Jobs", ":material/query_stats: Results"])


# =============================================================================== LAUNCH
def render_launch() -> None:
    left, right = st.columns([3, 2], gap="large")
    state: dict = {"points": None, "target": None, "filename": None, "countries": [], "highlight": (), "grid_errors": False}
    intercept_alt_default = float(CONFIG_DEFAULTS["intercept_alt_km"])
    plan, cfgs, tbl, dup = None, [], pd.DataFrame(), pd.Series(dtype=bool)
    fixed: dict = {}

    with left:
        # ---- 1. targets -------------------------------------------------------------------------
        with st.container(border=True):
            step_header(1, "Targets", "Where the missiles launch from: the points every salvo must be able to intercept.")
            mode = st.radio("Target mode", ["Prespecified points", "Country mode (deprecated)"], horizontal=True, label_visibility="collapsed",
                            help="Prespecified: exactly the points you upload. Country: every globe point inside the chosen country(ies).")
            if mode.startswith("Prespecified"):
                up = st.file_uploader("Upload a CSV with Name, Lat, Lon columns", type=["csv", "txt"], key="target_upload",
                                      help="One target per row; header like Name,Lat,Lon (decimal degrees, +north/+east). Comma, semicolon or tab separated.")
                with st.expander("...or paste CSV text"):
                    pasted = st.text_area("CSV text", height=110, key="target_paste", label_visibility="collapsed",
                                          placeholder="Name,Lat,Lon\nSohae,39.66,124.70\nYongbyon,39.79,125.75")
                text = source = None
                if up is not None:
                    try:
                        text, source, state["filename"] = up.getvalue().decode("utf-8-sig"), up.name, up.name
                    except UnicodeDecodeError:
                        st.error("The file is not UTF-8 text. In Excel use 'Save As > CSV UTF-8'.")
                elif pasted.strip():
                    text, source = pasted, "pasted text"
                if text is not None:
                    try:
                        points, warns = parse_target_points_text(text, source=source)
                        state["points"] = points
                        state["target"] = {"mode": "prespecified", "points": points}
                        s = lc.target_summary(points)
                        st.markdown(pill(f"{s['n']} targets", "ok") + pill(f"lat {s['lat_min']:.1f}° to {s['lat_max']:.1f}°", "mute")
                                    + pill(f"span {s['lat_span']:.1f}°", "warn" if s["lat_span"] > 15 else "mute"), unsafe_allow_html=True)
                        for w in warns:
                            st.warning(w)
                        if s["lat_span"] > 15:
                            st.warning("Targets span more than 15° of latitude: few synchronized orbit families can reach all of them, so the satellite "
                                       "count may be high. Consider splitting the set by latitude band.")
                    except TargetFileError as exc:
                        st.error(f"The target file cannot be used ({source}):")
                        for p in exc.problems:
                            st.markdown(f"- {p}")
                else:
                    st.info("Upload or paste a target list; it will appear on the globe at right.", icon=":material/upload_file:")
            else:
                st.warning("Country mode is deprecated: it covers every globe point inside the country, usually far more than the strategic sites. "
                           "Prefer prespecified points.")
                names = _countries()
                if names:
                    countries_sel = st.multiselect("Countries", names, help="Names from Natural Earth; several are combined into one target set.")
                else:
                    st.caption("Country list unavailable (offline or geopandas missing); type names instead.")
                    countries_sel = [c.strip() for c in st.text_input("Countries (comma separated)").split(",") if c.strip()]
                n_points = int(st.number_input("Globe points", min_value=1000, max_value=200000, value=15000, step=1000,
                                               help="Size of the Fibonacci sphere sampled before masking to the country. More points = finer coverage "
                                                    "but a much larger optimization; the points that land inside the country are the targets."))
                state["countries"] = countries_sel
                state["highlight"] = tuple(countries_sel)
                if countries_sel:
                    try:
                        pts = _country_points(tuple(countries_sel), n_points, intercept_alt_default)
                        state["points"] = pts
                        state["target"] = {"mode": "country", "countries": countries_sel, "n_points": n_points}
                        s = lc.target_summary(pts)
                        st.markdown(pill(f"{s['n']} target points", "warn" if s["n"] > 500 else "ok") + pill(f"lat {s['lat_min']:.1f}° to {s['lat_max']:.1f}°", "mute"),
                                    unsafe_allow_html=True)
                        if s["n"] > 500:
                            st.warning("More than 500 target points: the optimization will be large and slow. Use fewer globe points.")
                    except lc.InputProblems as exc:
                        for p in exc.problems:
                            st.error(p)
                else:
                    st.info("Pick at least one country; the target points appear on the globe at right.", icon=":material/public:")

        # ---- 2. labels --------------------------------------------------------------------------
        with st.container(border=True):
            step_header(2, "Run labels", "How these runs are named and tagged. Names look like CNp-s046-d15-v6-... (scenario, salvo, timing, ...).")
            c1, c2 = st.columns([1, 2])
            scen_default = lc.default_scenario(filename=state["filename"], countries=state["countries"])
            scenario = c1.text_input("Scenario code", value=scen_default, key=f"scen_{scen_default}",
                                     help="2-5 capital letters at the start of every run name, e.g. NK, CN.")
            label = c2.text_input("Note (optional)", value="", help="Free text stored inside each result file.")
            c3, c4 = st.columns(2)
            principal = c3.checkbox("Principal set", help="Marks the runs as part of a principal research set (lowercase p in the name).")
            test = c4.checkbox("Test run", value=True, help="Adds a TEST- prefix so these runs stay out of real results (hidden by default in Results).")

        # ---- 3. grid ----------------------------------------------------------------------------
        grid: dict[str, list] = {}
        with st.container(border=True):
            step_header(3, "Engagement settings", "Every field takes a list (5, 10, 20) and/or a range (10-100:10). All combinations are run.")
            gcols = st.columns(2)
            for i, (key, spec) in enumerate(lc.GRID_FIELDS.items()):
                with gcols[i % 2]:
                    txt = st.text_input(spec["label"], value=spec["default"], help=spec["help"], key=f"grid_{key}")
                    try:
                        grid[key] = lc.parse_number_list(txt, name=spec["label"], kind=spec["kind"], minimum=spec["minimum"], maximum=spec["maximum"])
                        if len(grid[key]) > 1:
                            st.caption(f"{len(grid[key])} values: {', '.join(f'{v:g}' for v in grid[key][:10])}{' ...' if len(grid[key]) > 10 else ''}")
                    except lc.InputProblems as exc:
                        state["grid_errors"] = True
                        st.error(exc.problems[0])

        # ---- 4. review --------------------------------------------------------------------------
        with st.container(border=True):
            step_header(4, "Review", "Advanced settings apply to every run. The combination table lists exactly what will be submitted.")
            with st.expander("Advanced settings"):
                acols = st.columns(3)
                for i, (key, spec) in enumerate(lc.ADVANCED_FIELDS.items()):
                    default = DEFAULT_TIME_LIMIT_S if key == "time_limit_s" else CONFIG_DEFAULTS[key]
                    with acols[i % 3]:
                        if spec["kind"] is int:
                            fixed[key] = int(st.number_input(spec["label"], min_value=int(spec["minimum"]), max_value=int(spec["maximum"]),
                                                             value=int(default), help=spec["help"], key=f"adv_{key}"))
                        else:
                            fixed[key] = float(st.number_input(spec["label"], min_value=float(spec["minimum"]), max_value=float(spec["maximum"]),
                                                               value=float(default), format="%g", help=spec["help"], key=f"adv_{key}"))
            skip_existing = st.checkbox("Skip combinations that already have a finished result in the cloud", value=False,
                                        help="Off: every combination runs, layering new runs on top (same-day repeats get the next sequence letter). "
                                             "On: combinations whose run identity already has a result are not submitted.")
            if state["target"] is None:
                st.info("Complete step 1 to see the run plan.")
            elif state["grid_errors"]:
                st.info("Fix the highlighted fields in step 3 to see the run plan.")
            else:
                try:
                    plan = lc.build_grid_plan(target=state["target"], grid=grid, fixed=fixed, scenario=scenario, label=label, principal=principal, test=test)
                except lc.InputProblems as exc:
                    for p in exc.problems:
                        st.error(p)
            if plan is not None:
                _, existing = _load_index(st.session_state.get("index_token", 0))
                for p in plan.problems[:10]:
                    st.error(f"Excluded invalid combination - {p}")
                if len(plan.problems) > 10:
                    st.error(f"... and {len(plan.problems) - 10} more invalid combinations.")
                for w in plan.warnings:
                    st.warning(w)
                tbl = pd.DataFrame(plan.table)
                if not tbl.empty:
                    tbl["already_exists"] = tbl["identity"].map(lambda k: ", ".join(existing.get(k, [])[:2]))
                    dup = tbl["already_exists"] != ""
                    cfgs = [c for c, d in zip(plan.configs, dup) if not (skip_existing and d)]
                    with st.expander(f"Combination table ({len(tbl)})", expanded=len(tbl) <= 24):
                        st.dataframe(tbl.drop(columns=["identity"]), width="stretch", hide_index=True)
                    export_buttons("plan (CSV)", exp.df_csv_bytes(tbl), f"launch_plan_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", "plan")

    # ---- right: sticky preview + summary + submit -----------------------------------------------
    with right:
        st.markdown('<div class="sticky-marker"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            panel_title("Target preview · ECI globe")
            pts = state["points"]
            show_globe(pts, alt_km=intercept_alt_default, highlight=state["highlight"], key="globe_launch")
            if pts:
                s = lc.target_summary(pts)
                st.caption(f"{s['n']} target(s) at {intercept_alt_default:.0f} km altitude · drag to rotate, scroll to zoom · X axis = 0° longitude at epoch")
            else:
                st.caption("Targets appear here as soon as a CSV is provided or countries are chosen.")
        with st.container(border=True):
            panel_title("Run summary")
            ready_t = state["target"] is not None
            ready_g = not state["grid_errors"]
            n_sub = len(cfgs)
            hours = n_sub * float(fixed.get("time_limit_s", DEFAULT_TIME_LIMIT_S)) / 3600.0
            m1, m2 = st.columns(2)
            m1.metric("Combinations", len(tbl) if not tbl.empty else 0)
            m2.metric("Will submit", n_sub)
            m3, m4 = st.columns(2)
            m3.metric("Already in cloud", int(dup.sum()) if len(dup) else 0)
            m4.metric("Worst-case time", f"{hours:,.1f} h")
            st.markdown(
                f'<div class="check">{"✅" if ready_t else "⬜"} Targets &nbsp; {"✅" if ready_g else "❌"} Settings grid &nbsp; '
                f'{"✅" if plan is not None and not tbl.empty else "⬜"} Valid run plan</div>', unsafe_allow_html=True)
            st.caption("Worst-case = every run uses its full time limit; runs go in parallel up to the queue's vCPU limit (32 vCPUs per job).")
            confirmed = True
            if n_sub > lc.CONFIRM_ABOVE:
                confirmed = st.checkbox(f"I understand this launches {n_sub} AWS Batch jobs (up to {hours:,.0f} compute-hours).", key="confirm_big")
            if st.button(f"Submit {n_sub} job(s) to AWS", type="primary", icon=":material/rocket_launch:", width="stretch",
                         disabled=not (n_sub > 0 and confirmed)):
                s3, batch = _clients()
                bar = st.progress(0.0, text="Submitting ...")
                with guard("submitting jobs"):
                    res = lc.submit_runs(cfgs, s3=s3, batch=batch, progress=lambda d, n: bar.progress(d / n, text=f"Submitted {d}/{n}"))
                    bar.empty()
                    n_ok = sum(r["status"] == "SUBMITTED" for r in res)
                    (st.success if n_ok == len(res) else st.warning)(f"{n_ok} of {len(res)} job(s) submitted. Track them in the Jobs tab.")
                    st.session_state.setdefault("submissions", []).extend(res)
                    bad = [r for r in res if r["status"] != "SUBMITTED"]
                    if bad:
                        st.error(f"{len(bad)} job(s) failed to submit; first error: {bad[0]['error']}")
                        st.dataframe(pd.DataFrame(bad), hide_index=True)
            if n_sub == 0 and ready_t and ready_g:
                st.caption("Nothing to submit: every combination is invalid or already exists.")


# =============================================================================== JOBS
def render_jobs() -> None:
    st.caption("Live status of the AWS Batch queue. A job goes SUBMITTED → RUNNABLE → STARTING → RUNNING → SUCCEEDED/FAILED "
               "(RUNNABLE can last a few minutes while compute starts).")
    with st.container(border=True):
        c1, c2, c3 = st.columns([3, 1, 1])
        sel_status = c1.multiselect("Statuses", list(lc.JOB_STATUSES), default=["RUNNABLE", "STARTING", "RUNNING", "FAILED"])
        hours_back = c2.number_input("Created in the last (hours)", min_value=1, max_value=720, value=24)
        c3.write("")
        if c3.button("Refresh", icon=":material/refresh:", width="stretch"):
            st.session_state["jobs_refresh"] = datetime.datetime.now().timestamp()
    with guard("reading the Batch queue"):
        jobs = _jobs(tuple(sel_status or lc.JOB_STATUSES), st.session_state.get("jobs_refresh", 0.0))
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=int(hours_back))
        jdf = pd.DataFrame([j for j in jobs if j["created"] and j["created"] >= cutoff])
        if jdf.empty:
            st.info("No jobs match. Launch runs from the Launch tab, or widen the filters.", icon=":material/inbox:")
        else:
            counts = jdf["status"].value_counts()
            cols = st.columns(max(len(counts), 1))
            for c, (s, n) in zip(cols, counts.items()):
                c.metric(f"{STATUS_ICON.get(s, '⚪')} {s}", int(n))
            jdf["minutes"] = ((jdf["stopped"].fillna(pd.Timestamp.now(tz="UTC")) - jdf["started"]).dt.total_seconds() / 60).round(1)
            jdf.insert(0, "state", jdf["status"].map(lambda s: f"{STATUS_ICON.get(s, '⚪')} {s}"))
            show = jdf.drop(columns=["job_id", "status"])
            st.dataframe(show, width="stretch", hide_index=True, column_config={
                "state": st.column_config.TextColumn("Status"), "job_name": st.column_config.TextColumn("Job", width="large"),
                "created": st.column_config.DatetimeColumn("Created", format="MMM D, HH:mm"), "started": st.column_config.DatetimeColumn("Started", format="MMM D, HH:mm"),
                "stopped": st.column_config.DatetimeColumn("Stopped", format="MMM D, HH:mm"), "minutes": st.column_config.NumberColumn("Minutes", format="%.1f"),
                "reason": st.column_config.TextColumn("Reason")})
            export_buttons("job list (CSV)", exp.df_csv_bytes(jdf), "batch_jobs.csv", "text/csv", "jobs")
            nfail = int((jdf["status"] == "FAILED").sum())
            if nfail:
                st.warning(f"{nfail} failed job(s). In Results, tick 'Include failed / incomplete': failed runs record their error there. Common causes: "
                           "settings no orbit family can serve, or running out of memory on very large target sets.")
    subs = st.session_state.get("submissions", [])
    if subs:
        with st.container(border=True):
            panel_title("Submitted from this session")
            st.dataframe(pd.DataFrame(subs).drop(columns=["config_uri"], errors="ignore"), width="stretch", hide_index=True)


# =============================================================================== RESULTS
def _filters(frame: pd.DataFrame, prefix: str, cols: list[str], per_row: int = 4) -> pd.DataFrame:
    out = frame
    ui = st.columns(min(len(cols), per_row) or 1)
    for n, c in enumerate(cols):
        opts = sorted(frame[c].dropna().unique().tolist())
        with ui[n % len(ui)]:
            sel = st.multiselect(FILTER_COLS[c], opts, key=f"{prefix}_{c}")
        if sel:
            out = out[out[c].isin(sel)]
    return out


def _table_view(f: pd.DataFrame) -> pd.DataFrame:
    """Display copy of the run table: friendly status, percentage gap."""
    v = f[[c for c in TABLE_COLS if c in f]].copy()
    v["result_status"] = v["result_status"].map(lambda s: ("✅ " if s == "Optimal" else "⏱ " if s == "Time limit" else "❌ " if s in ("FAILED", "UNREADABLE") else "") + str(s))
    v["mip_gap_achieved"] = v["mip_gap_achieved"] * 100.0
    return v


def _empty_results(message: str, kind: str) -> None:
    """Empty state for the Results tab; the globe window is still shown (without targets)."""
    left, right = st.columns([3, 2], gap="large")
    with left:
        (st.info if kind == "info" else st.warning)(message, icon=":material/inbox:")
    with right:
        st.markdown('<div class="sticky-marker"></div>', unsafe_allow_html=True)
        with st.container(border=True):
            panel_title("Run targets · ECI globe")
            show_globe(None, alt_km=200.0, key="globe_results")
            st.caption("Select a run to see its targets here.")


def render_results() -> None:
    with st.container(border=True):
        top = st.columns([1.5, 1.5, 2.1, 3.4])
        if top[0].button("Refresh from S3", icon=":material/refresh:", width="stretch"):
            st.session_state["index_token"] = st.session_state.get("index_token", 0) + 1
            _load_index.clear()
        show_tests = top[1].checkbox("Include TEST runs", value=False)
        show_bad = top[2].checkbox("Include failed / incomplete", value=False)
    with guard("reading results from S3"):
        df_all, _existing = _load_index(st.session_state.get("index_token", 0))
        if df_all.empty:
            _empty_results("No runs found in the cloud yet. Launch some in the Launch tab.", "info")
            return
        df = df_all.copy()
        if not show_tests:
            df = df[~df["test"].fillna(False).astype(bool)]
        if not show_bad:
            df = df[df["n_satellites"].notna()]
        df = df.reset_index(drop=True)
        top[3].caption(f"{len(df)} of {len(df_all)} runs shown · index cached for 5 min, use Refresh after new jobs finish")
        if df.empty:
            _empty_results("No runs match the visibility switches. Tick 'Include TEST runs' or 'Include failed / incomplete'.", "warning")
            return

        left, right = st.columns([5, 3], gap="large")
        with left:
            sub_table, sub_compare, sub_export = st.tabs(["Runs and details", "Compare / plot", "Export"])
            # ------------------------------------------------------------ table + details
            with sub_table:
                with st.expander("Filters", expanded=False):
                    f = _filters(df, "tbl", list(FILTER_COLS))
                st.caption(f"{len(f)} run(s) match. Click a row to inspect it; its targets appear on the globe.")
                ev = st.dataframe(_table_view(f), width="stretch", hide_index=True, on_select="rerun", selection_mode="single-row", key="runs_table",
                                  column_config={
                                      "name": st.column_config.TextColumn("Run", width="large"), "date_iso": st.column_config.TextColumn("Date"),
                                      "scenario": "Scenario", "principal": st.column_config.CheckboxColumn("Principal"), "test": st.column_config.CheckboxColumn("Test"),
                                      "salvo_size": "Salvo", "detection_time_s": "Detect [s]", "decision_time_s": "Decide [s]",
                                      "burnout_velocity_km_s": "Vbo [km/s]", "max_accel_g": "Accel [g]", "target_missile_burnout_time_s": "Burnout [s]",
                                      "n_interceptors_per_sat": "Int/sat", "n_satellites": st.column_config.NumberColumn("Satellites", format="%d"),
                                      "result_status": "Status", "mip_gap_achieved": st.column_config.NumberColumn("Gap", format="%.1f%%"),
                                      "anchor_alt_km": st.column_config.NumberColumn("Anchor alt [km]", format="%.0f"), "n_targets": "Targets",
                                      "families": st.column_config.TextColumn("Families (N_P:N_D @ alt × sats)", width="large"), "hash": "Variant"})
                if ev.selection.rows:
                    st.session_state["active_run"] = f.iloc[ev.selection.rows[0]]["name"]
                active_name = st.session_state.get("active_run")
                if active_name in set(df["name"]):
                    r = df[df["name"] == active_name].iloc[0]
                    files = r["files"] if isinstance(r["files"], dict) else {}
                    with st.container(border=True):
                        st.markdown(f"#### {r['name']}")
                        if "result.json" not in files:
                            st.warning(f"This run has no result file yet (status: {r['result_status']}). {r.get('status_note') or ''}")
                        else:
                            side = _fetch_sidecar(files["result.json"])
                            rc, res = side["run_config"], side["result"]
                            a, b, c, d, e = st.columns(5)
                            a.metric("Satellites", res.get("n_satellites")); b.metric("Status", res.get("status"))
                            gap = res.get("mip_gap_achieved")
                            c.metric("MIP gap", "n/a" if gap is None else f"{gap * 100:.1f}%")
                            d.metric("Anchor alt (km)", "n/a" if rc.get("anchor_alt_km") is None else f"{rc['anchor_alt_km']:.0f}")
                            e.metric("Wall time (min)", "n/a" if rc.get("wall_time_s") is None else f"{rc['wall_time_s'] / 60:.1f}")
                            fams = pd.DataFrame(rc.get("families") or [])
                            if not fams.empty:
                                panel_title("Orbit families")
                                st.dataframe(fams[[c for c in ["N_P", "N_D", "inc_deg", "alt_km", "reach_deg", "n_raan_offsets", "satellites_selected", "is_anchor"] if c in fams]],
                                             width="stretch", hide_index=True)
                            with st.expander("Inputs, seed generation and filter, provenance"):
                                st.json({"inputs": {k: rc.get(k) for k in ("salvo_size", "n_interceptors_per_sat", "burnout_velocity_km_s", "max_accel_g",
                                                                         "detection_time_s", "decision_time_s", "target_missile_burnout_time_s",
                                                                         "intercept_alt_km", "r_max_km", "r_required", "n_targets")},
                                         "seed_generation": rc.get("seed_generation"), "seed_filter": rc.get("seed_filter"),
                                         "solver": rc.get("solver"), "provenance": side.get("provenance")})
                            panel_title("Files and checks")
                            s3, _ = _clients()
                            b1, b2 = st.columns(2)
                            if b1.button("Load JSON + CSV for download", key="load_files", icon=":material/cloud_download:", width="stretch"):
                                st.session_state["loaded"] = {"name": r["name"], "json": exp.json_bytes(side),
                                                              "csv": s3_index.read_bytes(s3, S3_BUCKET, files["result.csv"]) if "result.csv" in files else None}
                            loaded = st.session_state.get("loaded")
                            if loaded and loaded["name"] == r["name"]:
                                export_buttons("result JSON", loaded["json"], f"{r['name']}_result.json", "application/json", "one_json")
                                if loaded["csv"] is not None:
                                    export_buttons("satellite CSV", loaded["csv"], f"{r['name']}_result.csv", "text/csv", "one_csv")
                            if "result.csv" in files and b2.button("Re-simulate and check coverage", key="cov_check", icon=":material/verified:", width="stretch"):
                                with st.spinner("Downloading and re-simulating with the current code ..."):
                                    from sbi_coverage.runs.replay import coverage_check, orb_from_csv_text
                                    orb = orb_from_csv_text(s3_index.read_bytes(s3, S3_BUCKET, files["result.csv"]).decode("utf-8"))
                                    chk = coverage_check(side, orb)
                                (st.success if chk["n_covered"] == chk["n_targets"] else st.error)(
                                    f"{chk['n_covered']}/{chk['n_targets']} targets covered at every time step "
                                    f"(min count {chk['min_count']}, required {chk['required']}, mean {chk['mean_count']:.2f}).")
                                if chk["under_covered"]:
                                    st.dataframe(pd.DataFrame(chk["under_covered"]), hide_index=True)

            # ------------------------------------------------------------ compare
            with sub_compare:
                st.caption("Pick a scenario, choose which input gets its own curve, and hold the other inputs fixed with the filters.")
                with st.container(border=True):
                    c1, c2, c3 = st.columns(3)
                    scen = c1.selectbox("Scenario", sorted(df["scenario"].unique()))
                    curve = c2.selectbox("One curve per", list(CURVE_BY), format_func=CURVE_BY.get)
                    dup = c3.radio("Runs with identical inputs", ["Newest", "Fewest satellites"], horizontal=True)
                    cx, cy, cq = st.columns(3)
                    x_axis = cx.selectbox("x axis", list(AXES), index=list(AXES).index("salvo_size"), format_func=AXES.get)
                    y_axis = cy.selectbox("y axis", list(AXES), index=list(AXES).index("n_satellites"), format_func=AXES.get)
                    max_gap = cq.number_input("Exclude runs with MIP gap above", 0.0, 1.0, 1.0, 0.01,
                                              help="1.0 keeps everything; 0.05 keeps only runs proven within 5% of optimal.")
                    base = df[(df["scenario"] == scen) & df["n_satellites"].notna()]
                    base = base[base["mip_gap_achieved"].fillna(1.0) <= max_gap]
                    others = [c for c in CURVE_BY if c != curve and c != x_axis]
                    with st.expander("Hold other inputs fixed (empty = include all values)"):
                        fixed_df = _filters(base, "cmp", others)
                    variants = sorted(fixed_df["hash"].dropna().unique())
                    if len(variants) > 1:
                        pick_v = st.multiselect("Variant (hash of hidden settings: targets, seed settings, ...)", variants, default=variants,
                                                help="Runs with different target sets or seed settings but identical readable inputs are different variants.")
                        fixed_df = fixed_df[fixed_df["hash"].isin(pick_v)]
                if x_axis == y_axis:
                    st.error("Choose different x and y axes.")
                elif fixed_df.empty:
                    st.warning("No runs match these filters. Loosen the filters or the MIP gap limit.")
                else:
                    chosen = st.multiselect("Runs included (deselect to drop individual runs)", fixed_df["name"].tolist(),
                                            default=fixed_df["name"].tolist()[:300], key=f"chosen_{scen}_{curve}")
                    fixed_df = fixed_df[fixed_df["name"].isin(chosen)]
                    if fixed_df.empty:
                        st.warning("No runs selected.")
                    else:
                        extra = [c for c in others if fixed_df[c].nunique() > 1]
                        keys = [x_axis, curve, "hash"] + extra
                        ordered = fixed_df.sort_values("date_iso", ascending=False) if dup == "Newest" else fixed_df.sort_values(
                            ["n_satellites", "date_iso"], ascending=[True, False])
                        pts_df = ordered.drop_duplicates(keys).sort_values(x_axis).copy()
                        pts_df[y_axis] = pd.to_numeric(pts_df[y_axis], errors="coerce")
                        pts_df = pts_df[pts_df[y_axis].notna()]
                        if pts_df.empty:
                            st.warning(f"The selected runs have no values for '{AXES[y_axis]}'.")
                        else:
                            import plotly.express as px
                            pts_df["curve"] = pts_df[curve].map(lambda v: f"{CURVE_BY[curve]} = {v:g}")
                            if len(variants) > 1:
                                pts_df["curve"] += pts_df["hash"].map(lambda h: f" [variant {h}]")
                            if extra:
                                pts_df["curve"] += pts_df[extra].astype(str).agg(
                                    lambda row: "  [" + ", ".join(f"{FILTER_COLS[c]}={row[c]}" for c in extra) + "]", axis=1)
                            fig = px.line(pts_df, x=x_axis, y=y_axis, color="curve", markers=True, hover_name="name",
                                          labels={x_axis: AXES[x_axis], y_axis: AXES[y_axis], "curve": ""}, template="plotly_white")
                            fig.update_layout(height=470, legend=dict(orientation="h", y=-0.22), margin=dict(l=10, r=10, t=10, b=10),
                                              font=dict(family="Inter, system-ui, sans-serif"))
                            with st.container(border=True):
                                st.plotly_chart(fig, width="stretch")
                            if extra:
                                st.caption("More than one value of a remaining input is included, so curves are split further. Use the filters to fix them.")
                            show = pts_df[["name", "curve", x_axis, y_axis, "result_status", "mip_gap_achieved"]]
                            st.dataframe(show, width="stretch", hide_index=True)
                            stamp = f"{scen}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
                            export_buttons("plotted data (CSV)", exp.df_csv_bytes(show), f"compare_{stamp}.csv", "text/csv", "cmp_csv")
                            export_buttons("interactive plot (HTML)", fig.to_html(include_plotlyjs="cdn").encode(), f"compare_{stamp}.html", "text/html", "cmp_html")
                            try:
                                export_buttons("plot (PNG)", fig.to_image(format="png", scale=2), f"compare_{stamp}.png", "image/png", "cmp_png")
                            except Exception:
                                st.caption("PNG export needs `pip install kaleido`; the HTML export or the chart's camera icon work without it.")
                            if st.button(f"Fetch the full JSON of these {len(pts_df)} runs for export", key="cmp_json_fetch", icon=":material/cloud_download:"):
                                bundle, prog = {}, st.progress(0.0)
                                for i, nm in enumerate(pts_df["name"]):
                                    fl = pts_df.loc[pts_df["name"] == nm, "files"].iloc[0]
                                    bundle[nm] = _fetch_sidecar(fl["result.json"])
                                    prog.progress((i + 1) / len(pts_df))
                                prog.empty()
                                st.session_state["cmp_bundle"] = exp.json_bytes(bundle)
                            if "cmp_bundle" in st.session_state:
                                export_buttons("runs JSON bundle", st.session_state["cmp_bundle"], f"runs_{stamp}.json", "application/json", "cmp_bundle")

            # ------------------------------------------------------------ export whole table
            with sub_export:
                st.caption("Export the run index (one row per run) for any filter. Per-run JSON/CSV is under 'Runs and details' and 'Compare / plot'.")
                with st.container(border=True):
                    fx = _filters(df, "exp", list(FILTER_COLS))
                    cols_sel = st.multiselect("Columns", [c for c in fx.columns if c not in ("files", "path", "last_modified")],
                                              default=[c for c in TABLE_COLS if c in fx.columns])
                    if not cols_sel:
                        st.warning("Choose at least one column.")
                    else:
                        st.caption(f"{len(fx)} row(s) × {len(cols_sel)} column(s)")
                        stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
                        export_buttons("table (CSV)", exp.df_csv_bytes(fx, cols_sel), f"runs_index_{stamp}.csv", "text/csv", "tbl_csv")
                        export_buttons("table (JSON)", exp.json_bytes(fx[cols_sel].to_dict("records")), f"runs_index_{stamp}.json", "application/json", "tbl_json")

        # ---- right: sticky globe of the active run's targets ---------------------------------------
        with right:
            st.markdown('<div class="sticky-marker"></div>', unsafe_allow_html=True)
            with st.container(border=True):
                panel_title("Run targets · ECI globe")
                names = df["name"].tolist()
                active = st.session_state.get("active_run")
                idx = names.index(active) if active in names else 0
                pick = st.selectbox("Run", names, index=idx, label_visibility="collapsed", key="globe_run_pick",
                                    help="Defaults to the run selected in the table (newest run otherwise).")
                row = df[df["name"] == pick].iloc[0]
                files = row["files"] if isinstance(row["files"], dict) else {}
                if "result.json" in files:
                    side = _fetch_sidecar(files["result.json"])
                    tg = [[t.get("name", ""), t["lat_deg"], t["lon_deg"]] for t in side.get("targets", [])]
                    alt = float(side.get("cov", {}).get("intercept_alt_km", 200.0))
                    show_globe(tg, alt_km=alt, highlight=tuple(side.get("countries") or ()), key="globe_results")
                    if tg:
                        s = lc.target_summary(tg)
                        st.caption(f"{s['n']} target(s), latitude {s['lat_min']:.1f}° to {s['lat_max']:.1f}° · {alt:.0f} km altitude")
                else:
                    show_globe(None, alt_km=200.0, key="globe_results")
                    st.caption("This run has no result file, so its targets are unknown.")


with tab_launch:
    render_launch()
with tab_jobs:
    render_jobs()
with tab_results:
    render_results()
