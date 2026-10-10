"""SBI optimizer GUI: launch many runs on AWS, watch the jobs, and pull/plot/export results.

    streamlit run scripts/sbi_gui.py

The GUI works only with the cloud: runs are launched as AWS Batch jobs and results are read straight from
s3://sbi-optimization-runs/runs/. It never reads your local results folder (exports are written to a folder
you choose, default ~/Downloads/SBI_exports). Needs AWS credentials (`aws configure`).

Tabs
----
 Launch runs   upload prespecified targets (CSV) or pick countries + globe points, define a grid of settings
               (lists/ranges), preview every combination, and submit them as AWS Batch jobs
 Jobs          live status of the Batch queue and of what you submitted in this session
 Results       filter the runs in S3, inspect one run (families, seeds, provenance, coverage re-check),
               compare many runs on any axes, and export CSV/JSON
"""
from __future__ import annotations

import datetime
import sys
import tempfile
import traceback
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi_coverage.core.target_points import TargetFileError, parse_target_points_text  # noqa: E402
from sbi_coverage.runs import export as exp  # noqa: E402
from sbi_coverage.runs import launch as lc  # noqa: E402
from sbi_coverage.runs import s3_index  # noqa: E402
from sbi_coverage.runs.config import CONFIG_DEFAULTS, JOB_QUEUE, S3_BUCKET, S3_RUNS_PREFIX  # noqa: E402
from sbi_coverage.runs.names import decode_run_name  # noqa: E402

st.set_page_config(page_title="SBI optimizer", layout="wide")

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


# =============================================================================== helpers
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
def _fetch_sidecar(key: str, etag: str):
    s3, _ = _clients()
    return s3_index.normalize_sidecar(s3_index.read_json(s3, S3_BUCKET, key, etag))


@st.cache_data(ttl=30, show_spinner="Reading the Batch queue ...")
def _jobs(statuses: tuple[str, ...], token: float):
    _, batch = _clients()
    return lc.list_recent_jobs(batch, statuses=statuses)


@st.cache_data(ttl=3600, show_spinner=False)
def _countries():
    return lc.country_names()


def _export_buttons(label: str, data: bytes, filename: str, mime: str, key: str):
    """A download button plus a 'save to export folder' button for the same data."""
    c1, c2 = st.columns([1, 1])
    c1.download_button(f"Download {label}", data, filename, mime, key=f"dl_{key}")
    if c2.button(f"Save {label} to export folder", key=f"sv_{key}"):
        folder, err = exp.validate_export_dir(st.session_state.get("export_dir", str(exp.DEFAULT_EXPORT_DIR)))
        if err:
            st.error(err)
        else:
            st.success(f"Saved: {exp.save_export(folder, filename, data)}")


# =============================================================================== sidebar / AWS check
st.sidebar.title("SBI optimizer")
st.sidebar.caption(f"Cloud only: results are read from s3://{S3_BUCKET}/{S3_RUNS_PREFIX}; jobs run on AWS Batch queue '{JOB_QUEUE}'.")
st.sidebar.text_input("Export folder (for 'Save to export folder')", value=str(exp.DEFAULT_EXPORT_DIR), key="export_dir",
                      help="CSV/JSON exports are written here. The browser 'Download' buttons use your normal download folder.")
if st.sidebar.button("Re-check AWS connection"):
    _aws_problems.clear()
aws_problems = _aws_problems()
if aws_problems:
    st.sidebar.error("AWS is not ready")
    st.title("SBI optimizer")
    for p in aws_problems:
        st.error(p)
    st.info("Fix the problem above, then use 'Re-check AWS connection' in the sidebar (or reload the page).")
    st.stop()
st.sidebar.success("AWS connected")

st.title("SBI optimizer")
tab_launch, tab_jobs, tab_results = st.tabs(["Launch runs", "Jobs", "Results"])

# =============================================================================== LAUNCH
with tab_launch:
    st.markdown("Define the targets, set the grid of engagement settings, review every combination, then submit. "
                "Each combination becomes one AWS Batch job; results appear in the **Results** tab when they finish.")
    st.subheader("1. Targets")
    mode = st.radio("Target mode", ["Prespecified points (CSV upload)", "Country mode (deprecated)"], horizontal=True,
                    help="Prespecified mode covers exactly the points you upload. Country mode covers every globe point inside the chosen country(ies).")
    target: dict | None = None
    summary_text = ""
    filename = None
    countries_sel: list[str] = []
    if mode.startswith("Prespecified"):
        up = st.file_uploader("Target file (CSV with Name, Lat, Lon columns)", type=["csv", "txt"], key="target_upload",
                              help="One target per row. Header like: Name,Lat,Lon (decimal degrees, + north / + east). Comma, semicolon or tab separated.")
        pasted = st.text_area("...or paste CSV text", height=80, key="target_paste",
                              placeholder="Name,Lat,Lon\nSohae,39.66,124.70\nYongbyon,39.79,125.75")
        text, source = (None, None)
        if up is not None:
            try:
                text, source = up.getvalue().decode("utf-8-sig"), up.name
                filename = up.name
            except UnicodeDecodeError:
                st.error("The file is not UTF-8 text. In Excel use 'Save As > CSV UTF-8'.")
        elif pasted.strip():
            text, source = pasted, "pasted text"
        if text is not None:
            try:
                points, warns = parse_target_points_text(text, source=source)
                target = {"mode": "prespecified", "points": points}
                s = lc.target_summary(points)
                a, b, c = st.columns(3)
                a.metric("Targets", s["n"]); b.metric("Latitude span (deg)", f"{s['lat_span']:.2f}")
                c.metric("Latitude range", f"{s['lat_min']:.1f} to {s['lat_max']:.1f}")
                for w in warns:
                    st.warning(w)
                if s["lat_span"] > 15:
                    st.warning("Targets span more than 15 degrees of latitude: few synchronized orbit families can reach all of them, "
                               "so the satellite count may be high. Consider splitting the set by latitude band.")
                dfp = pd.DataFrame(points, columns=["name", "lat", "lon"])
                st.map(dfp, latitude="lat", longitude="lon", size=20000)
                with st.expander("Preview targets"):
                    st.dataframe(dfp, width="stretch", hide_index=True)
                summary_text = f"{s['n']} targets"
            except TargetFileError as exc:
                st.error(f"The target file cannot be used ({source}):")
                for p in exc.problems:
                    st.markdown(f"- {p}")
        else:
            st.info("Upload or paste a target list to continue.")
    else:
        st.warning("Country mode is deprecated: it covers every globe point inside the country, which is usually far larger than "
                   "the set of strategic sites. Prefer prespecified points.")
        names = _countries()
        if names:
            countries_sel = st.multiselect("Countries", names, help="Names from Natural Earth. Several countries are combined into one target set.")
        else:
            st.caption("Country list unavailable (offline or geopandas missing); type names instead.")
            countries_sel = [c.strip() for c in st.text_input("Countries (comma separated)").split(",") if c.strip()]
        n_points = st.number_input("Globe points", min_value=1000, max_value=200000, value=15000, step=1000,
                                   help="Size of the Fibonacci sphere sampled before masking to the country. More points = finer coverage of the "
                                        "country but a much larger optimization; the number of points that fall inside the country is what counts.")
        if countries_sel:
            if st.button("Preview how many points fall inside", key="preview_country"):
                with guard("previewing the country mask"):
                    from sbi_coverage.core.config import EarthConstants
                    from sbi_coverage.core.shell import Shell
                    sh = Shell(EarthConstants(), n_points=int(n_points), shell_alt_km=200.0).mask_country(countries_sel)
                    if sh.n_points_used == 0:
                        st.error("No globe points fall inside the selection. Increase 'Globe points'.")
                    else:
                        st.success(f"{sh.n_points_used} target points, latitude {sh.lat_deg.min():.1f} to {sh.lat_deg.max():.1f}")
                        if sh.n_points_used > 500:
                            st.warning("More than 500 target points: the optimization will be large and slow. Consider fewer globe points.")
            target = {"mode": "country", "countries": countries_sel, "n_points": int(n_points)}
            summary_text = f"{', '.join(countries_sel)} @ {int(n_points)} globe points"
        else:
            st.info("Pick at least one country to continue.")

    st.subheader("2. Run labels")
    c1, c2, c3, c4 = st.columns([1, 2, 1, 1])
    scen_default = lc.default_scenario(filename=filename, countries=countries_sel)
    scenario = c1.text_input("Scenario code", value=scen_default, key=f"scen_{scen_default}",
                             help="2-5 capital letters shown at the start of every run name, e.g. NK, CN.")
    label = c2.text_input("Label (optional note stored in each result)", value="")
    principal = c3.checkbox("Principal set", help="Marks runs as part of a principal research set (lowercase p in the name).")
    test = c4.checkbox("Test run", value=True, help="Adds a TEST- prefix so these runs stay out of the real results (hidden by default in Results).")

    st.subheader("3. Settings grid")
    st.caption("Every field takes a comma list (5, 10, 20) and/or ranges (10-100:10). All combinations are run.")
    grid: dict[str, list] = {}
    grid_errors = False
    gcols = st.columns(2)
    for i, (key, spec) in enumerate(lc.GRID_FIELDS.items()):
        with gcols[i % 2]:
            txt = st.text_input(spec["label"], value=spec["default"], help=spec["help"], key=f"grid_{key}")
            try:
                grid[key] = lc.parse_number_list(txt, name=spec["label"], kind=spec["kind"], minimum=spec["minimum"], maximum=spec["maximum"])
                if len(grid[key]) > 1:
                    st.caption(f"{len(grid[key])} values: {', '.join(f'{v:g}' for v in grid[key][:12])}{' ...' if len(grid[key]) > 12 else ''}")
            except lc.InputProblems as exc:
                grid_errors = True
                st.error(exc.problems[0])
    fixed: dict = {}
    with st.expander("Advanced settings (applied to every run)"):
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

    st.subheader("4. Review and submit")
    skip_existing = st.checkbox("Skip combinations that already have a finished result in the cloud", value=False,
                                help="Off: every combination is run, layering new runs on top of existing ones (same-day repeats get the next "
                                     "sequence letter). On: combinations whose run identity already has a result are not submitted.")
    plan = None
    if target is None:
        st.info("Complete step 1 to see the run plan.")
    elif grid_errors:
        st.info("Fix the highlighted grid fields to see the run plan.")
    else:
        try:
            plan = lc.build_grid_plan(target=target, grid=grid, fixed=fixed, scenario=scenario, label=label,
                                      principal=principal, test=test)
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
            to_run = tbl[~dup] if skip_existing else tbl
            cfgs = [c for c, d in zip(plan.configs, dup) if not (skip_existing and d)]
            hours = len(cfgs) * float(fixed["time_limit_s"]) / 3600.0
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Combinations", len(tbl)); m2.metric("Already finished in cloud", int(dup.sum()))
            m3.metric("Will submit", len(cfgs)); m4.metric("Worst-case solver time", f"{hours:,.1f} h total")
            st.caption("Worst-case = every run uses its full time limit; runs execute in parallel up to the queue's vCPU limit "
                       "(each job reserves 32 vCPUs).")
            with st.expander(f"Combination table ({len(tbl)})", expanded=len(tbl) <= 30):
                st.dataframe(tbl.drop(columns=["identity"]), width="stretch", hide_index=True)
            _export_buttons("plan (CSV)", exp.df_csv_bytes(tbl), f"launch_plan_{datetime.datetime.now():%Y%m%d_%H%M%S}.csv", "text/csv", "plan")
            if skip_existing and dup.any():
                st.info(f"{int(dup.sum())} combination(s) skipped because a finished run already exists.")
            ok_to_submit = len(cfgs) > 0
            if not ok_to_submit:
                st.warning("Nothing to submit (all combinations are invalid or already exist).")
            confirmed = True
            if len(cfgs) > lc.CONFIRM_ABOVE:
                confirmed = st.checkbox(f"I understand this launches {len(cfgs)} AWS Batch jobs (up to {hours:,.0f} compute-hours).", key="confirm_big")
            if st.button(f"Submit {len(cfgs)} job(s) to AWS", type="primary", disabled=not (ok_to_submit and confirmed)):
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

# =============================================================================== JOBS
with tab_jobs:
    st.markdown("Live status of the AWS Batch queue. Jobs go SUBMITTED -> RUNNABLE -> STARTING -> RUNNING -> SUCCEEDED/FAILED "
                "(RUNNABLE can last minutes while compute starts).")
    sel_status = st.multiselect("Statuses", list(lc.JOB_STATUSES), default=["RUNNABLE", "STARTING", "RUNNING", "FAILED"])
    hours_back = st.number_input("Only jobs created in the last (hours)", min_value=1, max_value=720, value=24)
    if st.button("Refresh job list"):
        st.session_state["jobs_refresh"] = datetime.datetime.now().timestamp()
    with guard("reading the Batch queue"):
        jobs = _jobs(tuple(sel_status or lc.JOB_STATUSES), st.session_state.get("jobs_refresh", 0.0))
        cutoff = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(hours=int(hours_back))
        jdf = pd.DataFrame([j for j in jobs if j["created"] and j["created"] >= cutoff])
        if jdf.empty:
            st.info("No jobs match. Launch runs from the Launch tab, or widen the filters.")
        else:
            counts = jdf["status"].value_counts()
            cols = st.columns(max(len(counts), 1))
            for c, (s, n) in zip(cols, counts.items()):
                c.metric(s, int(n))
            jdf["minutes"] = ((jdf["stopped"].fillna(pd.Timestamp.now(tz="UTC")) - jdf["started"]).dt.total_seconds() / 60).round(1)
            st.dataframe(jdf.drop(columns=["job_id"]), width="stretch", hide_index=True)
            _export_buttons("job list (CSV)", exp.df_csv_bytes(jdf), "batch_jobs.csv", "text/csv", "jobs")
            nfail = int((jdf["status"] == "FAILED").sum())
            if nfail:
                st.warning(f"{nfail} failed job(s). Open the Results tab with 'Include failed' ticked: failed runs record their error there. "
                           "Common causes: infeasible settings (no orbit family reaches the targets), or out-of-memory for very large target sets.")
    subs = st.session_state.get("submissions", [])
    if subs:
        st.subheader("Submitted from this session")
        st.dataframe(pd.DataFrame(subs).drop(columns=["config_uri"], errors="ignore"), width="stretch", hide_index=True)

# =============================================================================== RESULTS
with tab_results:
    top = st.columns([1, 1, 1, 3])
    if top[0].button("Refresh from S3"):
        st.session_state["index_token"] = st.session_state.get("index_token", 0) + 1
        _load_index.clear()
    show_tests = top[1].checkbox("Include TEST runs", value=False)
    show_bad = top[2].checkbox("Include failed / incomplete", value=False)
    with guard("reading results from S3"):
        df_all, _existing = _load_index(st.session_state.get("index_token", 0))
        if df_all.empty:
            st.info("No runs found in the cloud yet. Launch some in the Launch tab.")
            st.stop()
        df = df_all.copy()
        if not show_tests:
            df = df[~df["test"].fillna(False).astype(bool)]
        if not show_bad:
            df = df[df["n_satellites"].notna()]
        df = df.reset_index(drop=True)
        top[3].caption(f"{len(df)} of {len(df_all)} runs shown (index cached for 5 minutes; use Refresh after new jobs finish).")
        if df.empty:
            st.warning("No runs match the visibility switches. Tick 'Include TEST runs' or 'Include failed / incomplete'.")
            st.stop()

        sub_table, sub_compare, sub_export = st.tabs(["Runs and details", "Compare / plot", "Export"])

        def _filters(frame: pd.DataFrame, prefix: str, cols: list[str]) -> pd.DataFrame:
            out = frame
            ui = st.columns(min(len(cols), 4) or 1)
            for n, c in enumerate(cols):
                opts = sorted(frame[c].dropna().unique().tolist())
                with ui[n % len(ui)]:
                    sel = st.multiselect(FILTER_COLS[c], opts, key=f"{prefix}_{c}")
                if sel:
                    out = out[out[c].isin(sel)]
            return out

        # ---------------------------------------------------------------- table + details
        with sub_table:
            f = _filters(df, "tbl", list(FILTER_COLS))
            st.caption(f"{len(f)} run(s) match. Select a row to inspect it.")
            ev = st.dataframe(f[[c for c in TABLE_COLS if c in f]], width="stretch", hide_index=True,
                              on_select="rerun", selection_mode="single-row")
            if ev.selection.rows:
                r = f.iloc[ev.selection.rows[0]]
                st.subheader(r["name"])
                files = r["files"] if isinstance(r["files"], dict) else {}
                if "result.json" not in files:
                    st.warning(f"This run has no result file yet (status: {r['result_status']}). {r.get('status_note') or ''}")
                else:
                    side = _fetch_sidecar(files["result.json"], "")
                    rc, res = side["run_config"], side["result"]
                    a, b, c, d, e = st.columns(5)
                    a.metric("Satellites", res.get("n_satellites")); b.metric("Status", res.get("status"))
                    gap = res.get("mip_gap_achieved")
                    c.metric("MIP gap", "n/a" if gap is None else f"{gap * 100:.1f}%")
                    d.metric("Anchor altitude (km)", None if rc.get("anchor_alt_km") is None else f"{rc['anchor_alt_km']:.0f}")
                    e.metric("Wall time (min)", None if rc.get("wall_time_s") is None else f"{rc['wall_time_s'] / 60:.1f}")
                    fams = pd.DataFrame(rc.get("families") or [])
                    if not fams.empty:
                        st.markdown("**Orbit families** (satellites selected per family)")
                        st.dataframe(fams[[c for c in ["N_P", "N_D", "inc_deg", "alt_km", "reach_deg", "n_raan_offsets", "satellites_selected", "is_anchor"] if c in fams]],
                                     width="stretch", hide_index=True)
                    with st.expander("Inputs, seed generation and filter, provenance"):
                        st.json({"inputs": {k: rc.get(k) for k in ("salvo_size", "n_interceptors_per_sat", "burnout_velocity_km_s", "max_accel_g",
                                                                 "detection_time_s", "decision_time_s", "target_missile_burnout_time_s",
                                                                 "intercept_alt_km", "r_max_km", "r_required", "n_targets")},
                                 "seed_generation": rc.get("seed_generation"), "seed_filter": rc.get("seed_filter"),
                                 "solver": rc.get("solver"), "provenance": side.get("provenance")})
                    st.markdown("**Files and checks**")
                    s3, _ = _clients()
                    cols = st.columns(3)
                    if cols[0].button("Load JSON + CSV for download", key="load_files"):
                        st.session_state["loaded"] = {"name": r["name"], "json": exp.json_bytes(side),
                                                      "csv": s3_index.read_bytes(s3, S3_BUCKET, files["result.csv"]) if "result.csv" in files else None}
                    loaded = st.session_state.get("loaded")
                    if loaded and loaded["name"] == r["name"]:
                        _export_buttons("result JSON", loaded["json"], f"{r['name']}_result.json", "application/json", "one_json")
                        if loaded["csv"] is not None:
                            _export_buttons("satellite CSV", loaded["csv"], f"{r['name']}_result.csv", "text/csv", "one_csv")
                    if "result.csv" in files and cols[1].button("Re-simulate and check coverage", key="cov_check"):
                        with st.spinner("Downloading and re-simulating with the current code ..."):
                            from sbi_coverage.runs.replay import coverage_check, orb_from_csv_text
                            orb = orb_from_csv_text(s3_index.read_bytes(s3, S3_BUCKET, files["result.csv"]).decode("utf-8"))
                            chk = coverage_check(side, orb)
                        (st.success if chk["n_covered"] == chk["n_targets"] else st.error)(
                            f"{chk['n_covered']}/{chk['n_targets']} targets covered at every time step "
                            f"(min count {chk['min_count']}, required {chk['required']}, mean {chk['mean_count']:.2f}).")
                        if chk["under_covered"]:
                            st.dataframe(pd.DataFrame(chk["under_covered"]), hide_index=True)

        # ---------------------------------------------------------------- compare
        with sub_compare:
            st.markdown("Pick a scenario, choose which input gets its own curve, and hold the other inputs fixed with the filters.")
            c1, c2, c3 = st.columns(3)
            scen = c1.selectbox("Scenario", sorted(df["scenario"].unique()))
            curve = c2.selectbox("One curve per", list(CURVE_BY), format_func=CURVE_BY.get)
            dup = c3.radio("If several runs share the same inputs", ["Newest", "Fewest satellites"], horizontal=True)
            cx, cy, cq = st.columns(3)
            x_axis = cx.selectbox("x axis", list(AXES), index=list(AXES).index("salvo_size"), format_func=AXES.get)
            y_axis = cy.selectbox("y axis", list(AXES), index=list(AXES).index("n_satellites"), format_func=AXES.get)
            max_gap = cq.number_input("Exclude runs with MIP gap above", 0.0, 1.0, 1.0, 0.01,
                                      help="1.0 keeps everything; 0.05 keeps only runs proven within 5% of optimal.")
            base = df[(df["scenario"] == scen) & df["n_satellites"].notna()]
            base = base[base["mip_gap_achieved"].fillna(1.0) <= max_gap]
            others = [c for c in CURVE_BY if c != curve and c != x_axis]
            fixed_df = _filters(base, "cmp", others)
            variants = sorted(fixed_df["hash"].dropna().unique())
            if len(variants) > 1:
                pick = st.multiselect("Variant (hash of hidden settings: targets, seed settings, ...)", variants, default=variants,
                                      help="Runs with different target sets or seed settings but identical readable inputs are different variants.")
                fixed_df = fixed_df[fixed_df["hash"].isin(pick)]
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
                    pts = ordered.drop_duplicates(keys).sort_values(x_axis).copy()
                    pts[y_axis] = pd.to_numeric(pts[y_axis], errors="coerce")
                    pts = pts[pts[y_axis].notna()]
                    if pts.empty:
                        st.warning(f"The selected runs have no values for '{AXES[y_axis]}'.")
                    else:
                        import plotly.express as px
                        pts["curve"] = pts[curve].map(lambda v: f"{CURVE_BY[curve]} = {v:g}")
                        if len(variants) > 1:
                            pts["curve"] += pts["hash"].map(lambda h: f" [variant {h}]")
                        if extra:
                            pts["curve"] += pts[extra].astype(str).agg(
                                lambda row: "  [" + ", ".join(f"{FILTER_COLS[c]}={row[c]}" for c in extra) + "]", axis=1)
                        fig = px.line(pts, x=x_axis, y=y_axis, color="curve", markers=True, hover_name="name",
                                      labels={x_axis: AXES[x_axis], y_axis: AXES[y_axis], "curve": ""})
                        fig.update_layout(height=520, legend=dict(orientation="h", y=-0.2))
                        st.plotly_chart(fig, width="stretch")
                        if extra:
                            st.caption("More than one value of a remaining input is included, so curves are split further. Use the filters to fix them.")
                        show = pts[["name", "curve", x_axis, y_axis, "result_status", "mip_gap_achieved"]]
                        st.dataframe(show, width="stretch", hide_index=True)
                        stamp = f"{scen}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
                        _export_buttons("plotted data (CSV)", exp.df_csv_bytes(show), f"compare_{stamp}.csv", "text/csv", "cmp_csv")
                        _export_buttons("interactive plot (HTML)", fig.to_html(include_plotlyjs="cdn").encode(), f"compare_{stamp}.html", "text/html", "cmp_html")
                        try:
                            _export_buttons("plot (PNG)", fig.to_image(format="png", scale=2), f"compare_{stamp}.png", "image/png", "cmp_png")
                        except Exception:
                            st.caption("PNG export needs `pip install kaleido`; the HTML export or the chart's camera icon work without it.")
                        if st.button(f"Fetch the full JSON of these {len(pts)} runs for export", key="cmp_json_fetch"):
                            s3, _ = _clients()
                            bundle = {}
                            prog = st.progress(0.0)
                            for i, nm in enumerate(pts["name"]):
                                fl = pts.loc[pts["name"] == nm, "files"].iloc[0]
                                bundle[nm] = _fetch_sidecar(fl["result.json"], "")
                                prog.progress((i + 1) / len(pts))
                            prog.empty()
                            st.session_state["cmp_bundle"] = exp.json_bytes(bundle)
                        if "cmp_bundle" in st.session_state:
                            _export_buttons("runs JSON bundle", st.session_state["cmp_bundle"], f"runs_{stamp}.json", "application/json", "cmp_bundle")

        # ---------------------------------------------------------------- export whole table
        with sub_export:
            st.markdown("Export the run index (one row per run) for any filter. Detailed per-run JSON/CSV is on the "
                        "'Runs and details' and 'Compare / plot' tabs.")
            fx = _filters(df, "exp", list(FILTER_COLS))
            cols_sel = st.multiselect("Columns", [c for c in fx.columns if c not in ("files", "path", "last_modified")],
                                      default=[c for c in TABLE_COLS if c in fx.columns])
            if not cols_sel:
                st.warning("Choose at least one column.")
            else:
                st.caption(f"{len(fx)} row(s) x {len(cols_sel)} column(s)")
                stamp = f"{datetime.datetime.now():%Y%m%d_%H%M%S}"
                _export_buttons("table (CSV)", exp.df_csv_bytes(fx, cols_sel), f"runs_index_{stamp}.csv", "text/csv", "tbl_csv")
                _export_buttons("table (JSON)", exp.json_bytes(fx[cols_sel].to_dict("records")), f"runs_index_{stamp}.json", "application/json", "tbl_json")
