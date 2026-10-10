"""Browse SBI optimization runs, compare them, and check their coverage.

    streamlit run scripts/run_browser.py -- --folder results

Runs are flat files named by sbi_coverage.runs.names, so everything here is derived from file names plus
each run's _result.json sidecar. Tabs:
  Runs      table of every run (filter by any subset of inputs), details of a selected run, coverage re-check
  Compare   any numeric input/result on x vs y (default salvo vs satellites), one curve per detection time /
            burnout velocity / max accel / ... (the rest held fixed with the filters)
  Name tools  decode a run name; "does a run with these inputs exist?" (local folder)
TEST runs are hidden unless ticked in the sidebar. Runs that differ only in hidden settings (target set, seed
settings, ...) are told apart by the Variant column (the 4-hex hash in the name).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi_coverage.runs.config import S3_BUCKET  # noqa: E402
from sbi_coverage.runs.index import list_s3_runs, load_index, sync_from_s3  # noqa: E402
from sbi_coverage.runs.names import decode_run_name  # noqa: E402

FILTER_COLS = {
    "scenario": "Scenario", "principal": "Principal set", "salvo_size": "Salvo", "detection_time_s": "Detection [s]",
    "decision_time_s": "Decision [s]", "burnout_velocity_km_s": "Burnout vel [km/s]", "max_accel_g": "Max accel [g]",
    "target_missile_burnout_time_s": "Burnout time [s]", "n_interceptors_per_sat": "Interceptors/sat",
}
CURVE_BY = {k: v for k, v in FILTER_COLS.items() if k not in ("scenario", "principal", "salvo_size")}
AXES = {**{k: v for k, v in FILTER_COLS.items() if k not in ("scenario", "principal")},
        "n_satellites": "Satellites required", "mip_gap_achieved": "MIP gap", "time_optimization_s": "Solve time [s]",
        "anchor_alt_km": "Anchor altitude [km]", "n_targets": "Targets", "r_required": "r required"}

st.set_page_config(page_title="SBI run browser", layout="wide")


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="results")
    return ap.parse_known_args(sys.argv[1:])[0]


@st.cache_data(show_spinner=False)
def _index(folder: str, stamp: float) -> pd.DataFrame:
    return load_index(folder)


def _stamp(folder: str) -> float:
    files = [*Path(folder).glob("*_result.json"), *Path(folder).glob("*_failed.json")]
    return max((f.stat().st_mtime for f in files), default=0.0) + len(files)


def _filters(df: pd.DataFrame, key_prefix: str, cols: list[str]) -> pd.DataFrame:
    out = df
    ui = st.columns(min(len(cols), 5) or 1)
    for n, c in enumerate(cols):
        opts = sorted(df[c].dropna().unique().tolist())
        with ui[n % len(ui)]:
            sel = st.multiselect(FILTER_COLS[c], opts, key=f"{key_prefix}_{c}")
        if sel:
            out = out[out[c].isin(sel)]
    return out


args = _args()
st.sidebar.header("Data")
folder = st.sidebar.text_input("Results folder", args.folder)
st.sidebar.caption("Flat folder of `<run name>_result.json/.csv/...` files.")
show_tests = st.sidebar.checkbox("Show TEST runs", value=False)
show_failed = st.sidebar.checkbox("Show failed runs", value=True)
hide_unfinished = st.sidebar.checkbox("Hide runs without a satellite count", value=True)
with st.sidebar.expander("Sync from S3"):
    sync_scen = st.text_input("Only scenario (e.g. NK; blank = all)", "")
    sync_overwrite = st.checkbox("Overwrite existing local files", value=False)
    if st.button(f"Download missing runs ({S3_BUCKET}/runs/)"):
        with st.spinner("Downloading..."):
            try:
                names = [n for n in list_s3_runs() if not sync_scen or decode_run_name(n)["scenario"] == sync_scen.upper()]
                new = sync_from_s3(folder, names=names, overwrite=sync_overwrite)
                st.success(f"{len(new)} run(s) updated")
            except Exception as exc:  # network / credentials
                st.error(f"Sync failed: {exc}")
        st.cache_data.clear()
if st.sidebar.button("Rescan folder"):
    st.cache_data.clear()

df_all = _index(folder, _stamp(folder))
df = df_all
if not df.empty:
    if not show_tests:
        df = df[~df["test"].fillna(False).astype(bool)]
    if not show_failed:
        df = df[df["result_status"].fillna("") != "FAILED"]
    if hide_unfinished:
        df = df[df["n_satellites"].notna() | (df["result_status"].fillna("") == "FAILED")]
    df = df.reset_index(drop=True)

st.title("SBI run browser")
if df_all.empty:
    st.info(f"No runs found in `{Path(folder).resolve()}`. Run an optimization or use Sync from S3.")
    st.stop()
st.caption(f"{len(df)} of {len(df_all)} runs shown in `{Path(folder).resolve()}`")
if df.empty:
    st.warning("All runs are hidden by the sidebar switches.")
    st.stop()

tab_runs, tab_cmp, tab_name = st.tabs(["Runs", "Compare", "Name tools"])

# --------------------------------------------------------------------------- Runs
with tab_runs:
    f = _filters(df, "runs", list(FILTER_COLS))
    f = f.assign(variant=f["hash"])
    show = ["name", "date_iso", "scenario", "principal", "test", "salvo_size", "detection_time_s", "decision_time_s",
            "burnout_velocity_km_s", "max_accel_g", "target_missile_burnout_time_s", "n_interceptors_per_sat",
            "n_satellites", "result_status", "mip_gap_achieved", "variant", "families"]
    sel = st.dataframe(f[show], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
    st.download_button("Download table (CSV)", f[show].to_csv(index=False), "runs.csv", "text/csv")
    if sel.selection.rows:
        r = f.iloc[sel.selection.rows[0]]
        st.subheader(r["name"])
        c1, c2 = st.columns(2)
        c1.json({k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.drop(labels=["path"]).items()})
        c2.code(r["path"])
        csv = Path(r["path"]).with_suffix(".csv")
        if csv.exists():
            c2.download_button("Download satellite table (CSV)", csv.read_bytes(), csv.name, "text/csv")
            if c2.button("Re-simulate and check coverage"):
                with st.spinner("Re-simulating with current code..."):
                    from sbi_coverage.core.simulate import run_simulation
                    from sbi_coverage.runs.replay import (PROPAGATOR, elements_from_orb, load_run, predefined_shell,
                                                          replay_configs)
                    run = load_run(str(csv), folder)
                    earth, cov, sim = replay_configs(run["sidecar"])
                    shell = predefined_shell(run["sidecar"], earth, cov)
                    ao = run_simulation(elements_from_orb(run["orb"]), sim, cov, earth, shell, [{"type": "stored"}],
                                        propagator=PROPAGATOR)
                    need = int(run["sidecar"]["run_config"].get("r_required", 1))
                    counts = ao.counts.astype(np.int32)
                    ok = int((counts.min(axis=0) >= need).sum())
                    c2.success(f"{ok}/{counts.shape[1]} targets covered at every step "
                               f"(min count {int(counts.min())}, required {need})")

# --------------------------------------------------------------------------- Compare
with tab_cmp:
    label = df.scenario + df.principal.map({True: " (principal)", False: ""})
    c1, c2, c3 = st.columns(3)
    scen = c1.selectbox("Scenario", sorted(label.unique()))
    curve = c2.selectbox("One curve per", list(CURVE_BY), format_func=CURVE_BY.get)
    dup = c3.radio("If several runs share the same inputs", ["Newest", "Fewest satellites"], horizontal=True)
    cx, cy, cq = st.columns(3)
    x_axis = cx.selectbox("x axis", list(AXES), index=list(AXES).index("salvo_size"), format_func=AXES.get)
    y_axis = cy.selectbox("y axis", list(AXES), index=list(AXES).index("n_satellites"), format_func=AXES.get)
    max_gap = cq.number_input("Exclude runs with MIP gap above", 0.0, 1.0, 1.0, 0.01)
    base = df[(label == scen) & df["n_satellites"].notna()]
    base = base[base["mip_gap_achieved"].fillna(1.0) <= max_gap]
    st.markdown("**Hold the other inputs fixed** (leave empty = include all values):")
    others = [c for c in CURVE_BY if c != curve and c != x_axis]
    fixed = _filters(base, "cmp", others)
    variants = sorted(fixed["hash"].dropna().unique())
    if len(variants) > 1:
        pick = st.multiselect("Variant (hash of hidden settings: target set, seed settings, ...)", variants, default=variants)
        fixed = fixed[fixed["hash"].isin(pick)]
    if fixed.empty:
        st.warning("No runs match.")
    else:
        extra = [c for c in others if fixed[c].nunique() > 1]
        keys = [x_axis, curve, "hash"] + extra
        fixed = fixed.sort_values("date_iso", ascending=False) if dup == "Newest" else fixed.sort_values(
            ["n_satellites", "date_iso"], ascending=[True, False])
        pts = fixed.drop_duplicates(keys).sort_values(x_axis).copy()
        pts["curve"] = pts[curve].map(lambda v: f"{CURVE_BY[curve]} = {v:g}")
        if len(variants) > 1:
            pts["curve"] += pts["hash"].map(lambda h: f" [variant {h}]")
        if extra:
            pts["curve"] += pts[extra].astype(str).agg(
                lambda r: "  [" + ", ".join(f"{FILTER_COLS[c]}={r[c]}" for c in extra) + "]", axis=1)
        fig = px.line(pts, x=x_axis, y=y_axis, color="curve", markers=True, hover_name="name",
                      labels={x_axis: AXES[x_axis], y_axis: AXES[y_axis], "curve": ""})
        fig.update_layout(height=520, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        if extra:
            st.caption("More than one value of a remaining input is included, so curves are split further. "
                       "Pick values above to hold them fixed.")
        st.dataframe(pts[["name", "curve", x_axis, y_axis, "result_status", "mip_gap_achieved"]],
                     use_container_width=True, hide_index=True)
        st.download_button("Download plotted data (CSV)", pts[["name", "curve", x_axis, y_axis]].to_csv(index=False),
                           "comparison.csv", "text/csv")
        try:
            st.download_button("Download figure (PNG)", fig.to_image(format="png", scale=2), "comparison.png", "image/png")
        except Exception:
            st.caption("PNG export needs `pip install kaleido`; use the camera icon on the chart instead.")

# --------------------------------------------------------------------------- Name tools
with tab_name:
    st.subheader("Decode a name")
    nm = st.text_input("Run name or file name", value=str(df.iloc[0]["name"]))
    try:
        st.json(decode_run_name(nm))
    except ValueError as exc:
        st.error(str(exc))
    st.subheader("Does a run with these inputs exist?")
    st.caption("Leave a field empty to ignore it. Checks the local folder (all runs, including hidden ones).")
    q = {}
    cols = st.columns(5)
    for n, c in enumerate(FILTER_COLS):
        q[c] = cols[n % 5].multiselect(FILTER_COLS[c], sorted(df_all[c].dropna().unique().tolist()), key=f"ex_{c}")
    m = df_all
    for c, v in q.items():
        if v:
            m = m[m[c].isin(v)]
    if any(q.values()):
        st.success(f"{len(m)} matching run(s)") if len(m) else st.warning("No matching run.")
        st.dataframe(m[["name", "date_iso", "n_satellites", "result_status", "hash"]], use_container_width=True, hide_index=True)
