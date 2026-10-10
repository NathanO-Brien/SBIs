"""Browse SBI optimization runs and compare them.

    streamlit run scripts/run_browser.py -- --folder results

Runs are flat files named by sbi_coverage.runnames, so everything here is derived from
file names plus each run's _result.json sidecar.  Tabs:
  Runs     table of every run (filter by any subset of inputs), name decoder, "does it exist?" check
  Compare  salvo size vs number of satellites for one scenario, one curve per detection time /
           burnout velocity / max accel / ... (the rest held fixed with the filters)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi_coverage.runindex import S3_BUCKET, load_index, sync_from_s3  # noqa: E402
from sbi_coverage.runnames import decode_run_name  # noqa: E402

FILTER_COLS = {
    "scenario": "Scenario", "principal": "Principal set", "salvo_size": "Salvo", "detection_time_s": "Detection [s]",
    "decision_time_s": "Decision [s]", "v_bo_km_s": "Burnout vel [km/s]", "a_g": "Max accel [g]",
    "target_missile_burnout_time_s": "Burnout time [s]", "n_interceptors_per_sat": "Interceptors/sat",
}
CURVE_BY = {k: v for k, v in FILTER_COLS.items() if k not in ("scenario", "principal", "salvo_size")}

st.set_page_config(page_title="SBI run browser", layout="wide")


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default="results")
    return ap.parse_known_args(sys.argv[1:])[0]


@st.cache_data(show_spinner=False)
def _index(folder: str, stamp: float) -> pd.DataFrame:
    return load_index(folder)


def _stamp(folder: str) -> float:
    p = Path(folder)
    return max((f.stat().st_mtime for f in p.glob("*_result.json")), default=0.0) + len(list(p.glob("*_result.json")))


def _multiselect_filters(df: pd.DataFrame, key_prefix: str, cols: list[str]) -> pd.DataFrame:
    out = df
    cols_ui = st.columns(min(len(cols), 5) or 1)
    for n, c in enumerate(cols):
        opts = sorted(df[c].dropna().unique().tolist())
        with cols_ui[n % len(cols_ui)]:
            sel = st.multiselect(FILTER_COLS[c], opts, key=f"{key_prefix}_{c}")
        if sel:
            out = out[out[c].isin(sel)]
    return out


args = _args()
st.sidebar.header("Data")
folder = st.sidebar.text_input("Results folder", args.folder)
st.sidebar.caption(f"Flat folder of `<run name>_result.json/.csv/...` files.")
if st.sidebar.button(f"Sync from S3 ({S3_BUCKET}/runs/)"):
    with st.spinner("Downloading runs missing locally..."):
        try:
            new = sync_from_s3(folder)
            st.sidebar.success(f"{len(new)} new run(s)")
        except Exception as exc:  # network / credentials
            st.sidebar.error(f"Sync failed: {exc}")
    st.cache_data.clear()
if st.sidebar.button("Rescan folder"):
    st.cache_data.clear()

df = _index(folder, _stamp(folder))
show_tests = st.sidebar.checkbox("Show TEST runs", value=False)
if not show_tests and "test" in df:
    df = df[~df["test"].fillna(False).astype(bool)].reset_index(drop=True)
st.title("SBI run browser")
if df.empty:
    st.info(f"No visible runs (test runs hidden unless ticked) in `{Path(folder).resolve()}`. Run an optimization or use Sync from S3.")
    st.stop()
st.caption(f"{len(df)} runs in `{Path(folder).resolve()}`")

tab_runs, tab_cmp, tab_name = st.tabs(["Runs", "Compare", "Name decoder / exists check"])

# --------------------------------------------------------------------------- Runs
with tab_runs:
    f = _multiselect_filters(df, "runs", list(FILTER_COLS))
    show = ["name", "date_iso", "scenario", "principal", "salvo_size", "detection_time_s", "decision_time_s", "v_bo_km_s",
            "a_g", "target_missile_burnout_time_s", "n_interceptors_per_sat", "n_satellites", "result_status",
            "mip_gap_achieved", "families"]
    sel = st.dataframe(f[show], use_container_width=True, hide_index=True, on_select="rerun", selection_mode="single-row")
    st.download_button("Download table (CSV)", f[show].to_csv(index=False), "runs.csv", "text/csv")
    if sel.selection.rows:
        r = f.iloc[sel.selection.rows[0]]
        st.subheader(r["name"])
        c1, c2 = st.columns(2)
        c1.json({k: (None if pd.isna(v) else v) for k, v in r.drop(labels=["path"]).items() if not isinstance(v, float) or v == v})
        c2.code(r["path"])
        csv = Path(r["path"]).with_suffix(".csv")
        if csv.exists():
            c2.download_button("Download satellite table (CSV)", csv.read_bytes(), csv.name, "text/csv")

# --------------------------------------------------------------------------- Compare
with tab_cmp:
    c1, c2, c3 = st.columns(3)
    scen_opts = sorted(df.assign(s=df.scenario + df.principal.map({True: " (principal)", False: ""}))["s"].unique())
    scen = c1.selectbox("Scenario", scen_opts)
    curve = c2.selectbox("One curve per", list(CURVE_BY), format_func=CURVE_BY.get)
    dup = c3.radio("If several runs share the same settings", ["Newest", "Fewest satellites"], horizontal=True)
    base = df[(df.scenario + df.principal.map({True: " (principal)", False: ""})) == scen]
    base = base[base.n_satellites.notna()]
    st.markdown("**Hold the other inputs fixed** (leave empty = include all values):")
    others = [c for c in CURVE_BY if c != curve]
    fixed = _multiselect_filters(base, "cmp", others)
    if "orbit_model" in fixed:
        fixed = fixed[fixed.orbit_model.fillna("lee_j2") == "lee_j2"]
    if fixed.empty:
        st.warning("No runs match.")
    else:
        keys = ["salvo_size", curve] + [c for c in others if fixed[c].nunique() > 1]
        fixed = fixed.sort_values("date", ascending=False) if dup == "Newest" else fixed.sort_values(
            ["n_satellites", "date"], ascending=[True, False])
        pts = fixed.drop_duplicates(keys).sort_values("salvo_size")
        extra = [c for c in others if fixed[c].nunique() > 1]
        pts["curve"] = pts[curve].map(lambda v: f"{CURVE_BY[curve]} = {v:g}")
        if extra:
            pts["curve"] += pts[extra].astype(str).agg(lambda r: "  [" + ", ".join(f"{FILTER_COLS[c]}={r[c]}" for c in extra) + "]", axis=1)
        fig = px.line(pts, x="salvo_size", y="n_satellites", color="curve", markers=True, hover_name="name",
                      labels={"salvo_size": "Salvo size", "n_satellites": "Satellites required", "curve": ""})
        fig.update_layout(height=520, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig, use_container_width=True)
        if extra:
            st.caption("Note: more than one value of a remaining input is included, so curves are split further. "
                       "Pick values above to hold them fixed.")
        st.dataframe(pts[["name", "curve", "salvo_size", "n_satellites", "result_status", "mip_gap_achieved"]],
                     use_container_width=True, hide_index=True)
        st.download_button("Download plotted data (CSV)", pts[["name", "curve", "salvo_size", "n_satellites"]].to_csv(index=False),
                           "comparison.csv", "text/csv")
        try:
            png = fig.to_image(format="png", scale=2)
            st.download_button("Download figure (PNG)", png, "comparison.png", "image/png")
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
    st.caption("Leave a field empty to ignore it.")
    q = {}
    cols = st.columns(5)
    for n, c in enumerate(FILTER_COLS):
        q[c] = cols[n % 5].multiselect(FILTER_COLS[c], sorted(df[c].dropna().unique().tolist()), key=f"ex_{c}")
    m = df
    for c, v in q.items():
        if v:
            m = m[m[c].isin(v)]
    if any(q.values()):
        st.success(f"{len(m)} matching run(s)") if len(m) else st.warning("No matching run.")
        st.dataframe(m[["name", "date_iso", "n_satellites", "result_status"]], use_container_width=True, hide_index=True)
