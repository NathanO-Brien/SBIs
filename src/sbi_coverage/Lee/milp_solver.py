"""
SCLP solver for SBI constellation design using Gurobi.

Implements the Spatial-temporal Coverage Location Problem from:
  Williams Rogers et al., "Optimal Satellite Constellation Configuration
  Design: A Collection of Mixed Integer Linear Programs," JSR, 2026.
  doi: 10.2514/1.A36518

Replicates the MILPCollection.SCLP() formulation and Gurobi tuning from
MILPCollection.m. The single-block RGT symmetry-breaking constraint (pin x[0]=1) is available but off
by default, because the seed-family pool is several circulant blocks.

Interrupt behaviour: pressing Ctrl+C during solve_sclp() stops Gurobi and
returns the best incumbent found so far, exactly as in the MATLAB workflow.
"""
from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np
import scipy.sparse as sp

try:
    import gurobipy as gp
    from gurobipy import GRB
    _GUROBI_AVAILABLE = True
except ImportError:
    _GUROBI_AVAILABLE = False

from .layer import Layer

if TYPE_CHECKING:
    from ..core.config import CoverageConfig, SimConfig


# ---------------------------------------------------------------------------
# Fixed-width progress callback
# ---------------------------------------------------------------------------

# Column widths (characters) for the B&B progress table.
_W_FLAG = 2   # "H" or " "
_W_EXPL = 9   # explored nodes
_W_REM  = 9   # remaining nodes
_W_INC  = 14  # incumbent objective
_W_BD   = 14  # best bound
_W_GAP  = 9   # MIP gap %
_W_TIME = 8   # elapsed time

_HEADER = (
    f"{'':>{_W_FLAG}}  "
    f"{'Explored':>{_W_EXPL}}  "
    f"{'Remaining':>{_W_REM}}  "
    f"{'Incumbent':>{_W_INC}}  "
    f"{'Best Bound':>{_W_BD}}  "
    f"{'Gap':>{_W_GAP}}  "
    f"{'Time':>{_W_TIME}}"
)
_SEPARATOR = "-" * len(_HEADER)


def _fmt_time(t: float) -> str:
    """Format seconds compactly, e.g. 45s, 3m05s, 2h07m."""
    if t < 60:
        return f"{t:.0f}s"
    if t < 3600:
        return f"{int(t) // 60}m{int(t) % 60:02d}s"
    return f"{int(t) // 3600}h{(int(t) % 3600) // 60:02d}m"


def _progress_row(flag: str, expl: float, rem: float,
                  inc: float | None, bd: float | None, t: float) -> str:
    """Format one fixed-width row of the B&B progress table."""
    inc_s = f"{inc:.4f}" if inc is not None else "-"
    bd_s  = f"{bd:.4f}"  if bd  is not None else "-"
    if inc is not None and bd is not None and abs(inc) > 1e-10:
        gap_s = f"{100.0 * abs(inc - bd) / abs(inc):.2f}%"
    else:
        gap_s = "-"
    return (
        f"{flag:>{_W_FLAG}}  "
        f"{int(expl):>{_W_EXPL}}  "
        f"{int(rem):>{_W_REM}}  "
        f"{inc_s:>{_W_INC}}  "
        f"{bd_s:>{_W_BD}}  "
        f"{gap_s:>{_W_GAP}}  "
        f"{_fmt_time(t):>{_W_TIME}}"
    )


def _sanitize_callback_obj(value: float) -> float | None:
    """Hide Gurobi sentinel values before formatting callback progress."""
    if abs(value) >= GRB.INFINITY / 2:
        return None
    return value


def _make_sclp_callback(header_every: int = 25, stop_at_gap: float | None = None) -> Callable:
    """Return a Gurobi callback that prints a fixed-width B&B progress table.

    Suppressing Gurobi's native console output and routing through this
    callback ensures every column stays in exactly the same horizontal
    position regardless of how many digits the objective values contain.

    stop_at_gap, if given, terminates the solve as soon as the periodic MIP
    callback observes a live incumbent whose gap (matching Gurobi's own
    |ObjBst - ObjBnd| / |ObjBst| formula) drops below that threshold -- e.g.
    stop_at_gap=0.999999 stops as soon as ANY non-trivial lower bound is
    established (gap below 100%), useful for quickly comparing seed pools
    without waiting out a full time limit on every test.
    """
    state = {"rows": 0, "last_rem": 0}

    def _maybe_header() -> None:
        """Reprint the column header every header_every rows."""
        if state["rows"] % header_every == 0:
            print(_HEADER)
            print(_SEPARATOR)

    def callback(model: gp.Model, where: int) -> None:
        """Print progress on new incumbents (MIPSOL) and periodic MIP updates."""
        if where == GRB.Callback.MIPSOL:
            obj  = _sanitize_callback_obj(model.cbGet(GRB.Callback.MIPSOL_OBJ))
            bd   = _sanitize_callback_obj(model.cbGet(GRB.Callback.MIPSOL_OBJBND))
            expl = model.cbGet(GRB.Callback.MIPSOL_NODCNT)
            t    = model.cbGet(GRB.Callback.RUNTIME)
            _maybe_header()
            print(_progress_row("H", expl, state["last_rem"], obj, bd, t))
            state["rows"] += 1

        elif where == GRB.Callback.MIP:
            raw_inc = model.cbGet(GRB.Callback.MIP_OBJBST)
            raw_bd  = model.cbGet(GRB.Callback.MIP_OBJBND)
            expl    = model.cbGet(GRB.Callback.MIP_NODCNT)
            rem     = model.cbGet(GRB.Callback.MIP_NODLFT)
            t       = model.cbGet(GRB.Callback.RUNTIME)
            state["last_rem"] = rem
            inc = _sanitize_callback_obj(raw_inc)
            bd = _sanitize_callback_obj(raw_bd)
            _maybe_header()
            print(_progress_row(" ", expl, rem, inc, bd, t))
            state["rows"] += 1

            if (
                stop_at_gap is not None
                and np.isfinite(raw_inc)
                and raw_inc != 0.0
            ):
                gap = abs(raw_inc - raw_bd) / abs(raw_inc)
                if gap < stop_at_gap:
                    model.terminate()

    return callback


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class SCLPResult:
    """Output from solve_sclp()."""
    selected_slots: np.ndarray   # 0-indexed selected orbital slot indices, shape (n_sat,)
    obj_val: float               # objective value / incumbent (= n_satellites for uniform cost)
    obj_bound: float             # best lower bound on the optimal objective at termination
    mip_gap: float               # final MIP optimality gap (0.0 = proven optimal)
    status: int                  # Gurobi status code
    status_str: str              # human-readable status string
    time_modeling_s: float       # model construction wall-clock time (s)
    time_optimization_s: float   # solver wall-clock time (s)
    x_binary: np.ndarray         # full binary solution vector, shape (nOrbitalSlots,)
    n_satellites: int            # number of selected satellites
    node_count: int = 0          # branch-and-bound nodes explored


_GUROBI_STATUS_STRINGS = {
    1:  "Loaded",
    2:  "Optimal",
    3:  "Infeasible",
    4:  "Infeasible or unbounded",
    5:  "Unbounded",
    6:  "Cutoff",
    7:  "Iteration limit",
    8:  "Node limit",
    9:  "Time limit",
    10: "Solution limit",
    11: "User interrupted",
    12: "Numeric issues",
    13: "Suboptimal",
    14: "In progress",
    15: "User objective limit",
    16: "Work limit",
    17: "Memory limit",
}

#: Gurobi parameters this module always sets (before caller overrides); recorded in every result JSON.
GUROBI_FIXED_PARAMS = {"CoverCuts": 2, "Cuts": 2, "Symmetry": 2, "LPWarmStart": 1}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def solve_sclp(
    param_V: np.ndarray | sp.spmatrix,
    param_r: np.ndarray,
    *,
    param_c: np.ndarray | None = None,
    apply_rgt_symmetry_break: bool = False,
    mip_gap: float = 0.0,
    time_limit_s: float = float("inf"),
    log_file: str | Path | None = None,
    verbose: bool = True,
    mip_focus: int = 0,
    heuristics_frac: float = 0.05,
    solver_params: dict[str, Any] | None = None,
    stop_at_gap: float | None = None,
) -> SCLPResult:
    """Solve the SCLP using Gurobi.

    Parameters
    ----------
    param_V : (L*n_targets, nOrbitalSlots) array-like or sparse matrix
        Visibility constraint matrix from build_param_V().
        Row j*L + t is target j at time step t.
    param_r : (L, n_targets) or (L,) float64
        Coverage requirement.  param_r[t, j] is the minimum number of
        satellites that must simultaneously see target j at time t.
    param_c : (nOrbitalSlots,) float64, optional
        Cost vector.  Defaults to all-ones (objective = satellite count).
    apply_rgt_symmetry_break : bool
        If True, apply the single-block RGT symmetry-breaking constraint
        (pin x[0] = 1). This is appropriate only when the candidate pool
        consists of one circulant common-ground-track block.
    mip_gap : float
        Gurobi MIPGap tolerance.  0.0 requires proven optimality.
    time_limit_s : float
        Wall-clock time limit in seconds.  Ctrl+C also stops the solver
        and returns the best incumbent found so far.
    log_file : str or Path, optional
        Path for the Gurobi log file.  Parent directory is created as needed.
    verbose : bool
        If False, suppress all Gurobi console output.
    mip_focus : int
        Gurobi MIPFocus parameter (0–3).
        0 = balanced/automatic (Gurobi's own default).
        1 = find feasible solutions fast (good default for sweep / large problems).
        2 = close the optimality gap (better when first incumbent is easy to find).
        3 = tighten the best bound (rarely useful here).
        Default: 0 (Gurobi's own default -- no repo-specific tuning applied;
        pass 1 explicitly to get the old sweep-friendly behavior back).
    heuristics_frac : float
        Fraction of B&B time Gurobi spends on primal heuristics (0–1).
        Higher values help find the first incumbent faster at the cost of
        slower bound improvement.  Default: 0.05 (Gurobi's own default).
    solver_params : dict, optional
        Arbitrary Gurobi parameter overrides applied after all defaults.
        Keys are Gurobi parameter name strings; values are the desired
        settings.  These take precedence over every hardcoded value above,
        including MIPFocus and Heuristics.  Useful for per-run tuning
        without touching the solver code.  Example::

            solver_params={
                "Method":        2,    # barrier root LP
                "NodefileStart": 4,    # disk offload at 4 GB
                "RINS":          50,   # neighbourhood search every 50 nodes
                "Presolve":      2,    # aggressive presolve
                "Symmetry":      0,    # disable (circulant near-symmetry)
            }

    stop_at_gap : float, optional
        If set, terminate the solve as soon as a live incumbent's gap drops
        below this value -- e.g. 0.999999 stops as soon as any non-trivial
        lower bound is established (gap under 100%), well short of proving
        the mip_gap tolerance. Only checked when verbose=True (it hooks into
        the same progress callback). Intended for quickly comparing seed
        pools/problem sizes across test runs without waiting out a full
        time_limit_s on each one. None (default): no early termination
        beyond mip_gap/time_limit_s.

    Returns
    -------
    SCLPResult
        Always contains the best incumbent found.  Check mip_gap and
        status_str to determine whether the solution is proven optimal.

    Raises
    ------
    ImportError
        If gurobipy is not installed or no valid license is found.
    RuntimeError
        If the solver terminates with no feasible solution at all.
    """
    if not _GUROBI_AVAILABLE:
        raise ImportError(
            "gurobipy is required: pip install gurobipy  "
            "(requires a valid Gurobi license)"
        )

    n_rows, n_orb_slots = param_V.shape

    # Flatten param_r to match param_V row ordering.
    # param_V rows are indexed j*L + t (target-major).
    # param_r shape is (L, n_targets); Fortran-order flatten gives
    # [target-0 block, target-1 block, ...] which matches.
    param_r = np.asarray(param_r, dtype=np.float64)
    if param_r.size != n_rows:
        raise ValueError(f"param_r has {param_r.size} entries but param_V has {n_rows} rows")
    param_r_flat = param_r.flatten(order="F")

    if param_c is None:
        param_c = np.ones(n_orb_slots, dtype=np.float64)
    else:
        param_c = np.asarray(param_c, dtype=np.float64).ravel()

    # ------------------------------------------------------------------
    # Build model
    # ------------------------------------------------------------------
    t0 = time.perf_counter()

    import os as _os
    _env = None
    _wls_access = _os.environ.get("GRB_WLSACCESSID", "").strip()
    _wls_secret = _os.environ.get("GRB_WLSSECRET", "").strip()
    _wls_licid  = _os.environ.get("GRB_LICENSEID", "").strip()
    if _wls_access and _wls_secret and _wls_licid:
        # Cloud path: explicit WLS env so Gurobi doesn't fall back to the
        # restricted default license when credentials are injected via env vars.
        _env = gp.Env(params={
            "WLSAccessID": _wls_access,
            "WLSSecret":   _wls_secret,
            "LicenseID":   int(_wls_licid),
        })
        m = gp.Model("SCLP", env=_env)
    else:
        # Local path: use whatever license Gurobi finds automatically
        # (named-user academic license, gurobi.lic file, etc.).
        m = gp.Model("SCLP")
    if verbose:
        # Suppress Gurobi's native stdout so our fixed-width callback owns
        # the console.  OutputFlag=1 keeps the full log flowing to log_file.
        m.Params.LogToConsole = 0
    else:
        m.Params.OutputFlag = 0

    # Gurobi tuning
    m.Params.MIPFocus    = int(mip_focus)      # 1=find incumbents fast, 2=close gap
    m.Params.Heuristics  = float(heuristics_frac)
    # GUROBI_FIXED_PARAMS: aggressive cover cuts / cuts for the set-covering structure, internal
    # symmetry detection, LP warm start. Recorded in the result JSON.
    for _pname, _pval in GUROBI_FIXED_PARAMS.items():
        m.setParam(_pname, _pval)
    m.Params.MIPGap      = mip_gap

    if time_limit_s < float("inf"):
        m.Params.TimeLimit = float(time_limit_s)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        m.Params.LogFile = str(log_file)

    # Apply caller-supplied overrides last so they take precedence over
    # every hardcoded value above (including MIPFocus, Heuristics, etc.).
    if solver_params:
        for _param_name, _param_val in solver_params.items():
            m.setParam(_param_name, _param_val)

    # Decision variable: x ∈ {0,1}^nOrbitalSlots
    x = m.addMVar(n_orb_slots, vtype=GRB.BINARY, name="x")

    # Objective: min c'x
    m.setObjective(param_c @ x, GRB.MINIMIZE)

    # Coverage constraint: V * x >= r  (sparse for large L)
    V_sp = param_V.tocsr() if sp.issparse(param_V) else sp.csr_matrix(param_V)
    m.addMConstr(V_sp, x, GRB.GREATER_EQUAL, param_r_flat, name="coverage")

    # RGT symmetry breaking: pin x[0] = 1.
    # Valid only when all slots share a common RGT ground track (circulant
    # param_V block).  Pins the seed satellite, collapsing L equivalent
    # cyclic rotations into one — reduces B&B tree by factor of ~L.
    if apply_rgt_symmetry_break:
        m.addConstr(x[0] == 1, name="sym_break")

    t_model = time.perf_counter() - t0
    if verbose:
        print("SCLP modeling done")

    # ------------------------------------------------------------------
    # Solve — Ctrl+C stops Gurobi and returns best incumbent
    # ------------------------------------------------------------------
    t_solve_start = time.perf_counter()
    callback = _make_sclp_callback(stop_at_gap=stop_at_gap) if verbose else None
    try:
        m.optimize(callback)
    except KeyboardInterrupt:
        print("\nInterrupted — stopping solver, returning best incumbent...")
        m.terminate()

    t_solve = time.perf_counter() - t_solve_start

    status_str = _GUROBI_STATUS_STRINGS.get(m.Status, f"Unknown ({m.Status})")

    try:
        if m.SolCount > 0:
            x_sol = x.X
            selected = np.where(x_sol > 0.5)[0]
            return SCLPResult(
                selected_slots=selected,
                obj_val=float(m.ObjVal),
                obj_bound=float(m.ObjBound),
                mip_gap=float(m.MIPGap),
                status=int(m.Status),
                status_str=status_str,
                time_modeling_s=t_model,
                time_optimization_s=t_solve,
                x_binary=x_sol,
                n_satellites=int(selected.size),
                node_count=int(m.NodeCount),
            )
        raise RuntimeError(
            f"SCLP returned no feasible solution.  "
            f"Gurobi status: {status_str}.  "
            f"Check coverage requirements or engagement parameters."
        )
    finally:
        # Release the model and (cloud) WLS session so repeated solves in one process do not leak seats.
        m.dispose()
        if _env is not None:
            _env.dispose()


# ---------------------------------------------------------------------------
# Post-processing helpers
# ---------------------------------------------------------------------------

def build_orb_elems_table(seed_layers: list[Layer], cols_per_seed: list[int]) -> np.ndarray:
    """Build the orbital elements table for all candidate slots (the columns of param_V).

    Parameters
    ----------
    seed_layers : list of Layer
        One Layer (L slots) per kept seed.
    cols_per_seed : list of int
        Slots kept per seed (must match build_param_V); a reducible family keeps only L // g.

    Returns
    -------
    orb_elems : (nOrbitalSlots, 6) float64
        Columns: a_km | e | i_deg | raan_deg | argp_deg | arg_lat_deg, where arg_lat_deg = argp + M0
        (= M0 for circular orbits). Row r is column r of param_V: seed 0 slots 0.., then seed 1, etc.
    """
    n_orb_slots = int(sum(cols_per_seed))
    orb_elems = np.zeros((n_orb_slots, 6), dtype=np.float64)
    start = 0
    for s, layer in enumerate(seed_layers):
        e = layer.elems
        m = int(cols_per_seed[s])
        sl = slice(0, m)
        orb_elems[start:start + m, 0] = np.asarray(e.a_km,     dtype=np.float64)[sl]
        orb_elems[start:start + m, 1] = np.asarray(e.e,        dtype=np.float64)[sl]
        orb_elems[start:start + m, 2] = np.rad2deg(np.asarray(e.i_rad,    dtype=np.float64))[sl]
        orb_elems[start:start + m, 3] = np.rad2deg(np.asarray(e.raan_rad, dtype=np.float64))[sl]
        orb_elems[start:start + m, 4] = np.rad2deg(np.asarray(e.argp_rad, dtype=np.float64))[sl]
        orb_elems[start:start + m, 5] = np.rad2deg(
            np.asarray(e.argp_rad, dtype=np.float64)[sl]
            + np.asarray(e.M0_rad,  dtype=np.float64)[sl]
        )
        start += m
    return orb_elems


def _finite_or_none(v: float) -> float | None:
    """Return v if finite, otherwise None.

    JSON does not support Infinity or NaN.  Gurobi reports MIPGap = inf when
    no lower bound has been established yet (e.g. solver terminated before the
    first incumbent or before any LP relaxation was solved).
    """
    f = float(v)
    return f if math.isfinite(f) else None


_ORB_COLS = ["slot_index", "a_km", "e", "i_deg", "raan_deg", "argp_deg", "arg_lat_deg"]


def save_orb_table_csv(
    orb_elems: np.ndarray,
    selected_slots: np.ndarray,
    output_path: str | Path,
    extra_columns: dict[str, list] | None = None,
) -> None:
    """Save the orbital characteristics table for the selected slots to CSV.

    Parameters
    ----------
    orb_elems : (nOrbitalSlots, 6) float64
        Full table from build_orb_elems_table().
    selected_slots : (n_selected,) int
        0-indexed slot indices from SCLPResult.selected_slots.
    output_path : str or Path
        Destination CSV file. Parent directory is created as needed.
    extra_columns : dict, optional
        Name -> per-slot values for ALL slots (length nOrbitalSlots); the selected rows are written
        after ``slot_index`` (e.g. ``seed_index``, ``family``, ``seed_raan_deg``).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected_elems = orb_elems[selected_slots]
    extra_columns = extra_columns or {}
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([_ORB_COLS[0], *extra_columns.keys(), *_ORB_COLS[1:]])
        for idx, row in zip(selected_slots, selected_elems):
            writer.writerow([int(idx)] + [extra_columns[k][int(idx)] for k in extra_columns]
                            + [f"{v:.6f}" for v in row])
    print(f"Saved orbital table ({len(selected_slots)} satellites): {output_path}")


def print_orb_table(orb_elems: np.ndarray, selected_slots: np.ndarray) -> None:
    """Print the orbital characteristics table to stdout.

    Matches the four-column MATLAB table() output.
    """
    selected_elems = orb_elems[selected_slots]
    header = f"{'a_km':>12}  {'i_deg':>10}  {'raan_deg':>10}  {'arg_lat_deg':>13}"
    print(header)
    print("-" * len(header))
    for row in selected_elems:
        print(f"{row[0]:12.2f}  {row[2]:10.4f}  {row[3]:10.4f}  {row[5]:13.4f}")


SCHEMA_VERSION = 2


def save_sclp_result(
    result: SCLPResult,
    orb_elems: np.ndarray,
    csv_path: str | Path,
    *,
    cov: CoverageConfig,
    sim: SimConfig,
    targets: list[dict[str, Any]],
    countries: list[str],
    run_name: str | None = None,
    run_config: dict[str, Any] | None = None,
    provenance: dict[str, Any] | None = None,
    extra_columns: dict[str, list] | None = None,
) -> Path:
    """Save an SCLP result as ``<stem>.csv`` (selected satellites) and ``<stem>.json`` (sidecar).

    The JSON sidecar (schema_version 2) is everything the replay/analysis scripts need to
    reconstruct CoverageConfig, SimConfig and the target points without manual re-entry:

        schema_version, run_name, timestamp_utc
        result      : n_satellites, status, obj_val, obj_bound, mip_gap_achieved, timings, node_count
        cov / sim   : CoverageConfig / SimConfig fields
        targets     : [{name, lat_deg, lon_deg}, ...];  countries: country names (country mode only)
        run_config  : inputs and derived quantities of the run (see Lee.run_pipeline.build_run_config)
        provenance  : code/library versions and the machine the run executed on

    Returns the JSON path.
    """
    csv_path = Path(csv_path)
    save_orb_table_csv(orb_elems, result.selected_slots, csv_path, extra_columns)

    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "result": {
            "n_satellites": int(result.n_satellites),
            "status": result.status_str,
            "obj_val": _finite_or_none(result.obj_val),
            "obj_bound": _finite_or_none(result.obj_bound),
            "mip_gap_achieved": _finite_or_none(result.mip_gap),
            "node_count": int(result.node_count),
            "time_modeling_s": float(result.time_modeling_s),
            "time_optimization_s": float(result.time_optimization_s),
        },
        "cov": {
            "detection_time_s":               float(cov.detection_time_s),
            "decision_time_s":                float(cov.decision_time_s),
            "target_missile_burnout_time_s":  float(cov.target_missile_burnout_time_s),
            "interceptor_engagement_time_s":  float(cov.interceptor_engagement_time_s),
            "burnout_velocity_km_s":          float(cov.burnout_velocity_km_s),
            "max_accel_g":                    float(cov.max_accel_g),
            "intercept_alt_km":               float(cov.intercept_alt_km),
            "min_elev_deg":                   float(cov.min_elev_deg),
            "max_range_km":                   float(cov.max_range_km),
        },
        "sim": {
            "horizon_s":             float(sim.horizon_s),
            "dt_s":                  float(sim.dt_s),
            "use_j2":                bool(sim.use_j2),
            "analysis_matrix_dtype": str(sim.analysis_matrix_dtype),
        },
        "targets":    targets,
        "countries":  [str(country) for country in countries],
        "run_config": run_config if run_config is not None else {},
        "provenance": provenance if provenance is not None else {},
    }

    json_path = csv_path.with_suffix(".json")
    with json_path.open("w") as f:
        json.dump(_json_safe(payload), f, indent=2, allow_nan=False)
    print(f"Saved config sidecar: {json_path}")
    return json_path


def _json_safe(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and non-finite floats into strict-JSON values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, np.generic):
        return _json_safe(obj.item())
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj
