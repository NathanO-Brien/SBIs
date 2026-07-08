from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from .analysis_matrix import Analysis
from .roi_inputs import build_roi_mask_from_inputs


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _print_kv(key: str, value: Any, indent: int = 2) -> None:
    pad = " " * indent
    print(f"{pad}{key}: {value}")


def _print_dict(d: Dict[str, Any], indent: int = 2) -> None:
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            _print_dict(v, indent=indent + 2)
        elif isinstance(v, list):
            print(f"{pad}{k}:")
            _print_list(v, indent=indent + 2)
        else:
            print(f"{pad}{k}: {v}")


def _print_list(items: List[Any], indent: int = 2) -> None:
    pad = " " * indent
    for i, v in enumerate(items):
        if isinstance(v, dict):
            print(f"{pad}- [{i}]")
            _print_dict(v, indent=indent + 2)
        elif isinstance(v, list):
            print(f"{pad}- [{i}] (list)")
            _print_list(v, indent=indent + 2)
        else:
            print(f"{pad}- [{i}] {v}")


def _normalize_layers(meta_layers: Any) -> List[Dict[str, Any]]:
    """Normalize layer metadata into a list of dicts."""
    if isinstance(meta_layers, list):
        return [x for x in meta_layers if isinstance(x, dict)]
    if isinstance(meta_layers, dict):
        maybe_list = meta_layers.get("layers")
        if isinstance(maybe_list, list):
            return [x for x in maybe_list if isinstance(x, dict)]
    return []


def _sum_numeric(layers: List[Dict[str, Any]], key: str) -> Optional[int]:
    vals: List[int] = []
    for layer in layers:
        v = layer.get(key)
        if isinstance(v, (int, np.integer)):
            vals.append(int(v))
    if not vals:
        return None
    return int(sum(vals))


def _print_layers_hierarchical(layers_raw: Any, indent: int = 2) -> None:
    layers = _normalize_layers(layers_raw)

    pad = " " * indent
    if not layers:
        if layers_raw is None:
            print(f"{pad}(no layer metadata found)")
        else:
            print(f"{pad}(unrecognized layers format; printing raw)")
            if isinstance(layers_raw, dict):
                _print_dict(layers_raw, indent=indent + 2)
            elif isinstance(layers_raw, list):
                _print_list(layers_raw, indent=indent + 2)
            else:
                print(f"{pad}  {layers_raw}")
        return

    for i, layer in enumerate(layers):
        layer_type = layer.get("type", "unknown")
        print(f"{pad}- layer[{i}] ({layer_type})")
        _print_dict(layer, indent=indent + 2)


def _get_summary_from_meta(meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # New schema
    result = meta.get("result")
    if isinstance(result, dict):
        s = result.get("counts_summary")
        if isinstance(s, dict):
            if isinstance(s.get("active"), dict):
                active = dict(s["active"])
                if isinstance(s.get("summary_scope"), str):
                    active.setdefault("summary_scope", s["summary_scope"])
                if isinstance(s.get("global"), dict):
                    active["global"] = s["global"]
                if isinstance(s.get("regional"), dict):
                    active["regional"] = s["regional"]
                return active
            return s
    # Backward compatibility
    s_old = meta.get("analysis_summary")
    return s_old if isinstance(s_old, dict) else None


def _infer_summary_scope(meta: Dict[str, Any]) -> str:
    summary = _get_summary_from_meta(meta)
    if isinstance(summary, dict) and isinstance(summary.get("summary_scope"), str):
        return str(summary["summary_scope"])

    inputs = meta.get("inputs")
    if isinstance(inputs, dict):
        shell = inputs.get("shell")
        if isinstance(shell, dict):
            n_req = shell.get("n_points_requested")
            n_used = shell.get("n_points_used")
            shell_meta = shell.get("meta")
            if isinstance(shell_meta, dict) and "roi_mode" in shell_meta:
                return "regional"
            if isinstance(n_req, (int, np.integer)) and isinstance(n_used, (int, np.integer)):
                if int(n_used) < int(n_req):
                    return "regional"
    return "global"


def _summary_counts_view(counts: np.ndarray, meta: Dict[str, Any], scope: str) -> np.ndarray:
    if scope != "regional":
        return counts
    result = meta.get("result")
    if not isinstance(result, dict):
        return counts

    idx = result.get("summary_point_indices")
    if isinstance(idx, list) and len(idx) > 0:
        idx_arr = np.asarray(idx, dtype=np.int64)
        return counts[:, idx_arr]

    mask = result.get("summary_point_mask")
    if isinstance(mask, list) and len(mask) == counts.shape[1]:
        mask_arr = np.asarray(mask, dtype=bool)
        return counts[:, mask_arr]

    return counts


def _print_analysis_summary(ao: Analysis) -> None:
    """Print summary using meta if present; otherwise compute from counts."""
    counts = ao.counts
    meta = ao.meta if isinstance(ao.meta, dict) else {"meta": ao.meta}

    _print_section("ANALYSIS MATRIX SUMMARY")
    scope = _infer_summary_scope(meta)
    _print_kv("summary_scope", scope)
    _print_kv("shape (Nt, Np)", counts.shape)
    _print_kv("dtype", counts.dtype)

    summary = _get_summary_from_meta(meta)
    if summary is not None:
        # Prefer stored values
        if "min" in summary:
            _print_kv("min count", summary["min"])
        if "max" in summary:
            _print_kv("max count", summary["max"])
        if "mean" in summary:
            _print_kv("mean count", summary["mean"])
        # Optional extras if you store them
        if "nonzero_fraction" in summary:
            _print_kv("nonzero_fraction", summary["nonzero_fraction"])
    else:
        # Fallback: compute (may scan memmap)
        counts_view = _summary_counts_view(np.asarray(counts), meta, scope)
        _print_kv("min count", int(np.min(counts_view)))
        _print_kv("max count", int(np.max(counts_view)))
        _print_kv("mean count", f"{float(np.mean(counts_view)):.3f}")
        _print_kv("nonzero_fraction", float(np.count_nonzero(counts_view) / counts_view.size))


def _print_time_grid(ao: Analysis) -> None:
    times_s = ao.times_s
    Nt = len(times_s)
    meta = ao.meta if isinstance(ao.meta, dict) else {"meta": ao.meta}
    grid = meta.get("grid") if isinstance(meta.get("grid"), dict) else {}

    _print_section("TIME GRID")
    _print_kv("dt_s", grid.get("dt_s", float(times_s[1] - times_s[0]) if Nt > 1 else None))
    _print_kv("t_start_s", grid.get("t_start_s", float(times_s[0]) if Nt > 0 else None))
    _print_kv("t_end_s", grid.get("t_end_s", float(times_s[-1]) if Nt > 0 else None))
    _print_kv("Nt", Nt)


def _print_ground_grid(ao: Analysis) -> None:
    lat_deg = ao.lat_deg
    lon_deg = ao.lon_deg
    Np = len(lat_deg)

    _print_section("SHELL POINT GRID")
    meta = ao.meta if isinstance(ao.meta, dict) else {"meta": ao.meta}
    grid = meta.get("grid") if isinstance(meta.get("grid"), dict) else {}
    inputs = meta.get("inputs") if isinstance(meta.get("inputs"), dict) else {}
    shell_inputs = inputs.get("shell") if isinstance(inputs.get("shell"), dict) else {}
    shell_meta = shell_inputs.get("meta") if isinstance(shell_inputs.get("meta"), dict) else {}

    lat_range = grid.get("lat_deg_range")
    lon_range = grid.get("lon_deg_range")
    shell_id = grid.get("shell_id", ao.shell_id)
    shell_mode = grid.get("analysis_shell_mode", shell_inputs.get("analysis_shell_mode"))
    point_method = shell_meta.get("point_generation_method")
    roi_mode = shell_meta.get("roi_mode")
    countries = shell_meta.get("countries")

    if isinstance(lat_range, list) and len(lat_range) == 2:
        _print_kv("lat_deg range", f"[{lat_range[0]:.2f}, {lat_range[1]:.2f}]")
    else:
        _print_kv("lat_deg range", f"[{lat_deg.min():.2f}, {lat_deg.max():.2f}]")

    if isinstance(lon_range, list) and len(lon_range) == 2:
        _print_kv("lon_deg range", f"[{lon_range[0]:.2f}, {lon_range[1]:.2f}]")
    else:
        _print_kv("lon_deg range", f"[{lon_deg.min():.2f}, {lon_deg.max():.2f}]")

    if shell_id is not None:
        _print_kv("shell_id (compatibility key)", shell_id)
    if isinstance(shell_mode, str) and shell_mode:
        _print_kv("analysis_shell_mode", shell_mode)
    if isinstance(point_method, str) and point_method:
        _print_kv("point_generation_method", point_method)
    if isinstance(roi_mode, str) and roi_mode:
        _print_kv("roi_mode", roi_mode)
    if isinstance(countries, list) and countries:
        _print_kv("countries", ", ".join(str(c) for c in countries))
    _print_kv("Np", Np)


def _print_multi_country_shell_summary(ao: Analysis) -> None:
    meta = ao.meta if isinstance(ao.meta, dict) else {"meta": ao.meta}
    inputs = meta.get("inputs") if isinstance(meta.get("inputs"), dict) else {}
    shell_inputs = inputs.get("shell") if isinstance(inputs.get("shell"), dict) else {}
    shell_meta = shell_inputs.get("meta") if isinstance(shell_inputs.get("meta"), dict) else {}

    countries = shell_meta.get("countries")
    if not isinstance(countries, list) or len(countries) <= 1:
        return

    analysis_shell_mode = shell_inputs.get("analysis_shell_mode", shell_meta.get("analysis_shell_mode"))
    if not isinstance(analysis_shell_mode, str) or not analysis_shell_mode:
        analysis_shell_mode = "full"

    _print_section("PER-COUNTRY SHELL MASK SUMMARY")
    for country in countries:
        country_mask, summary = build_roi_mask_from_inputs(
            np.asarray(ao.lat_deg, dtype=np.float64),
            np.asarray(ao.lon_deg, dtype=np.float64),
            countries=[str(country)],
            analysis_shell_mode=analysis_shell_mode,
        )
        country_mask = np.asarray(country_mask, dtype=bool)
        if country_mask.ndim != 1 or country_mask.shape[0] != ao.counts.shape[1]:
            continue
        if not np.any(country_mask):
            print(f"  {country}: no shell points selected")
            continue

        counts_view = np.asarray(ao.counts[:, country_mask])
        print(f"  {country}")
        _print_kv("roi_points", summary.get("roi_count"), indent=4)
        _print_kv("min count", int(np.min(counts_view)), indent=4)
        _print_kv("max count", int(np.max(counts_view)), indent=4)
        _print_kv("mean count", f"{float(np.mean(counts_view)):.3f}", indent=4)
        _print_kv("nonzero_fraction", f"{float(np.count_nonzero(counts_view) / counts_view.size):.6f}", indent=4)


def _print_metadata(ao: Analysis, *, advanced: bool = False) -> None:
    meta = ao.meta if isinstance(ao.meta, dict) else {"meta": ao.meta}

    _print_section("METADATA / PROVENANCE")

    run = meta.get("run")
    if isinstance(run, dict):
        print("run:")
        _print_dict(run, indent=2)

    inputs = meta.get("inputs")
    if isinstance(inputs, dict):
        layers_raw = inputs.get("layer_specs")
        layers_norm = _normalize_layers(layers_raw)
        print("\ninputs.layers:")
        _print_kv("total_layers", len(layers_norm), indent=2)
        total_planes = _sum_numeric(layers_norm, "n_planes")
        total_sats = _sum_numeric(layers_norm, "n_total")
        if total_planes is not None:
            _print_kv("total_planes", total_planes, indent=2)
        if total_sats is not None:
            _print_kv("total_satellites", total_sats, indent=2)
        _print_layers_hierarchical(layers_raw, indent=2)
        for top_key in ["shell", "sim", "coverage", "earth"]:
            if top_key in inputs:
                print(f"\ninputs.{top_key}:")
                if top_key == "shell" and isinstance(inputs[top_key], dict):
                    shell_info = dict(inputs[top_key])
                    shell_meta = shell_info.get("meta")
                    if isinstance(shell_meta, dict):
                        shell_meta = dict(shell_meta)
                        countries = shell_meta.get("countries")
                        components = shell_meta.get("country_components")
                        if isinstance(countries, list) and countries:
                            shell_meta["countries"] = ", ".join(str(c) for c in countries)
                        if isinstance(components, list) and components:
                            shell_meta["country_components"] = [
                                {
                                    "country_match": comp.get("country_match", comp.get("country_query", "unknown")),
                                    "ISO_A3": comp.get("ISO_A3"),
                                    "lat_min_deg": comp.get("lat_min_deg"),
                                    "lat_max_deg": comp.get("lat_max_deg"),
                                    "lon_min_deg": comp.get("lon_min_deg"),
                                    "lon_max_deg": comp.get("lon_max_deg"),
                                    "n_roi": comp.get("n_roi"),
                                }
                                for comp in components
                                if isinstance(comp, dict)
                            ]
                        shell_info["meta"] = shell_meta
                    _print_dict(shell_info, indent=2)
                elif isinstance(inputs[top_key], dict):
                    _print_dict(inputs[top_key], indent=2)
                else:
                    _print_kv(top_key, inputs[top_key], indent=2)
        if advanced:
            for top_key in ["elements0", "physical"]:
                if top_key in inputs:
                    print(f"\ninputs.{top_key}:")
                    if isinstance(inputs[top_key], dict):
                        _print_dict(inputs[top_key], indent=2)
                    else:
                        _print_kv(top_key, inputs[top_key], indent=2)
    else:
        # Backward compatibility for old metadata shape
        print("inputs.layers:")
        _print_layers_hierarchical(meta.get("layers"), indent=2)
        for top_key in ["shell_meta", "sim", "coverage", "earth"]:
            if top_key in meta:
                print(f"\n{top_key}:")
                if isinstance(meta[top_key], dict):
                    _print_dict(meta[top_key], indent=2)
                else:
                    _print_kv(top_key, meta[top_key], indent=2)

    if "propagator_meta" in meta:
        print("\npropagator_meta:")
        if isinstance(meta["propagator_meta"], dict):
            _print_dict(meta["propagator_meta"], indent=2)
        else:
            _print_kv("propagator_meta", meta["propagator_meta"], indent=2)

    remaining = {
        k: v for k, v in meta.items()
        if k not in {"run", "inputs", "grid", "result", "propagator_meta", "layers", "shell_meta", "sim", "coverage", "earth", "analysis_summary"}
    }
    if remaining:
        print("\nadditional metadata:")
        _print_dict(remaining, indent=2)

    print("\n[End of metadata]")


def report(
    ao: Analysis,
    *,
    print_meta: bool = False,
    advanced_output: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Print a structured report of an Analysis object.

    - Analysis Matrix Summary is always first.
    - Summary values are pulled from meta['result']['counts_summary'] when available
      (with compatibility fallback to legacy meta['analysis_summary']).

    Returns None. This function is print-oriented.
    """
    if not isinstance(ao, Analysis):
        raise TypeError("report(): input must be an Analysis")

    _print_analysis_summary(ao)
    _print_time_grid(ao)
    _print_ground_grid(ao)
    _print_multi_country_shell_summary(ao)
    
    if print_meta:
        _print_metadata(ao, advanced=advanced_output)

    print("\n[End of report]")

    return None
