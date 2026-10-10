"""SCLP constellation optimization pipeline after Lee et al. (2020):
RGT slot construction, access-profile assembly, and MILP solve.
"""
from .optimal_altitude import optimal_sat_altitude_km, footprint_half_angles_deg
from .seed_families import plan_seed_families, solve_member, footprint_reach_deg, ladder_from_pin, resolve_alt_bounds, n_distinct_slots
from .rgt_slots import lee_j2_rates, rgt_closure_residual, a_km_j2_rgt, rgt_common_ground_track_layer, rgt_subconstellation_layers, seed_layer
from .seed_pipeline import (
    build_seed_pool, simulate_seed_pool, filter_seed_pool, family_satellite_counts, seed_run_config,
    greedy_select_seeds, trim_seeds, rarity_weights,
)
from .param_v import build_param_V, param_V_summary
from .milp_solver import (
    solve_sclp,
    build_orb_elems_table,
    save_orb_table_csv,
    save_sclp_result,
    print_orb_table,
    SCLPResult,
)

__all__ = [
    "optimal_sat_altitude_km",
    "footprint_half_angles_deg",
    "a_km_j2_rgt",
    "lee_j2_rates",
    "rgt_closure_residual",
    "plan_seed_families",
    "solve_member",
    "footprint_reach_deg",
    "ladder_from_pin",
    "build_seed_pool",
    "simulate_seed_pool",
    "filter_seed_pool",
    "family_satellite_counts",
    "seed_run_config",
    "greedy_select_seeds",
    "trim_seeds",
    "rarity_weights",
    "resolve_alt_bounds",
    "n_distinct_slots",
    "rgt_common_ground_track_layer",
    "rgt_subconstellation_layers",
    "seed_layer",
    "build_param_V",
    "param_V_summary",
    "solve_sclp",
    "build_orb_elems_table",
    "save_orb_table_csv",
    "save_sclp_result",
    "print_orb_table",
    "SCLPResult",
]
