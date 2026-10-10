"""SCLP constellation optimization after Lee et al. (2020), with exact secular-J2 repeat ground tracks.

Layout:
  seed_families.py  plan the synchronized N_D-day seed families, anchored top-down from target latitudes
  rgt_slots.py      closure solve (semi-major axis) and the L time-shifted slots of one seed
  seed_pipeline.py  seed pool (family x RAAN), parallel seed simulation, greedy build/trim filter
  param_v.py        circulant constraint matrix param_V from seed access profiles
  milp_solver.py    Gurobi MILP solve, result CSV/JSON writer
  run_pipeline.py   run_sclp(): the whole flow behind lee_test2 / cloud_sweep --local / batch_worker
"""
from .layer import Layer
from .milp_solver import SCLPResult, build_orb_elems_table, print_orb_table, save_orb_table_csv, save_sclp_result, solve_sclp
from .param_v import build_param_V
from .rgt_slots import a_km_j2_rgt, lee_j2_rates, rgt_closure_residual, rgt_common_ground_track_layer, seed_layer
from .run_pipeline import run_sclp, write_failure_record
from .seed_families import footprint_reach_deg, n_distinct_slots, plan_seed_families
from .seed_pipeline import (build_seed_pool, family_satellite_counts, filter_seed_pool, seed_run_config,
                            simulate_seed_pool)

__all__ = [
    "Layer", "SCLPResult", "a_km_j2_rgt", "build_orb_elems_table", "build_param_V", "build_seed_pool",
    "family_satellite_counts", "filter_seed_pool", "footprint_reach_deg", "lee_j2_rates", "n_distinct_slots",
    "plan_seed_families", "print_orb_table", "rgt_closure_residual", "rgt_common_ground_track_layer",
    "run_sclp", "save_orb_table_csv", "save_sclp_result", "seed_layer", "seed_run_config", "simulate_seed_pool",
    "solve_sclp", "write_failure_record",
]
