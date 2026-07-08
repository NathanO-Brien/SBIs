"""SCLP constellation optimization pipeline after Lee et al. (2020):
RGT slot construction, access-profile assembly, and MILP solve.
"""
from .optimal_altitude import optimal_sat_altitude_km, footprint_half_angles_deg
from .rgt_ratio import candidate_rgt_ratios
from .rgt_slots import a_km_j2_rgt, rgt_common_ground_track_layer, rgt_subconstellation_layers, seed_layer
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
    "candidate_rgt_ratios",
    "a_km_j2_rgt",
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
