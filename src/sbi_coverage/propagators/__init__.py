"""Propagator registry + loader.

run_simulation() selects a propagator by string name; each propagator is a module implementing
init_state() and step_state_to_eci_positions(), so the coverage loop stays independent of the
motion model. Only the secular-J2 ``Nominal_Propagator`` exists.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Dict

# Map user-facing names -> importable module paths
PROPAGATOR_REGISTRY: Dict[str, str] = {
    "Nominal_Propagator": "sbi_coverage.propagators.nominal_propagator",
}

_REQUIRED_FUNCS = ("init_state", "step_state_to_eci_positions")


def get_propagator(name: str) -> ModuleType:
    """Load a propagator module by registry name and validate it implements the required interface."""
    if name not in PROPAGATOR_REGISTRY:
        valid = ", ".join(sorted(PROPAGATOR_REGISTRY.keys()))
        raise ValueError(f"Unknown propagator '{name}'. Valid options: {valid}")

    mod_path = PROPAGATOR_REGISTRY[name]
    mod = importlib.import_module(mod_path)

    missing = [fn for fn in _REQUIRED_FUNCS if not hasattr(mod, fn)]
    if missing:
        raise AttributeError(f"Propagator module '{mod_path}' is missing required functions: {missing}")

    return mod
