"""Constellation layer container."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from ..core.elements import OrbitalElements, SatellitePhysical


@dataclass(frozen=True)
class Layer:
    """One constellation layer: orbital elements, physical properties, and
    the generator inputs (spec) that produced it."""
    elems: OrbitalElements
    phys: SatellitePhysical
    spec: Dict[str, Any]   # original generator inputs + layer type
