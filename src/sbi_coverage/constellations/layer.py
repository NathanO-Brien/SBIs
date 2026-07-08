from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

from ..core.elements import OrbitalElements, SatellitePhysical

@dataclass(frozen=True)
class Layer:
    elems: OrbitalElements
    phys: SatellitePhysical
    spec: Dict[str, Any]   # original inputs + type
