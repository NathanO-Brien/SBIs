"""Container for one seed's orbital slots."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from ..core.elements import OrbitalElements


@dataclass(frozen=True)
class Layer:
    """The L orbital slots of one seed: their orbital elements and the generator inputs (spec)
    that produced them."""
    elems: OrbitalElements
    spec: Dict[str, Any]
