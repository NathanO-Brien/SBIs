"""Minimal per-satellite physical record used only by the legacy generators (drag/bc was removed from core)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SatellitePhysical:
    bc_kg_m2: Any = 80.0
    mass_kg: Any = None
    area_m2: Any = None
    cd: Any = None
