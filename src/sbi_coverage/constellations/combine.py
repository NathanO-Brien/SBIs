from __future__ import annotations

import numpy as np
from ..core.elements import OrbitalElements, SatellitePhysical
from .layer import Layer


def combine_layers(layers: list[Layer]) -> tuple[OrbitalElements, SatellitePhysical, list[dict]]:
    """
    Concatenate multiple Layer objects into one constellation.

    Returns
    -------
    elems0 : OrbitalElements
        Flattened per-satellite arrays.
    phys : SatellitePhysical
        Flattened per-satellite physical parameters.
    layer_specs : list[dict]
        Exact input specs captured by each layer generator, in the same order.
    """
    if len(layers) == 0:
        raise ValueError("combine_layers() received an empty layer list.")

    a = np.concatenate([L.elems.a_km for L in layers])
    e = np.concatenate([L.elems.e for L in layers])
    i = np.concatenate([L.elems.i_rad for L in layers])
    raan = np.concatenate([L.elems.raan_rad for L in layers])
    argp = np.concatenate([L.elems.argp_rad for L in layers])
    M0 = np.concatenate([L.elems.M0_rad for L in layers])
    bc = np.concatenate([L.phys.bc_kg_m2 for L in layers])

    elems0 = OrbitalElements(a_km=a, e=e, i_rad=i, raan_rad=raan, argp_rad=argp, M0_rad=M0)
    phys = SatellitePhysical(bc_kg_m2=bc)

    layer_specs = [dict(L.spec) for L in layers]
    return elems0, phys, layer_specs
