import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:
    NUMBA_AVAILABLE = False
    njit = None


def _validate_coverage_inputs(
    r_sat_ecef_km: np.ndarray,
    r_pts_ecef_km: np.ndarray,
    n_hat_ecef: np.ndarray,
    max_range_km: float,
    min_elev_deg: float,
) -> None:
    if r_sat_ecef_km.ndim != 2 or r_sat_ecef_km.shape[1] != 3:
        raise ValueError(f"coverage_counts_ecef(): r_sat_ecef_km must have shape (Ns,3), got {r_sat_ecef_km.shape}")
    if r_pts_ecef_km.ndim != 2 or r_pts_ecef_km.shape[1] != 3:
        raise ValueError(f"coverage_counts_ecef(): r_pts_ecef_km must have shape (Np,3), got {r_pts_ecef_km.shape}")
    if n_hat_ecef.shape != r_pts_ecef_km.shape:
        raise ValueError(
            "coverage_counts_ecef(): n_hat_ecef must have same shape as r_pts_ecef_km, "
            f"got {n_hat_ecef.shape} vs {r_pts_ecef_km.shape}"
        )
    if not np.isfinite(max_range_km) or max_range_km < 0.0:
        raise ValueError("coverage_counts_ecef(): max_range_km must be finite and >= 0")
    if not np.isfinite(min_elev_deg) or min_elev_deg < -90.0 or min_elev_deg > 90.0:
        raise ValueError("coverage_counts_ecef(): min_elev_deg must be finite and in [-90, 90]")
    if not (np.all(np.isfinite(r_sat_ecef_km)) and np.all(np.isfinite(r_pts_ecef_km)) and np.all(np.isfinite(n_hat_ecef))):
        raise ValueError("coverage_counts_ecef(): inputs contain NaN or inf")


if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _coverage_counts_numba(
        r_sat_ecef_km: np.ndarray,   # (Ns, 3)
        r_pts_ecef_km: np.ndarray,   # (Np, 3)
        n_hat_ecef: np.ndarray,      # (Np, 3) precomputed
        max_range2: float,
        min_sin_elev: float,
    ) -> np.ndarray:
        Ns = r_sat_ecef_km.shape[0]
        Np = r_pts_ecef_km.shape[0]
        counts = np.zeros(Np, dtype=np.int32)

        for j in range(Np):
            px = r_pts_ecef_km[j, 0]
            py = r_pts_ecef_km[j, 1]
            pz = r_pts_ecef_km[j, 2]

            nx = n_hat_ecef[j, 0]
            ny = n_hat_ecef[j, 1]
            nz = n_hat_ecef[j, 2]

            c = 0
            for s in range(Ns):
                sx = r_sat_ecef_km[s, 0]
                sy = r_sat_ecef_km[s, 1]
                sz = r_sat_ecef_km[s, 2]

                rx = sx - px
                ry = sy - py
                rz = sz - pz

                rho2 = rx*rx + ry*ry + rz*rz
                if rho2 > max_range2:
                    continue

                rho = rho2 ** 0.5
                if rho < 1e-12:
                    continue

                sin_elev = (rx*nx + ry*ny + rz*nz) / rho
                if sin_elev >= min_sin_elev:
                    c += 1

            counts[j] = c

        return counts


def coverage_counts_ecef(
    r_sat_ecef_km: np.ndarray,
    r_pts_ecef_km: np.ndarray,
    n_hat_ecef: np.ndarray,
    max_range_km: float,
    min_elev_deg: float,
) -> np.ndarray:
    _validate_coverage_inputs(r_sat_ecef_km, r_pts_ecef_km, n_hat_ecef, max_range_km, min_elev_deg)
    max_range2 = float(max_range_km * max_range_km)
    min_sin_elev = float(np.sin(np.deg2rad(min_elev_deg)))

    if NUMBA_AVAILABLE:
        return _coverage_counts_numba(r_sat_ecef_km, r_pts_ecef_km, n_hat_ecef, max_range2, min_sin_elev)

    # fallback (slow)
    counts = np.zeros(r_pts_ecef_km.shape[0], dtype=np.int32)

    for s in range(r_sat_ecef_km.shape[0]):
        rs = r_sat_ecef_km[s, :][None, :]
        rho = rs - r_pts_ecef_km
        rho2 = np.einsum("ij,ij->i", rho, rho)
        in_range = rho2 <= max_range2
        rho_norm = np.sqrt(np.maximum(rho2, 1e-12))
        sin_elev = np.einsum("ij,ij->i", rho, n_hat_ecef) / rho_norm
        above = sin_elev >= min_sin_elev
        counts += (in_range & above).astype(np.int32)

    return counts

if NUMBA_AVAILABLE:
    @njit(cache=True, fastmath=True)
    def _satellite_point_counts_numba(
        r_sat_ecef_km: np.ndarray,   # (Ns, 3)
        r_pts_ecef_km: np.ndarray,   # (Np, 3)
        n_hat_ecef: np.ndarray,      # (Np, 3) precomputed normals
        max_range2: float,
        min_sin_elev: float,
    ) -> np.ndarray:
        """
        Return counts per satellite: how many ROI points each satellite covers at this time.
        Output shape: (Ns,)
        """
        Ns = r_sat_ecef_km.shape[0]
        Np = r_pts_ecef_km.shape[0]
        sat_counts = np.zeros(Ns, dtype=np.int32)

        for s in range(Ns):
            sx = r_sat_ecef_km[s, 0]
            sy = r_sat_ecef_km[s, 1]
            sz = r_sat_ecef_km[s, 2]

            c = 0
            for j in range(Np):
                px = r_pts_ecef_km[j, 0]
                py = r_pts_ecef_km[j, 1]
                pz = r_pts_ecef_km[j, 2]

                nx = n_hat_ecef[j, 0]
                ny = n_hat_ecef[j, 1]
                nz = n_hat_ecef[j, 2]

                rx = sx - px
                ry = sy - py
                rz = sz - pz

                rho2 = rx*rx + ry*ry + rz*rz
                if rho2 > max_range2:
                    continue

                rho = rho2 ** 0.5
                if rho < 1e-12:
                    continue

                sin_elev = (rx*nx + ry*ny + rz*nz) / rho
                if sin_elev >= min_sin_elev:
                    c += 1

            sat_counts[s] = c

        return sat_counts


def satellite_point_counts_ecef(
    r_sat_ecef_km: np.ndarray,
    r_pts_ecef_km: np.ndarray,
    n_hat_ecef: np.ndarray,
    max_range_km: float,
    min_elev_deg: float,
) -> np.ndarray:
    """
    Public wrapper: counts per satellite (Ns,).
    A satellite's count is # of ROI points it covers at this timestep.
    """
    _validate_coverage_inputs(r_sat_ecef_km, r_pts_ecef_km, n_hat_ecef, max_range_km, min_elev_deg)
    max_range2 = float(max_range_km * max_range_km)
    min_sin_elev = float(np.sin(np.deg2rad(min_elev_deg)))

    if NUMBA_AVAILABLE:
        return _satellite_point_counts_numba(r_sat_ecef_km, r_pts_ecef_km, n_hat_ecef, max_range2, min_sin_elev)

    # fallback (slow)
    Ns = r_sat_ecef_km.shape[0]
    Np = r_pts_ecef_km.shape[0]
    sat_counts = np.zeros(Ns, dtype=np.int32)

    for s in range(Ns):
        rs = r_sat_ecef_km[s, :][None, :]  # (1,3)
        rho = rs - r_pts_ecef_km           # (Np,3)
        rho2 = np.einsum("ij,ij->i", rho, rho)
        in_range = rho2 <= max_range2
        rho_norm = np.sqrt(np.maximum(rho2, 1e-12))
        sin_elev = np.einsum("ij,ij->i", rho, n_hat_ecef) / rho_norm
        above = sin_elev >= min_sin_elev

        sat_counts[s] = int(np.sum(in_range & above))

    return sat_counts
