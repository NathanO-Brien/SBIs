"""
verify_frames.py

Standalone, self-contained verification of the ECI<->ECEF rotation direction
in src/sbi_coverage/core/frames.py.

No simulation code, no MILP, no coverage logic — just the frame transform
tested against three physical facts that do not depend on any convention
choice in this repo:

  TEST 1 — Geostationary satellite.
      A satellite in a circular, equatorial, prograde orbit whose period
      equals one sidereal day is DEFINED by appearing motionless in the
      Earth-fixed frame. Its ECEF position must be constant.

  TEST 2 — Ground station.
      A point fixed on the rotating Earth has constant ECEF coordinates by
      definition. Its ECI position at time t is known in closed form:
      the Earth rotates eastward (+Z, counterclockwise viewed from the
      north pole), so the point's ECI longitude increases with time.
      Feeding that known ECI position into eci_to_ecef() must return the
      original fixed ECEF coordinates.

  TEST 3 — LEO ground-track drift direction.
      For a prograde LEO orbit (period ~90 min), each successive ascending
      node crosses the equator WEST of the previous one (the Earth rotates
      eastward underneath the orbit). This westward node drift is a basic,
      observable fact of spaceflight (e.g., ISS ground tracks).

Each test is run twice:
  (a) with the repo's current frames.py functions, and
  (b) with the TRANSPOSED rotation (the alternative sign convention),
so the two candidate conventions can be compared side by side.

Run:
    python scripts/verify_frames.py
"""
from __future__ import annotations

import numpy as np

from sbi_coverage.core.config import EarthConstants
from sbi_coverage.core.frames import eci_to_ecef, rot_z

earth = EarthConstants()
OMEGA = earth.omega_earth_rad_s
MU    = earth.mu_km3_s2


def eci_to_ecef_A(points_eci_km: np.ndarray, t_s: float) -> np.ndarray:
    """Convention A: points @ rot_z(theta)   (the repo's pre-July-8 code)."""
    R = rot_z(OMEGA * t_s)
    return points_eci_km @ R


def eci_to_ecef_B(points_eci_km: np.ndarray, t_s: float) -> np.ndarray:
    """Convention B: points @ rot_z(theta).T (the proposed fix)."""
    R = rot_z(OMEGA * t_s)
    return points_eci_km @ R.T


def which_convention_is_live() -> str:
    """Report which convention the repo's eci_to_ecef currently implements."""
    probe = np.array([[7000.0, 1234.0, 42.0]])
    t = 5000.0
    live = eci_to_ecef(probe, t, earth)
    if np.allclose(live, eci_to_ecef_A(probe, t)):
        return "A  (points @ R)"
    if np.allclose(live, eci_to_ecef_B(probe, t)):
        return "B  (points @ R.T)"
    return "neither (unexpected)"


def lon_deg_of(xy: np.ndarray) -> float:
    """Longitude (deg) of a position vector's x,y components."""
    return float(np.rad2deg(np.arctan2(xy[1], xy[0])))


def run_all() -> None:
    print(f"repo eci_to_ecef currently implements convention: {which_convention_is_live()}")
    print()
    print("=" * 72)
    print("TEST 1 — GEOSTATIONARY SATELLITE")
    print("=" * 72)
    print("""
A geostationary satellite orbits prograde (same direction Earth spins,
counterclockwise viewed from above the north pole) with period equal to
one sidereal day. 'Geostationary' MEANS it stays put in the Earth-fixed
frame — TV dishes do not track. Its ECEF position must be constant.

The satellite's inertial (ECI) position is computed from first principles
below — no repo code involved. Only the final ECI->ECEF conversion uses
the function under test.
""")
    a_geo = (MU / OMEGA**2) ** (1.0 / 3.0)      # mean motion == OMEGA
    n = np.sqrt(MU / a_geo**3)
    print(f"  a_geo = {a_geo:.3f} km   (textbook value ~42164.17 km)")
    print(f"\n  {'t (hr)':>7} | {'convention A ECEF lon':>22} | {'convention B ECEF lon':>22}")
    print("  " + "-" * 60)
    for t in [0.0, 2 * 3600.0, 4 * 3600.0, 6 * 3600.0]:
        th = n * t                               # satellite's own ECI angle
        r_eci = np.array([[a_geo * np.cos(th), a_geo * np.sin(th), 0.0]])
        lon_A = lon_deg_of(eci_to_ecef_A(r_eci, t)[0])
        lon_B = lon_deg_of(eci_to_ecef_B(r_eci, t)[0])
        print(f"  {t/3600:7.1f} | {lon_A:21.4f}  | {lon_B:21.4f}")
    print("""
  VERDICT CRITERION: the correct transform holds ECEF longitude at 0.0 deg
  for every t. A transform that is backwards shows the longitude sweeping
  at twice Earth's rotation rate (~30 deg/hour).
""")

    print("=" * 72)
    print("TEST 2 — GROUND STATION (fixed point on rotating Earth)")
    print("=" * 72)
    print("""
Take a ground station on the equator at ECEF longitude 0 deg (r = Re, 0, 0).
Earth rotates eastward: counterclockwise viewed from the north pole, i.e.
+OMEGA about +Z. Therefore after time t the station's TRUE inertial (ECI)
position is the fixed vector rotated counterclockwise by +OMEGA*t:

      x_eci = Re*cos(OMEGA*t),   y_eci = +Re*sin(OMEGA*t)

(The + sign on y is the only physics input here: eastward = increasing
inertial longitude. If you doubt this line, check any astrodynamics text
for Earth's rotation direction — everything else below is arithmetic.)

Feeding this true ECI position into eci_to_ecef() must return the original
fixed ECEF coordinates (Re, 0, 0) at every t.
""")
    Re = earth.r_eq_km
    print(f"  {'t (hr)':>7} | {'convention A ECEF':>26} | {'convention B ECEF':>26}")
    print("  " + "-" * 68)
    for t in [0.0, 3 * 3600.0, 6 * 3600.0]:
        th = OMEGA * t
        r_eci_true = np.array([[Re * np.cos(th), Re * np.sin(th), 0.0]])
        p_A = eci_to_ecef_A(r_eci_true, t)[0]
        p_B = eci_to_ecef_B(r_eci_true, t)[0]
        print(f"  {t/3600:7.1f} | ({p_A[0]:9.1f}, {p_A[1]:9.1f}) km | ({p_B[0]:9.1f}, {p_B[1]:9.1f}) km")
    print(f"""
  VERDICT CRITERION: the correct transform returns ({Re:.1f}, 0.0) at
  every t. A backwards transform shows the station drifting around the
  planet at twice Earth's rotation rate.
""")

    print("=" * 72)
    print("TEST 3 — LEO GROUND-TRACK NODE DRIFT DIRECTION")
    print("=" * 72)
    print("""
A prograde LEO satellite (e.g., ISS at ~400 km, period ~92.6 min) crosses
the equator northbound once per orbit. Because Earth rotates eastward
underneath the orbit, each successive northbound equator crossing lands
WEST of the previous one by about (period/sidereal day)*360 deg ~ 23 deg.
This is directly observable on any live ISS tracker.

Below, the satellite's ECI motion is computed from first principles for a
51.6 deg-inclined circular orbit; only the ECI->ECEF step uses the function
under test. The script finds successive northbound equator crossings and
reports the longitude change between them.
""")
    alt = 400.0
    a = Re + alt
    n_leo = np.sqrt(MU / a**3)
    inc = np.deg2rad(51.6)
    period_s = 2 * np.pi / n_leo
    expected_shift = -(period_s / (2 * np.pi / OMEGA)) * 360.0
    print(f"  Orbit period: {period_s/60:.1f} min   expected node shift: {expected_shift:.1f} deg (westward)")

    def node_lons(transform) -> list[float]:
        lons = []
        t_grid = np.arange(0.0, 3.2 * period_s, 1.0)
        prev_z = None
        for t in t_grid:
            u = n_leo * t                        # argument of latitude
            # standard prograde orbit in ECI, RAAN=0
            x = a * (np.cos(u))
            y = a * (np.sin(u) * np.cos(inc))
            z = a * (np.sin(u) * np.sin(inc))
            if prev_z is not None and prev_z < 0.0 <= z:   # northbound equator crossing
                p = transform(np.array([[x, y, z]]), t)[0]
                lons.append(lon_deg_of(p))
            prev_z = z
        return lons

    lon_A = node_lons(lambda p, t: eci_to_ecef_A(p, t))
    lon_B = node_lons(lambda p, t: eci_to_ecef_B(p, t))

    def shifts(lons: list[float]) -> list[float]:
        out = []
        for i in range(1, len(lons)):
            d = lons[i] - lons[i - 1]
            d = (d + 180.0) % 360.0 - 180.0
            out.append(d)
        return out

    print(f"\n  convention A : node longitudes {['%.1f deg' % x for x in lon_A]}")
    print(f"                 successive shifts {['%+.1f deg' % x for x in shifts(lon_A)]}")
    print(f"  convention B : node longitudes {['%.1f deg' % x for x in lon_B]}")
    print(f"                 successive shifts {['%+.1f deg' % x for x in shifts(lon_B)]}")
    print(f"""
  VERDICT CRITERION: real prograde LEO ground tracks drift WESTWARD
  (negative shift, ~{expected_shift:.0f} deg per orbit — check any ISS tracker).
  A backwards transform shows them drifting EASTWARD instead.
""")

    print("=" * 72)
    print("Summary: all three tests share one property — the correct rotation")
    print("direction is fixed by observable physics (geostationary TV dishes,")
    print("ground stations, ISS ground tracks), not by any convention in this")
    print("repo or in app.js. Whichever column passes all three is correct.")
    print("=" * 72)


if __name__ == "__main__":
    run_all()
