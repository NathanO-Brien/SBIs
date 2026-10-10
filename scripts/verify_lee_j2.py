"""Verification of Lee's exact-J2 model and the synchronized seed-family planner.

Run:  python scripts/verify_lee_j2.py

Checks (all must pass before any optimization):
  1. Lee (2020) Example 5 reproduced (8:1 @ 70 deg, 6:1 @ 47.915 deg).
  2. Family closure residuals, exactly shared T_r, shared L / dt.
  3. Ground-track closure of one satellite after exactly T_r (J2 on, Lee Mdot),
     with a negative control: the old Keplerian-M propagator must FAIL.
  4. Circulant exactness: slot k's ECEF track == slot 0's track delayed by k*dt
     (including wrap) and access profile v_k == roll(v0, k) on a real target set.
  5. Families table for the principal target sets (informational).
"""
from __future__ import annotations

import sys
from math import gcd
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sbi_coverage.core.config import EarthConstants, SimConfig, CoverageConfig
from sbi_coverage.core.elements import OrbitalElements
from sbi_coverage.core.frames import eci_to_ecef
from sbi_coverage.core.shell import Shell
from sbi_coverage.core.simulate import run_simulation
from sbi_coverage.Lee import (
    a_km_j2_rgt, lee_j2_rates, rgt_closure_residual, plan_seed_families,
    rgt_common_ground_track_layer,
)
from sbi_coverage.propagators import nominal_propagator as NP

EARTH = EarthConstants()
# Principal-scenario engagement: R_max ~ 746 km, so the altitude upper bound is h_s + R_max ~ 946 km.
COV = CoverageConfig(earth=EARTH, detection_time_s=15.0, decision_time_s=0.0, target_missile_burnout_time_s=170.0,
                     burnout_velocity_km_s=6.0, max_accel_g=10.0, intercept_alt_km=200.0, min_elev_deg=0.0)
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}  {detail}")
    if not ok:
        FAILS.append(name)


# Principal target sets: lat ranges from Missile Threat.xlsx location tabs
# (only latitude matters for the planner; longitudes are representative).
TARGET_SETS = {
    "NK": [["Sohae", 39.660, 124.705], ["Hoejung-ni", 41.370, 126.914], ["Winter Palace", 38.916, 125.922]],
    "Iran": [["Imam Khomeini", 35.239, 53.950], ["Shahroud", 36.201, 55.334]],
    "Russia": [["Dombarovsky", 51.062, 60.212], ["Nizhny Tagil", 58.230, 60.677], ["Tatishchevo", 51.879, 45.337]],
    "China (Yumen/Hami/Ordos)": [["Yumen", 40.269, 96.579], ["Hami", 42.328, 92.492], ["Ordos", 40.113, 108.104]],
    "China (Yibin..Jilin)": [["Yibin", 28.761, 104.791], ["Jilin", 43.936, 126.451], ["Hancheng", 35.388, 110.375]],
}


def _propagate_positions(elems: OrbitalElements, n_steps: int, dt_s: float, use_j2: bool, mdot: bool):
    """ECEF positions at t = 0, dt, ..., n_steps*dt via the repo propagator.

    mdot=False reproduces the OLD propagator (Keplerian n for M even with J2 on)
    by temporarily swapping the Lee Mdot function for plain mean motion.
    """
    sim = SimConfig(horizon_s=n_steps * dt_s, dt_s=dt_s, use_j2=use_j2)
    state = NP.init_state(elems0=elems, sim=sim, earth=EARTH)
    orig = NP.lee_mean_anomaly_rate_rad_s
    if not mdot:
        NP.lee_mean_anomaly_rate_rad_s = lambda a, e, i, earth: NP.mean_motion_rad_s(a, earth.mu_km3_s2)
    try:
        out = []
        for k in range(n_steps + 1):
            dt = 0.0 if k == 0 else dt_s
            r = NP.step_state_to_eci_positions(state=state, dt_s=dt, sim=sim, earth=EARTH)
            out.append(eci_to_ecef(r, k * dt_s, EARTH))
    finally:
        NP.lee_mean_anomaly_rate_rad_s = orig
    return np.stack(out)  # (n_steps+1, Ns, 3)


def test_lee_example5() -> None:
    print("\n1. Lee (2020) Example 5")
    for N_P, i_deg, alt_ref in ((8, 70.0, 4149.0), (6, 47.915, 6380.0)):
        a = a_km_j2_rgt(N_P, 1, i_deg, EARTH)
        Om, _, _ = lee_j2_rates(a, i_deg, EARTH)
        T_r = 2 * np.pi / (EARTH.omega_earth_rad_s - Om)
        check(f"{N_P}:1 @ {i_deg} deg altitude", abs(a - EARTH.r_eq_km - alt_ref) < 1.0,
              f"alt={a - EARTH.r_eq_km:.3f} km (Lee {alt_ref:.0f})")
        check(f"{N_P}:1 @ {i_deg} deg T_r, L", abs(T_r - 86024.0) < 1.0 and round(T_r / 120.0) == 717,
              f"T_r={T_r:.3f} s, L={round(T_r / 120.0)} (Lee ~86024 s, 717)")


def test_families() -> dict:
    print("\n2. Family closure and exact T_r sharing")
    plans = {}
    for name, tp in TARGET_SETS.items():
        plan = plan_seed_families(tp, earth=EARTH, cov=COV)
        plans[name] = plan
        Trs = np.array([f["T_r_s"] for f in plan["families"]])
        res = max(abs(f["closure_residual_rad_s"]) for f in plan["families"])
        check(f"{name}: closure residual < 1e-12 rad/s", res < 1e-12, f"max={res:.1e}")
        check(f"{name}: T_r shared to < 1e-6 s", np.ptp(Trs) < 1e-6, f"spread={np.ptp(Trs):.2e} s")
        check(f"{name}: anchor is highest inclination / lowest altitude",
              all(f["inc_deg"] <= plan["anchor"]["inc_deg"] and f["alt_km"] >= plan["anchor"]["alt_km"]
                  for f in plan["families"]))
        check(f"{name}: altitudes within bounds",
              all(plan["alt_bounds_km"][0] <= f["alt_km"] <= plan["alt_bounds_km"][1] for f in plan["families"]))
        check(f"{name}: reducible rungs flagged consistently (N_P:N_D not in lowest terms)",
              all(f["reducible"] == (gcd(f["N_P"], f["N_D"]) != 1) for f in plan["families"]))
        check(f"{name}: RAAN window = 360/reduced N_P",
              all(abs(f["raan_interval_deg"] - 360.0 / (f["N_P"] // gcd(f["N_P"], f["N_D"]))) < 1e-9 for f in plan["families"]))
        L, dt = plan["L"], plan["dt_s"]
        check(f"{name}: L*dt == T_r", abs(L * dt - plan["T_r_s"]) < 1e-6, f"L={L} dt={dt:.4f}")
        check(f"{name}: L is a multiple of every repeat factor g (slot k+L/g exactly repeats slot k)",
              all(L % f["repeat_factor"] == 0 for f in plan["families"]), f"L={L}")
    return plans


def _elems_from_layer(layer, idx) -> OrbitalElements:
    e = layer.elems
    return OrbitalElements(a_km=e.a_km[idx], e=e.e[idx], i_rad=e.i_rad[idx], raan_rad=e.raan_rad[idx],
                           argp_rad=e.argp_rad[idx], M0_rad=e.M0_rad[idx])


def _layer_for(plan, fam, raan_deg=0.0):
    return rgt_common_ground_track_layer(
        inc_deg=fam["inc_deg"], N_P=fam["N_P"], N_D=fam["N_D"], L=plan["L"], earth=EARTH,
        raan0_deg=raan_deg, a_km=fam["a_km"], dt_s=plan["dt_s"],
    )


def test_track_closure(plans) -> None:
    print("\n3. Ground-track closure after exactly T_r (1 m tolerance); negative control = old Keplerian M")
    for name, plan in plans.items():
        for fam in plan["families"]:
            layer = _layer_for(plan, fam, raan_deg=17.0)
            el = _elems_from_layer(layer, slice(0, 1))
            pos = _propagate_positions(el, plan["L"], plan["dt_s"], use_j2=True, mdot=True)
            err_new = np.linalg.norm(pos[-1, 0] - pos[0, 0]) * 1000.0
            pos_old = _propagate_positions(el, plan["L"], plan["dt_s"], use_j2=True, mdot=False)
            err_old = np.linalg.norm(pos_old[-1, 0] - pos_old[0, 0]) * 1000.0
            tag = f"{name} N_P={fam['N_P']} i={fam['inc_deg']:.2f}"
            check(f"{tag}: closure with Lee Mdot", err_new < 1.0, f"{err_new:.4f} m")
            check(f"{tag}: old Keplerian-M propagator fails (control)", err_old > 1000.0, f"{err_old / 1000.0:.1f} km")


def test_circulant(plans) -> None:
    print("\n4. Circulant exactness (slot k == slot 0 delayed by k*dt) and access-profile roll")
    rng = np.random.default_rng(0)
    for name, plan in plans.items():
        L, dt = plan["L"], plan["dt_s"]
        tp = TARGET_SETS[name]
        lat = np.array([p[1] for p in tp]); lon = np.array([p[2] for p in tp])
        for fam in plan["families"][:2]:
            layer = _layer_for(plan, fam, raan_deg=5.0)
            ks = sorted({1, 2, 3, L // 3, L - 2, L - 1, *rng.integers(1, L, 3).tolist()})
            idx = np.array([0] + ks)
            el = OrbitalElements(*(getattr(layer.elems, f)[idx] for f in
                                   ("a_km", "e", "i_rad", "raan_rad", "argp_rad", "M0_rad")))
            pos = _propagate_positions(el, L, dt, use_j2=True, mdot=True)  # (L+1, 1+len(ks), 3)
            worst = 0.0
            for col, k in enumerate(ks, start=1):
                # slot k at step t == slot 0 at step (t - k) mod L
                t = np.arange(L)
                ref = pos[(t - k) % L, 0]
                d = np.linalg.norm(pos[t, col] - ref, axis=1).max() * 1000.0
                worst = max(worst, d)
            tag = f"{name} N_P={fam['N_P']} i={fam['inc_deg']:.2f}"
            check(f"{tag}: slot-k track == slot-0 delayed", worst < 1.0, f"max {worst:.4f} m over {len(ks)} slots")

            # access-profile roll on a real target set
            cov = CoverageConfig(earth=EARTH, detection_time_s=15.0, decision_time_s=0.0,
                                 target_missile_burnout_time_s=170.0, burnout_velocity_km_s=6.0, max_accel_g=10.0,
                                 intercept_alt_km=200.0, min_elev_deg=10.0)
            shell = Shell.from_points(EARTH, lat_deg=lat, lon_deg=lon, shell_alt_km=cov.intercept_alt_km)
            sim = SimConfig(horizon_s=(L - 1) * dt, dt_s=dt, use_j2=True, analysis_matrix_dtype="uint16")
            sl = [0, 7, L // 2]
            acc = []
            for s in sl:
                e1 = OrbitalElements(*(getattr(layer.elems, f)[[s]] for f in
                                       ("a_km", "e", "i_rad", "raan_rad", "argp_rad", "M0_rad")))
                ao = run_simulation(e1, sim, cov, EARTH, shell, [{"type": "x"}], build_meta=False)
                check(f"{tag}: simulation returns exactly L={L} time steps", ao.counts.shape[0] == L)
                acc.append((ao.counts > 0))
            v0 = acc[0]
            ok = all(np.array_equal(acc[j], np.roll(v0, sl[j], axis=0)) for j in (1, 2))
            check(f"{tag}: access profile v_k == roll(v0, k)", ok, f"v0 has {int(v0.sum())} access samples")


def test_distinct_slots(plans) -> None:
    print("\n5a. Reducible rungs: distinct-slot restriction")
    from sbi_coverage.Lee import n_distinct_slots, build_param_V
    plan = plans["Russia"]
    red = [f for f in plan["families"] if f["reducible"]]
    check("Russia plan contains a reducible rung", len(red) > 0)
    for fam in plan["families"]:
        g = fam["repeat_factor"]
        check(f"N_P={fam['N_P']}: n_distinct_slots == L//g", n_distinct_slots(fam, plan["L"]) == plan["L"] // g)
    for fam in red:
        g = fam["repeat_factor"]
        # exact-duplicate case: choose L divisible by g
        L = g * int(round(plan["T_r_s"] / (120.0 * g))); dt = plan["T_r_s"] / L
        layer = rgt_common_ground_track_layer(inc_deg=fam["inc_deg"], N_P=fam["N_P"], N_D=fam["N_D"], L=L, earth=EARTH,
                                              raan0_deg=9.0, a_km=fam["a_km"], dt_s=dt)
        m = L // g
        idx = np.array([0, 5, m, m + 5, 2 * m])
        pos = _propagate_positions(_elems_from_layer(layer, idx), L, dt, use_j2=True, mdot=True)
        e1 = np.linalg.norm(pos[:, 2] - pos[:, 0], axis=1).max() * 1000.0
        e2 = np.linalg.norm(pos[:, 3] - pos[:, 1], axis=1).max() * 1000.0
        e3 = np.linalg.norm(pos[:, 4] - pos[:, 0], axis=1).max() * 1000.0
        check(f"N_P={fam['N_P']} ({fam['reduced_ratio']}): slot k+L/g is the SAME orbit position as slot k (L={L})",
              max(e1, e2, e3) < 1.0, f"max {max(e1, e2, e3):.4f} m")
        v0 = (np.random.default_rng(1).random((L, 3)) < 0.02).astype(np.float64)
        Vfull = build_param_V([v0]); Vr = build_param_V([v0], [m])
        check(f"N_P={fam['N_P']}: restricted V == first L/g columns of full V",
              (Vr != Vfull[:, :m]).nnz == 0 and Vr.shape == (L * 3, m))


def print_family_tables(plans) -> None:
    print("\n5. Families by principal target set (N_D=3, altitude 200 km to intercept alt + R_max, anchor = lat_max)")
    for name, plan in plans.items():
        g = plan["generation"]
        print(f"\n  {name}: lat {g['target_abs_lat_min_deg']:.1f}-{g['target_abs_lat_max_deg']:.1f}  "
              f"i_top={plan['anchor']['inc_deg']:.2f}  T_r={plan['T_r_s']:.3f} s  L={plan['L']}  dt={plan['dt_s']:.4f}")
        for f in plan["families"]:
            print(f"    {'anchor' if f['is_anchor'] else '      '} N_P={f['N_P']:3d}:{f['N_D']}  i={f['inc_deg']:7.3f}  "
                  f"alt={f['alt_km']:7.2f} km  RAAN interval={f['raan_interval_deg']:.3f} deg")


def main() -> int:
    test_lee_example5()
    plans = test_families()
    test_track_closure(plans)
    test_circulant(plans)
    test_distinct_slots(plans)
    print_family_tables(plans)
    print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED:\n  " + "\n  ".join(FAILS)))
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main())
