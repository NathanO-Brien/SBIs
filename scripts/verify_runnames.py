"""Self-checks for sbi_coverage.runs (names, config, JSON vocabulary): round trips, hash policy, defaults parity."""
import datetime
import random
import re
import sys

from sbi_coverage.runs.config import CONFIG_DEFAULTS, normalize_config, settings_from_config
from sbi_coverage.runs.index import normalize_sidecar
from sbi_coverage.runs.names import (HASHED_SCALARS, READABLE, decode_run_name, encode_run_name, find_runs, next_seq,
                                     parse_filename, scenario_code, settings_key)

FAILS: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


rng = random.Random(7)
PTS = [["A", 40.1, 126.5], ["B", 42.3, 125.0]]
base_cfg = dict(target_mode="prespecified", target_points=PTS, scenario="CN", is_principal_set=True, salvo_size=46,
                n_interceptors_per_sat=1, burnout_velocity_km_s=6.0, detection_time_s=15.0, time_limit_s=60.0)
base = settings_from_config(base_cfg)
d = datetime.date(2026, 10, 9)
name = encode_run_name(base, d)
print(name)

print("\nname format and round trip")
check("expected readable prefix", name.startswith("CNp-s046-d15-v6-g10-b170-k0-i1-261009a-"))
ok = True
for _ in range(2000):
    s = dict(base, salvo_size=rng.randint(1, 200), detection_time_s=rng.choice([0, 5, 7.5, 15, 30, 0.25]),
             burnout_velocity_km_s=rng.choice([4, 4.5, 6, 7.25]), max_accel_g=rng.choice([5, 10, 15, 12.5]),
             target_missile_burnout_time_s=rng.choice([100, 170, 250]), decision_time_s=rng.choice([0, 2.5, 10]),
             n_interceptors_per_sat=rng.randint(1, 4), scenario=rng.choice(["NK", "CN", "RU", "IR", "BR", "PS", "NEP", "AB1"]),
             principal=rng.random() < 0.5, test=rng.random() < 0.3)
    nm = encode_run_name(s, d, rng.choice("abc"))
    i = decode_run_name(nm)
    ok &= (i["salvo_size"] == s["salvo_size"] and i["detection_time_s"] == s["detection_time_s"]
           and i["burnout_velocity_km_s"] == s["burnout_velocity_km_s"] and i["max_accel_g"] == s["max_accel_g"]
           and i["decision_time_s"] == s["decision_time_s"]
           and i["target_missile_burnout_time_s"] == s["target_missile_burnout_time_s"]
           and i["n_interceptors_per_sat"] == s["n_interceptors_per_sat"] and i["scenario"] == s["scenario"]
           and i["principal"] == s["principal"] and i["test"] == s["test"] and i["key"] == settings_key(s)
           and len(nm) <= 58 and re.fullmatch(r"[A-Za-z0-9.\-]+", nm) is not None
           and parse_filename(nm + "_result.json") == (nm, "result.json"))
check("2000 random settings round-trip (incl. scenario codes ending in P, test flag)", ok)
nep = decode_run_name(encode_run_name(dict(base, scenario="NEP", principal=False), d))
check("'NEP' scenario is not mistaken for principal", (not nep["principal"]) and nep["scenario"] == "NEP")
for bad in (dict(detection_time_s=1e-5), dict(detection_time_s=-1.0), dict(burnout_velocity_km_s=6.0000001),
            dict(scenario="X"), dict(salvo_size=-3), dict(salvo_size=2.5)):
    try:
        encode_run_name(dict(base, **bad), d)
        check(f"rejects unencodable setting {bad}", False)
    except ValueError:
        check(f"rejects unencodable setting {bad}", True)
try:
    decode_run_name("nonsense")
    check("decode rejects nonsense", False)
except ValueError:
    check("decode rejects nonsense", True)

print("\nidentity vs provenance (hash policy)")
for k, v in dict(raan_step_deg=0.5, nd_repeat=2, margin_multiplier=4.0, intercept_alt_km=250.0, min_elev_deg=5.0,
                 dt_target_s=60.0, allow_reducible=False, useful_reach_deg=3.0, max_seeds=100,
                 altitude_min_km=250.0, anchor_lat_margin_deg=1.0, trim_margin_multiplier=1.5).items():
    check(f"hidden setting {k} changes the hash", settings_key(dict(base, **{k: v})) != settings_key(base))
check("target points change the hash", settings_key(dict(base, target_points=PTS[:1])) != settings_key(base))
check("point order/name do not matter",
      settings_key(dict(base, target_points=[["z", 42.3, 125.0], ["y", 40.1, 126.5]])) == settings_key(base))
check("principal / test flags change the hash",
      len({settings_key(base), settings_key(dict(base, principal=False)), settings_key(dict(base, test=True))}) == 3)
check("3 and 3.0 hash alike", settings_key(dict(base, nd_repeat=3)) == settings_key(dict(base, nd_repeat=3.0)))
check("date/sequence do not change the key",
      settings_key(base) == decode_run_name(encode_run_name(base, datetime.date(2027, 1, 2), "c"))["key"])
for k, v in dict(time_limit_s=3600.0, mip_gap=0.05, stop_at_gap=0.5, mip_focus=1, heuristics_frac=0.2,
                 solver_params={"Method": 2}, label="x").items():
    check(f"solver/bookkeeping setting {k} does NOT change run identity",
          settings_key(settings_from_config({**base_cfg, k: v})) == settings_key(base))
check("every readable token key is produced by settings_from_config", all(k in base for _, k, _, _ in READABLE))
check("every hashed setting is covered by settings_from_config",
      set(HASHED_SCALARS) - {"target_country", "target_shell_n_points"} <= set(base))
check("country config hashes the country, not points",
      "target_country" in settings_from_config(dict(base_cfg, target_mode="country", target_country="China", target_points=None)))

print("\nsequence letters and queries")
names = [encode_run_name(base, d, "a"), encode_run_name(dict(base, salvo_size=10), d, "a"),
         encode_run_name(dict(base, detection_time_s=30.0), d, "a")]
check("next_seq increments for identical settings, restarts otherwise",
      next_seq(names, base, d) == "b" and next_seq(names, dict(base, max_accel_g=5.0), d) == "a")
check("find_runs partial queries", len(find_runs(names, scenario="cn")) == 3 and len(find_runs(names, detection_time_s=[15])) == 2
      and len(find_runs(names, salvo_size=10, principal=True)) == 1)
check("scenario_code from country", scenario_code({"target_country": "North Korea"}) == "NK")

print("\nconfig defaults and legacy vocabulary")
n = normalize_config(base_cfg)
check("defaults filled", n["raan_step_deg"] == 0.25 and n["nd_repeat"] == 3 and n["max_accel_g"] == 10.0
      and n["margin_multiplier"] == 5.0)
legacy = {k: v for k, v in base_cfg.items() if k != "burnout_velocity_km_s"}
legacy.update(v_bo_km_s=6.0, a_g=12.0, dt_s=100.0, max_seeds_safety_cap=77, run_name="old label",
              target_altitude_km=400, n_raan_offsets=9)
nl = normalize_config(legacy)
check("legacy keys mapped (v_bo_km_s, a_g, dt_s, max_seeds_safety_cap, run_name->label)",
      nl["burnout_velocity_km_s"] == 6.0 and nl["max_accel_g"] == 12.0 and nl["dt_target_s"] == 100.0
      and nl["max_seeds"] == 77 and nl["label"] == "old label")
check("retired altitude/inclination keys are dropped", "target_altitude_km" not in nl and "n_raan_offsets" not in nl)
for missing in ("salvo_size", "time_limit_s"):
    try:
        normalize_config({k: v for k, v in base_cfg.items() if k != missing})
        check(f"missing {missing} rejected", False)
    except ValueError:
        check(f"missing {missing} rejected", True)
check("normalize_config returns every CONFIG_DEFAULTS key", set(normalize_config(base_cfg)) >= set(CONFIG_DEFAULTS))
old = {"version": 1, "n_satellites": 12, "status_str": "Optimal", "mip_gap": 0.0,
       "cov": {"v_bo_km_s": 6.0, "a_g": 10.0}, "sim": {"use_j2": True}, "targets": [], "countries": [], "bc_kg_m2": 80.0,
       "run_config": {"actual_alt_km": 286.0, "run_name": "lab", "seeds_selected": [], "a_g": 10.0, "rgt_ratio": "47:3"}}
ns = normalize_sidecar(old)
check("schema-1 sidecar normalized", ns["result"]["n_satellites"] == 12 and ns["cov"]["burnout_velocity_km_s"] == 6.0
      and ns["run_config"]["anchor_alt_km"] == 286.0 and ns["run_config"]["label"] == "lab"
      and ns["run_config"]["max_accel_g"] == 10.0)

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED:\n  " + "\n  ".join(FAILS)))
sys.exit(0 if not FAILS else 1)
