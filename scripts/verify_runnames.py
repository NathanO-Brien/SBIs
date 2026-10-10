"""Self-checks for sbi_coverage.runnames (round trip, collisions, length, key stability)."""
import datetime
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sbi_coverage.runnames import (decode_run_name, encode_run_name, find_runs, next_seq,
                                   parse_filename, settings_key)

rng = random.Random(7)
base = dict(scenario="CN", principal=True, salvo_size=46, detection_time_s=15.0, v_bo_km_s=6.0, a_g=10.0,
            target_missile_burnout_time_s=170.0, decision_time_s=0.0, n_interceptors_per_sat=1,
            intercept_alt_km=200.0, nd_repeat=3, altitude_min_km=200.0, anchor_lat_margin_deg=0.0,
            raan_step_deg=0.25, margin_multiplier=5.0, trim_margin_multiplier=2.0, max_seeds=200,
            orbit_model="lee_j2", target_points=[["A", 40.1, 126.5], ["B", 42.3, 125.0]])
d = datetime.date(2026, 10, 9)

n = encode_run_name(base, d)
print(n)
assert n.startswith("CNP-s046-d15-v6-g10-b170-k0-i1-261009a-"), n

# round trip over random settings
for _ in range(2000):
    s = dict(base, salvo_size=rng.randint(1, 200), detection_time_s=rng.choice([0, 5, 7.5, 15, 30]),
             v_bo_km_s=rng.choice([4, 4.5, 6, 7.25]), a_g=rng.choice([5, 10, 15]),
             target_missile_burnout_time_s=rng.choice([100, 170, 250]), decision_time_s=rng.choice([0, 2.5, 10]),
             n_interceptors_per_sat=rng.randint(1, 4), scenario=rng.choice(["NK", "CN", "RU", "IR", "BR", "PS"]),
             principal=rng.random() < 0.5)
    name = encode_run_name(s, d, rng.choice("abc"))
    i = decode_run_name(name)
    assert i["salvo_size"] == s["salvo_size"] and i["detection_time_s"] == s["detection_time_s"]
    assert i["v_bo_km_s"] == s["v_bo_km_s"] and i["a_g"] == s["a_g"] and i["decision_time_s"] == s["decision_time_s"]
    assert i["target_missile_burnout_time_s"] == s["target_missile_burnout_time_s"]
    assert i["n_interceptors_per_sat"] == s["n_interceptors_per_sat"]
    assert i["scenario"] == s["scenario"] and i["principal"] == s["principal"] and i["date_iso"] == "2026-10-09"
    assert i["key"] == settings_key(s), (i["key"], settings_key(s))
    assert len(name) <= 52 and re.fullmatch(r"[A-Za-z0-9.\-]+", name), name
    fn = parse_filename(name + "_result.json")
    assert fn == (name, "result.json"), fn
print("round trip OK (2000 random settings)")

# hidden settings change the hash; date/seq do not change the key
for k, v in dict(raan_step_deg=0.5, nd_repeat=2, margin_multiplier=4.0, intercept_alt_km=250.0,
                 target_points=[["A", 40.1, 126.5]], orbit_model="keplerian_rgt").items():
    assert settings_key(dict(base, **{k: v})) != settings_key(base), k
assert settings_key(base) == decode_run_name(encode_run_name(base, datetime.date(2027, 1, 2), "c"))["key"]
# point order/name must not matter
assert settings_key(dict(base, target_points=[["z", 42.3, 125.0], ["y", 40.1, 126.5]])) == settings_key(base)
t = encode_run_name(dict(base, test=True), d)
assert t.startswith("TEST-CNP-s046") and decode_run_name(t)["test"] and not decode_run_name(n)["test"]
assert decode_run_name(t + "_result.json")["scenario"] == "CN" and parse_filename(t + "_result.json") == (t, "result.json")
assert settings_key(dict(base, test=True)) != settings_key(base)
print("hash/key behaviour OK")

# sequence letters and querying
names = [encode_run_name(base, d, "a"), encode_run_name(dict(base, salvo_size=10), d, "a"),
         encode_run_name(dict(base, detection_time_s=30.0), d, "a")]
assert next_seq(names, base, d) == "b" and next_seq(names, dict(base, a_g=5.0), d) == "a"
assert len(find_runs(names, scenario="cn")) == 3 and len(find_runs(names, detection_time_s=[15])) == 2
assert len(find_runs(names, salvo_size=10, principal=True)) == 1
assert parse_filename("results.csv") is None
try:
    decode_run_name("nonsense")
    raise SystemExit("decode should reject nonsense")
except ValueError:
    pass
print("query/sequence OK\nALL CHECKS PASSED")
