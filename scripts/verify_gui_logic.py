"""Self-checks for the GUI's logic (no AWS, no browser): target-file parsing, grid/number-list parsing and
validation, run planning, job submission, the S3 index and exports. Uses fake S3/Batch clients."""
import io
import json
import sys
import tempfile
from pathlib import Path

from sbi_coverage.core.target_points import TargetFileError, parse_target_points_text
from sbi_coverage.runs import export as exp
from sbi_coverage.runs import launch as lc
from sbi_coverage.runs import s3_index
from sbi_coverage.runs.names import encode_run_name
from sbi_coverage.runs.config import settings_from_config

FAILS: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


def raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


print("target file parsing")
pts, w = parse_target_points_text("Name,Lat,Lon\nSohae,39.66,124.70\nYongbyon ,39.79 ,125.75\n")
check("basic csv", len(pts) == 2 and pts[1] == ["Yongbyon", 39.79, 125.75] and not w)
check("semicolon + latitude/longitude aliases", len(parse_target_points_text("Location;Latitude;Longitude\nA;10;20\nB;11;21")[0]) == 2)
check("tab separated", len(parse_target_points_text("name\tlat\tlon\nA\t10\t20\nB\t11\t21")[0]) == 2)
check("BOM tolerated", len(parse_target_points_text("﻿Name,Lat,Lon\nA,1,2")[0]) == 1)
p2, w2 = parse_target_points_text("A,10,20\nB,11,21")
check("header-less 3 columns accepted with a warning", len(p2) == 2 and any("No header" in x for x in w2))
p3, w3 = parse_target_points_text("Name,Lat,Lon\nA,10,200")
check("longitude > 180 wrapped with a warning", p3[0][2] == -160.0 and any("wrapped" in x for x in w3))
check("duplicate coordinates warned", any("same coordinates" in x for x in parse_target_points_text("Name,Lat,Lon\nA,1,2\nB,1,2")[1]))
check("no name column -> numbered", parse_target_points_text("Lat,Lon\n1,2")[0][0][0] == "Target 1")
for label, txt in (("empty", ""), ("only header", "Name,Lat,Lon\n"), ("missing lon column", "Name,Lat\nA,1"),
                   ("latitude out of range", "Name,Lat,Lon\nA,95,10"), ("text coordinates", "Name,Lat,Lon\nA,x,10"),
                   ("longitude out of range", "Name,Lat,Lon\nA,10,500")):
    check(f"rejects: {label}", raises(lambda t=txt: parse_target_points_text(t), TargetFileError))
try:
    parse_target_points_text("Name,Lat,Lon\n" + "\n".join(f"A{i},{95 + i},0" for i in range(20)))
except TargetFileError as e:
    check("many row problems are summarized (max 8 + 1)", len(e.problems) == 9)

print("\nnumber lists")
P = lc.parse_number_list
check("list", P("5, 10, 20", name="x", kind=int) == [5, 10, 20])
check("range", P("10-50:10", name="x", kind=int) == [10, 20, 30, 40, 50])
check("float range + list + dedupe", P("0.5-1.5:0.5, 1", name="x") == [0.5, 1.0, 1.5])
check("semicolons/newlines", P("1;2\n3", name="x", kind=int) == [1, 2, 3])
for label, kw in (("empty", dict(text="")), ("garbage", dict(text="abc")), ("negative", dict(text="-5")),
                  ("fraction for int", dict(text="2.5", kind=int)), ("above max", dict(text="500", maximum=100)),
                  ("below min", dict(text="0", minimum=1)), ("4 decimals", dict(text="1.2345")),
                  ("backwards range", dict(text="10-5:1")), ("zero step", dict(text="1-5:0")),
                  ("huge range", dict(text="1-100000:1", kind=int))):
    kw.setdefault("kind", float)
    t = kw.pop("text")
    check(f"rejects: {label}", raises(lambda: P(t, name="field", **kw), lc.InputProblems))

print("\nscenario and summary")
check("scenario validation", lc.validate_scenario("nk") == "NK" and raises(lambda: lc.validate_scenario("n"), lc.InputProblems)
      and raises(lambda: lc.validate_scenario("TOOLONG"), lc.InputProblems))
check("default scenario from file path / country", lc.default_scenario(filename="CN/Prespecified_100.csv") == "CN"
      and lc.default_scenario(countries=["North Korea"]) == "NK")

print("\nplanning")
TARGET = {"mode": "prespecified", "points": pts}
FIXED = {k: (lc.CONFIG_DEFAULTS[k] if k != "time_limit_s" else 600.0) for k in lc.ADVANCED_FIELDS}
GRID = {k: lc.parse_number_list(v["default"], name=k, kind=v["kind"]) for k, v in lc.GRID_FIELDS.items()}
plan = lc.build_grid_plan(target=TARGET, grid=GRID, fixed=FIXED, scenario="NK", label="t", principal=True, test=True)
check("default grid -> 3 combinations (salvo 5,10,20)", len(plan.configs) == 3 and not plan.problems)
check("identities are unique and name-encodable", len({r["identity"] for r in plan.table}) == 3)
check("configs carry flags", all(c["is_test_run"] and c["is_principal_set"] and c["scenario"] == "NK" for c in plan.configs))
bad_grid = {**GRID, "detection_time_s": [200.0], "target_missile_burnout_time_s": [170.0]}
p_bad = lc.build_grid_plan(target=TARGET, grid=bad_grid, fixed=FIXED, scenario="NK", label=None, principal=False, test=False)
check("detection >= burnout time -> excluded with guidance", not p_bad.configs and p_bad.problems)
big = {**GRID, "salvo_size": list(range(1, 2000)), "n_interceptors_per_sat": list(range(1, 5))}
check("combination cap enforced", raises(lambda: lc.build_grid_plan(target=TARGET, grid=big, fixed=FIXED, scenario="NK", label=None, principal=False, test=False), lc.InputProblems))
ctarget = {"mode": "country", "countries": ["Brazil"], "n_points": 8000}
pc = lc.build_grid_plan(target=ctarget, grid=GRID, fixed=FIXED, scenario="BR", label=None, principal=False, test=True)
check("country mode configs", pc.configs[0]["target_mode"] == "country" and pc.configs[0]["target_shell_n_points"] == 8000)

print("\nsubmission (fake clients)")
class FakeS3:
    def __init__(self): self.puts = {}
    def put_object(self, Bucket, Key, Body, **kw): self.puts[Key] = Body
class FakeBatch:
    def __init__(self, fail_every=0): self.n = 0; self.fail_every = fail_every; self.jobs = []
    def submit_job(self, **kw):
        self.n += 1
        if self.fail_every and self.n % self.fail_every == 0:
            raise RuntimeError("throttled")
        self.jobs.append(kw)
        return {"jobId": f"id-{self.n}"}
s3f, bf = FakeS3(), FakeBatch(fail_every=3)
prog = []
res = lc.submit_runs(plan.configs, s3=s3f, batch=bf, progress=lambda d, n: prog.append(d), max_workers=2)
check("every config uploaded, one result each", len(res) == 3 and len(s3f.puts) == 3)
check("one failure captured, others submitted", sorted(r["status"] for r in res) == ["FAILED_TO_SUBMIT", "SUBMITTED", "SUBMITTED"])
check("progress reported", prog and prog[-1] == 3)
check("job names unique and Batch-legal", len({r["job_name"] for r in res}) == 3 and all(__import__("re").fullmatch(r"[A-Za-z0-9_-]{1,128}", r["job_name"]) for r in res))
sent = json.loads(next(iter(s3f.puts.values())))
check("config JSON uses current vocabulary", "burnout_velocity_km_s" in sent and "max_accel_g" in sent and "v_bo_km_s" not in sent)

print("\nS3 index (fake)")
S = settings_from_config(plan.configs[0])
nm_ok = encode_run_name(S, "261010", "a")
nm_fail = encode_run_name(settings_from_config(plan.configs[1]), "261010", "a")
nm_pending = encode_run_name(settings_from_config(plan.configs[2]), "261010", "a")
sidecar = {"schema_version": 2, "run_name": nm_ok, "result": {"n_satellites": 12, "status": "Optimal", "mip_gap_achieved": 0.0},
           "cov": {}, "sim": {}, "targets": [], "countries": [], "provenance": {"git_commit": "abc"},
           "run_config": {"anchor_alt_km": 286.0, "families": [{"N_P": 47, "N_D": 3, "alt_km": 286.0, "satellites_selected": 12}], "r_required": 1}}
OBJECTS = {f"runs/{nm_ok}_result.json": json.dumps(sidecar).encode(), f"runs/{nm_ok}_result.csv": b"a_km\n1\n",
           f"runs/{nm_fail}_failed.json": json.dumps({"error": "boom"}).encode(), f"runs/{nm_pending}_config.json": b"{}",
           "runs/README.txt": b"x"}
class FakeS3b:
    def get_paginator(self, _):
        class P:
            def paginate(s, Bucket, Prefix):
                yield {"Contents": [{"Key": k, "Size": len(v), "ETag": '"e"', "LastModified": None} for k, v in OBJECTS.items()]}
        return P()
    def get_object(self, Bucket, Key):
        return {"Body": io.BytesIO(OBJECTS[Key])}
df = s3_index.build_index(FakeS3b())
st = dict(zip(df["name"], df["result_status"]))
check("3 runs indexed, junk file ignored", len(df) == 3)
check("finished run has results", st[nm_ok] == "Optimal" and int(df.set_index("name").loc[nm_ok, "n_satellites"]) == 12)
check("failed run flagged with its error", st[nm_fail] == "FAILED" and "boom" in df.set_index("name").loc[nm_fail, "status_note"])
check("name-only run flagged as incomplete", "INCOMPLETE" in st[nm_pending])
ex_ids = lc.existing_identities(s3_index.list_run_files(FakeS3b()))
check("existing identities include only finished runs", list(ex_ids) == [lc.settings_key(S)] if hasattr(lc, "settings_key") else len(ex_ids) == 1)

print("\nexports")
csvb = exp.df_csv_bytes(df, ["name", "n_satellites", "files"])
check("csv export with dict column", b"name,n_satellites,files" in csvb and b"result.json" in csvb)
jb = exp.json_bytes({"x": float("nan"), "d": __import__("datetime").datetime(2026, 1, 1), "n": __import__("numpy").int64(3)})
check("json export cleans nan/datetime/numpy", json.loads(jb) == {"x": None, "d": "2026-01-01T00:00:00", "n": 3})
with tempfile.TemporaryDirectory() as td:
    folder, err = exp.validate_export_dir(str(Path(td) / "a" / "b"))
    check("export folder created", folder is not None and err is None)
    p1 = exp.save_export(folder, "my file?.csv", b"1")
    p2 = exp.save_export(folder, "my file?.csv", b"2")
    check("filename sanitized, no overwrite", p1.name == "my_file_.csv" and p2.name == "my_file__1.csv" and p1.read_bytes() == b"1")
check("empty export folder rejected", exp.validate_export_dir("  ")[1] is not None)

print("\n" + ("ALL CHECKS PASSED" if not FAILS else f"{len(FAILS)} CHECK(S) FAILED:\n  " + "\n  ".join(FAILS)))
sys.exit(0 if not FAILS else 1)
