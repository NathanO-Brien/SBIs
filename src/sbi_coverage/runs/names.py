"""Short, decodable run names for SBI optimization results.

Stdlib only, so this single file can be handed to someone else as-is.

A run name looks like::

    CNp-s046-d15-v6-g10-b170-k0-i1-261009a-3f9c
    |   |    |   |  |   |    |  |  |      |
    |   |    |   |  |   |    |  |  |      +-- 4-hex hash of every result-affecting setting NOT shown
    |   |    |   |  |   |    |  |  +--------- date (yymmdd, UTC) + same-day sequence letter
    |   |    |   |  |   |    |  +------------ interceptors per satellite
    |   |    |   |  |   |    +--------------- decision time [s]
    |   |    |   |  |   +-------------------- target-missile burnout time [s]
    |   |    |   |  +------------------------ max interceptor acceleration [g]
    |   |    |   +--------------------------- interceptor burnout velocity [km/s]
    |   |    +------------------------------- detection time [s]
    |   +------------------------------------ salvo size
    +---------------------------------------- scenario code (2-5 capitals); lowercase p = principal set

Test runs get a ``TEST-`` prefix (``TEST-NKp-s005-...``) so they are never mistaken for real
results; the run browser hides them by default.

Every output product of a run shares the name as its file stem (``<name>_result.json``,
``<name>_result.csv``, ``<name>_config.json``, ...), and all products live in one flat folder /
S3 prefix.

Identity vs. provenance
-----------------------
The readable tokens are the settings people filter and plot by. Everything else that changes the
*result* (target points, intercept altitude, minimum elevation, time step, N_D, altitude floor,
anchor margin, RAAN step, seed-filter margins and cap, reducible-rung switch, method version, and the
principal/test flags) is folded into the 4-hex hash, so two runs that differ in any of them never
share a name. Solver *limits* (time limit, MIP gap, Gurobi parameters) are deliberately NOT part of
the identity: a longer re-solve of the same problem is the same run configuration; those values are
recorded in the result JSON instead. ``settings_key`` is the name without date/sequence, so a
"does this run already exist?" check ignores *when* it ran.

Usage::

    from sbi_coverage.runs.names import encode_run_name, decode_run_name, settings_key
    name = encode_run_name(settings)           # settings: dict with the keys in READABLE/HASHED below
    info = decode_run_name(name)               # dict of readable fields + date + hash
    python -m sbi_coverage.runs.names encode settings.json
    python -m sbi_coverage.runs.names decode CNp-s046-d15-v6-g10-b170-k0-i1-261009a-3f9c
"""
from __future__ import annotations

import datetime
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

#: Bump when a code change alters results for identical inputs (it changes every hash).
METHOD_VERSION = "lee_j2.v1"

# Readable tokens, in name order: (token letter, settings key, python type, zero-pad width)
READABLE: tuple[tuple[str, str, type, int], ...] = (
    ("s", "salvo_size",                    int,   3),
    ("d", "detection_time_s",              float, 0),
    ("v", "burnout_velocity_km_s",         float, 0),
    ("g", "max_accel_g",                   float, 0),
    ("b", "target_missile_burnout_time_s", float, 0),
    ("k", "decision_time_s",               float, 0),
    ("i", "n_interceptors_per_sat",        int,   0),
)

# Result-affecting settings folded into the hash (missing keys are simply absent from the hash input).
HASHED_SCALARS: tuple[str, ...] = (
    "intercept_alt_km", "min_elev_deg", "dt_target_s", "nd_repeat", "altitude_min_km",
    "anchor_lat_margin_deg", "raan_step_deg", "margin_multiplier", "trim_margin_multiplier",
    "max_seeds", "allow_reducible", "useful_reach_deg", "target_country", "target_shell_n_points",
)

# Scenario codes. Country names map to a code; an explicit ``scenario`` setting wins.
SCENARIO_CODES: dict[str, str] = {
    "north korea": "NK", "china": "CN", "russia": "RU", "iran": "IR", "brazil": "BR",
    "united states": "US", "india": "IN", "pakistan": "PK", "israel": "IL",
}
KNOWN_CODES = frozenset(SCENARIO_CODES.values())

_NUM = r"\d+(?:\.\d{1,3})?"
_NAME_RE = re.compile(
    r"^(?:(?P<test>TEST)-)?(?P<tgt>[A-Z][A-Z0-9]{1,4})(?P<principal>p?)"
    rf"-s(?P<s>\d+)-d(?P<d>{_NUM})-v(?P<v>{_NUM})-g(?P<g>{_NUM})-b(?P<b>{_NUM})"
    rf"-k(?P<k>{_NUM})-i(?P<i>\d+)-(?P<date>\d{{6}})(?P<seq>[a-z])-(?P<hash>[0-9a-f]{{4}})$"
)
# A name followed by an optional "_<product>" suffix (e.g. "_result.json").
_FILE_RE = re.compile(r"^(?P<name>.+?-[0-9a-f]{4})(?:_(?P<product>.+))?$")


def _num(x: float | int, what: str = "value") -> str:
    """Compact fixed-point number text (<= 3 decimals): 6.0 -> '6', 6.5 -> '6.5', 0.25 -> '0.25'.

    Raises ValueError for negatives, non-finite values, or values that need more than 3 decimals
    (they would not survive a decode round trip, so the name would silently collide).
    """
    f = float(x)
    if not math.isfinite(f) or f < 0.0:
        raise ValueError(f"run name: {what}={x!r} must be finite and >= 0")
    txt = f"{f:.3f}".rstrip("0").rstrip(".")
    if abs(float(txt) - f) > 1e-9:
        raise ValueError(f"run name: {what}={x!r} has more than 3 decimals; it cannot be encoded")
    return txt


def scenario_code(settings: Mapping[str, Any]) -> str:
    """Scenario code (2-5 capitals) from an explicit ``scenario`` setting, else the country, else 'PS'."""
    code = str(settings.get("scenario") or "").strip().upper()
    if not code:
        c = settings.get("target_country")
        if isinstance(c, (list, tuple)):
            c = "+".join(sorted(str(x) for x in c))
        code = SCENARIO_CODES.get(str(c or "").strip().lower(), "")
        if not code and c:
            code = re.sub(r"[^A-Z0-9]", "", str(c).upper())[:3]
    if not code:
        code = "PS"
    code = re.sub(r"[^A-Z0-9]", "", code)
    if not re.fullmatch(r"[A-Z][A-Z0-9]{1,4}", code):
        raise ValueError(f"run name: scenario code {code!r} must be 2-5 capitals/digits starting with a letter")
    return code


def scenario_from_path(path: str | Path) -> str | None:
    """Guess a scenario code from any folder/file name component (e.g. '.../NK/...')."""
    for part in reversed(Path(str(path)).parts):
        p = part.strip().upper()
        if p in KNOWN_CODES:
            return p
        low = part.strip().lower().replace("_", " ").replace("-", " ")
        if low in SCENARIO_CODES:
            return SCENARIO_CODES[low]
    return None


def _canon_points(points: Iterable[Any]) -> list[list[float]]:
    out = []
    for p in points:
        lat, lon = (p[1], p[2]) if not isinstance(p, Mapping) else (p["lat_deg"], p["lon_deg"])
        out.append([round(float(lat), 4), round(float(lon), 4)])
    return sorted(out)


def _canon_scalar(v: Any) -> Any:
    """Type-stable value for hashing: 3 and 3.0 hash alike; strings are lower-cased; lists sorted."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return round(float(v), 6)
    if isinstance(v, (list, tuple)):
        return sorted(str(x).strip().lower() for x in v)
    return str(v).strip().lower()


def _hash4(settings: Mapping[str, Any]) -> str:
    h: dict[str, Any] = {"method": METHOD_VERSION,
                         "principal": bool(settings.get("principal")), "test": bool(settings.get("test"))}
    for k in HASHED_SCALARS:
        v = settings.get(k)
        if v is not None:
            h[k] = _canon_scalar(v)
    pts = settings.get("target_points")
    if pts:
        h["target_points"] = _canon_points(pts)
    blob = json.dumps(h, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:4]


def settings_hash(settings: Mapping[str, Any]) -> str:
    """The 4-hex hash of the result-affecting settings that are not shown in the readable tokens."""
    return _hash4(settings)


def _readable(settings: Mapping[str, Any]) -> str:
    code = scenario_code(settings)
    prefix = "TEST-" if settings.get("test") else ""
    parts = [prefix + code + ("p" if settings.get("principal") else "")]
    for letter, key, typ, width in READABLE:
        if key not in settings:
            raise KeyError(f"run-name setting '{key}' is missing")
        v = settings[key]
        if typ is int:
            iv = int(round(float(v)))
            if iv < 0 or abs(iv - float(v)) > 1e-9:
                raise ValueError(f"run name: {key}={v!r} must be a non-negative integer")
            parts.append(f"{letter}{iv:0{width}d}")
        else:
            parts.append(f"{letter}{_num(v, key)}")
    return "-".join(parts)


def settings_key(settings: Mapping[str, Any]) -> str:
    """Date-free identity of a run: readable tokens + hash (used for 'does it exist?')."""
    return f"{_readable(settings)}-{_hash4(settings)}"


def _yymmdd(date: datetime.date | str | None) -> str:
    if date is None:
        date = datetime.datetime.now(datetime.timezone.utc).date()
    if isinstance(date, datetime.datetime):
        date = date.date()
    if isinstance(date, datetime.date):
        return date.strftime("%y%m%d")
    d = str(date).replace("-", "")[-6:]
    datetime.datetime.strptime(d, "%y%m%d")   # validates
    return d


def encode_run_name(settings: Mapping[str, Any], date: datetime.date | str | None = None, seq: str = "a") -> str:
    """Build the run name. ``date`` defaults to today (UTC); ``seq`` is a-z for same-day reruns."""
    if not re.fullmatch(r"[a-z]", seq):
        raise ValueError("seq must be a single letter a-z")
    return f"{_readable(settings)}-{_yymmdd(date)}{seq}-{_hash4(settings)}"


def next_seq(existing_names: Iterable[str], settings: Mapping[str, Any], date: datetime.date | str | None = None) -> str:
    """First unused sequence letter for these settings on this (UTC) date."""
    key, d = settings_key(settings), _yymmdd(date)
    used = {m["seq"] for n in existing_names if (m := _try_decode(n)) and m["key"] == key and m["date"] == d}
    for c in "abcdefghijklmnopqrstuvwxyz":
        if c not in used:
            return c
    raise RuntimeError("more than 26 runs with identical settings in one day")


def local_run_name(settings: Mapping[str, Any], folder: str | Path) -> str:
    """Run name for a local run: today's date plus the first sequence letter unused in ``folder``."""
    folder = Path(folder)
    names = [x[0] for p in (folder.glob("*") if folder.exists() else []) if (x := parse_filename(p.name))]
    return encode_run_name(settings, None, next_seq(names, settings))


def _try_decode(name: str) -> dict[str, Any] | None:
    try:
        return decode_run_name(name)
    except ValueError:
        return None


def decode_run_name(name: str) -> dict[str, Any]:
    """Decode a run name (or a file name carrying one) into its readable fields."""
    m = _FILE_RE.match(Path(name).name)
    stem = m.group("name") if m else name
    mm = _NAME_RE.match(stem)
    if not mm:
        raise ValueError(f"not a valid run name: {name!r}")
    g = mm.groupdict()
    d = datetime.datetime.strptime(g["date"], "%y%m%d").date()
    return {
        "name": stem,
        "scenario": g["tgt"], "principal": bool(g["principal"]), "test": bool(g["test"]),
        "salvo_size": int(g["s"]), "detection_time_s": float(g["d"]), "burnout_velocity_km_s": float(g["v"]),
        "max_accel_g": float(g["g"]), "target_missile_burnout_time_s": float(g["b"]),
        "decision_time_s": float(g["k"]), "n_interceptors_per_sat": int(g["i"]),
        "date": g["date"], "date_iso": d.isoformat(), "seq": g["seq"], "hash": g["hash"],
        "key": re.sub(r"-\d{6}[a-z]-", "-", stem),
    }


def parse_filename(filename: str) -> tuple[str, str] | None:
    """('<name>', '<product>') for e.g. 'CN...-3f9c_result.json' -> (name, 'result.json'), else None."""
    base = Path(filename).name
    m = _FILE_RE.match(base)
    if not m or not _try_decode(m.group("name")):
        return None
    return m.group("name"), (m.group("product") or "")


def matches(info: Mapping[str, Any], **query: Any) -> bool:
    """True if a decoded name satisfies every given field (unspecified/None fields are wildcards).

    Values may be a scalar or a list/set of acceptable values; numbers compare with a 1e-9
    tolerance, scenario codes ignore case.  e.g. matches(info, scenario="CN", detection_time_s=[15, 30])
    """
    for k, want in query.items():
        if want is None:
            continue
        have = info.get(k)
        wants = want if isinstance(want, (list, tuple, set)) else [want]
        ok = False
        for w in wants:
            if isinstance(have, (int, float)) and not isinstance(have, bool):
                ok |= abs(float(have) - float(w)) < 1e-9
            elif k == "scenario":
                ok |= str(have).upper() == str(w).upper()
            else:
                ok |= have == w
        if not ok:
            return False
    return True


def find_runs(names: Iterable[str], **query: Any) -> list[dict[str, Any]]:
    """Decode and filter names; partial queries allowed. Newest first."""
    out = [i for n in set(names) if (i := _try_decode(n)) and matches(i, **query)]
    return sorted(out, key=lambda i: (i["date"], i["seq"]), reverse=True)


def _cli(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in {"encode", "decode"}:
        print(__doc__)
        return 2
    if argv[0] == "decode":
        print(json.dumps(decode_run_name(argv[1]), indent=2))
    else:
        print(encode_run_name(json.loads(Path(argv[1]).read_text())))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
