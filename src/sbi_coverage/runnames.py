"""Short, decodable run names for SBI optimization results.

Stdlib only, so this single file can be handed to someone else as-is.

A run name looks like::

    CNP-s046-d15-v6-g10-b170-k0-i1-261009a-3f9c
    |   |    |   |  |   |    |  |  |      |
    |   |    |   |  |   |    |  |  |      +-- 4-hex hash of every setting NOT shown
    |   |    |   |  |   |    |  |  +--------- date (yymmdd) + same-day sequence letter
    |   |    |   |  |   |    |  +------------ interceptors per satellite
    |   |    |   |  |   |    +--------------- decision time [s]
    |   |    |   |  |   +-------------------- target-missile burnout time [s]
    |   |    |   |  +------------------------ max interceptor acceleration [g]
    |   |    |   +--------------------------- burnout velocity [km/s]
    |   |    +------------------------------- detection time [s]
    |   +------------------------------------ salvo size
    +---------------------------------------- scenario code (+P = principal set)

Test runs get a ``TEST-`` prefix (``TEST-NKP-s005-...``) so they are never mistaken for
real results; the browser hides them by default.

Every output product of a run shares the name as its file stem
(``<name>_result.json``, ``<name>_result.csv``, ``<name>_config.json``, ...), and all
products live in one flat folder / S3 prefix.

The readable tokens are the settings people filter and plot by.  Everything else that
changes a result (exact target points, intercept altitude, N_D, altitude floor, anchor
margin, RAAN step, seed-filter margins, orbit model, ...) is folded into the 4-hex hash,
so two runs that differ only in hidden settings never share a name.  ``settings_key`` is
the name without date/sequence, so "does this run already exist?" ignores *when* it ran.

Usage::

    from sbi_coverage.runnames import encode_run_name, decode_run_name, settings_key
    name = encode_run_name(settings)           # settings: dict, see SETTING_KEYS
    info = decode_run_name(name)               # dict of readable fields + date + hash
    python -m sbi_coverage.runnames encode settings.json
    python -m sbi_coverage.runnames decode CNP-s046-d15-v6-g10-b170-k0-i1-261009a-3f9c
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

# Readable tokens, in name order: (token letter, settings key, python type, zero-pad width)
_TOKENS: tuple[tuple[str, str, type, int], ...] = (
    ("s", "salvo_size",                    int,   3),
    ("d", "detection_time_s",              float, 0),
    ("v", "v_bo_km_s",                     float, 0),
    ("g", "a_g",                           float, 0),
    ("b", "target_missile_burnout_time_s", float, 0),
    ("k", "decision_time_s",               float, 0),
    ("i", "n_interceptors_per_sat",        int,   0),
)

# Settings folded into the hash (anything missing is simply absent from the hash input).
_HASHED_SCALARS: tuple[str, ...] = (
    "intercept_alt_km", "nd_repeat", "altitude_min_km", "anchor_lat_margin_deg",
    "raan_step_deg", "margin_multiplier", "trim_margin_multiplier", "max_seeds",
    "orbit_model", "target_country", "target_shell_n_points", "doctrine",
)

SETTING_KEYS = (
    ("scenario", "principal", "test")
    + tuple(k for _, k, _, _ in _TOKENS)
    + _HASHED_SCALARS
    + ("target_points",)
)

# Scenario codes.  Country names map to a code; an explicit ``scenario`` setting wins.
SCENARIO_CODES: dict[str, str] = {
    "north korea": "NK", "china": "CN", "russia": "RU", "iran": "IR", "brazil": "BR",
    "united states": "US", "india": "IN", "pakistan": "PK", "israel": "IL",
}
KNOWN_CODES = frozenset(SCENARIO_CODES.values())

_DEFAULTS = {"doctrine": 1, "orbit_model": "lee_j2"}

_NAME_RE = re.compile(
    r"^(?:(?P<test>TEST)-)?(?P<tgt>[A-Z][A-Z0-9]{1,5}?)(?P<principal>P?)"
    r"-s(?P<s>\d+)-d(?P<d>[\d.]+)-v(?P<v>[\d.]+)-g(?P<g>[\d.]+)-b(?P<b>[\d.]+)"
    r"-k(?P<k>[\d.]+)-i(?P<i>\d+)-(?P<date>\d{6})(?P<seq>[a-z])-(?P<hash>[0-9a-f]{4})$"
)
# A name followed by an optional "_<product>" suffix (e.g. "_result.json").
_FILE_RE = re.compile(r"^(?P<name>.+?-[0-9a-f]{4})(?:_(?P<product>.+))?$")


def _num(x: float | int) -> str:
    """Compact, round-trip-safe number text: 6.0 -> '6', 6.5 -> '6.5', 0.25 -> '0.25'."""
    f = float(x)
    return str(int(f)) if f == int(f) else f"{f:.6g}"



# Worker-side defaults for config keys a launcher may omit (must match batch_worker's
# _CONFIG_DEFAULTS, because the run name is computed from the *effective* config).
_CFG_DEFAULTS: dict[str, Any] = {
    "detection_time_s": 0.0, "decision_time_s": 0.0, "target_missile_burnout_time_s": 170.0,
    "a_g": 10.0, "intercept_alt_km": 200.0, "nd_repeat": 3, "altitude_min_km": 200.0,
    "anchor_lat_margin_deg": 0.0, "raan_step_deg": 0.25, "margin_multiplier": 5.0,
    "trim_margin_multiplier": 2.0, "max_seeds_safety_cap": 200, "target_shell_n_points": 15000,
    "is_principal_set": False, "is_test_run": False, "target_mode": "country",
}


def settings_from_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Run-identity settings from a batch_worker/cloud_run config dict (defaults filled in).

    The worker, the launchers and the run browser all call this, so they agree on the name.
    """
    c = {**_CFG_DEFAULTS, **{k: v for k, v in cfg.items() if v is not None}}
    st: dict[str, Any] = {k: c[k] for k in (
        "salvo_size", "detection_time_s", "v_bo_km_s", "a_g", "target_missile_burnout_time_s",
        "decision_time_s", "n_interceptors_per_sat", "intercept_alt_km", "nd_repeat",
        "altitude_min_km", "anchor_lat_margin_deg", "raan_step_deg", "margin_multiplier",
        "trim_margin_multiplier")}
    st["max_seeds"] = c["max_seeds_safety_cap"]
    st["orbit_model"] = "lee_j2"
    st["scenario"] = c.get("scenario")
    st["principal"] = bool(c["is_principal_set"])
    st["test"] = bool(c["is_test_run"])
    if c["target_mode"] == "prespecified":
        st["target_points"] = c["target_points"]
    else:
        st["target_country"] = c.get("target_country")
        st["target_shell_n_points"] = c["target_shell_n_points"]
    return st


def scenario_code(settings: Mapping[str, Any]) -> tuple[str, bool]:
    """(code, principal) for a settings dict.  Falls back to country name, else 'PS'."""
    code = str(settings.get("scenario") or "").strip().upper()
    if not code:
        c = settings.get("target_country")
        if isinstance(c, (list, tuple)):
            c = "+".join(sorted(str(x) for x in c))
        code = SCENARIO_CODES.get(str(c or "").strip().lower(), "")
        if not code and c:
            code = re.sub(r"[^A-Z0-9]", "", str(c).upper())[:3] or "XX"
    if not code:
        code = "PS"
    code = re.sub(r"[^A-Z0-9]", "", code)[:5] or "XX"
    return code, bool(settings.get("principal", settings.get("is_principal_set", False)))


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


def _canon_points(points: Iterable[Any]) -> list[list[Any]]:
    out = []
    for p in points:
        name, lat, lon = (p[0], p[1], p[2]) if not isinstance(p, Mapping) else (
            p.get("name", ""), p["lat_deg"], p["lon_deg"])
        out.append([round(float(lat), 4), round(float(lon), 4)])
    return sorted(out)


def _hash4(settings: Mapping[str, Any]) -> str:
    h: dict[str, Any] = {}
    for k in _HASHED_SCALARS:
        v = settings.get(k, _DEFAULTS.get(k))
        if v is None:
            continue
        if isinstance(v, (list, tuple)):
            v = sorted(str(x) for x in v)
        elif isinstance(v, float):
            v = round(v, 6)
        h[k] = v
    pts = settings.get("target_points")
    if pts:
        h["target_points"] = _canon_points(pts)
    blob = json.dumps(h, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:4]


def _readable(settings: Mapping[str, Any]) -> str:
    code, principal = scenario_code(settings)
    parts = [("TEST-" if settings.get("test") else "") + code + ("P" if principal else "")]
    for letter, key, typ, width in _TOKENS:
        if key not in settings:
            raise KeyError(f"run-name setting '{key}' is missing")
        v = settings[key]
        parts.append(f"{letter}{int(round(float(v))):0{width}d}" if typ is int else f"{letter}{_num(v)}")
    return "-".join(parts)


def settings_key(settings: Mapping[str, Any]) -> str:
    """Date-free identity of a run: readable tokens + hash (used for 'does it exist?')."""
    return f"{_readable(settings)}-{_hash4(settings)}"


def encode_run_name(settings: Mapping[str, Any], date: datetime.date | str | None = None,
                    seq: str = "a") -> str:
    """Build the run name.  ``date`` defaults to today (UTC); ``seq`` is a-z for same-day reruns."""
    if date is None:
        date = datetime.datetime.now(datetime.timezone.utc).date()
    if isinstance(date, (datetime.date, datetime.datetime)):
        d = date.strftime("%y%m%d")
    else:
        d = str(date).replace("-", "")[-6:]
    if not re.fullmatch(r"[a-z]", seq):
        raise ValueError("seq must be a single letter a-z")
    return f"{_readable(settings)}-{d}{seq}-{_hash4(settings)}"


def next_seq(existing_names: Iterable[str], settings: Mapping[str, Any],
             date: datetime.date | None = None) -> str:
    """First unused sequence letter for these settings on this date."""
    if date is None:
        date = datetime.datetime.now(datetime.timezone.utc).date()
    key, d = settings_key(settings), date.strftime("%y%m%d")
    used = {m["seq"] for n in existing_names if (m := _try_decode(n)) and m["key"] == key and m["date"] == d}
    for c in "abcdefghijklmnopqrstuvwxyz":
        if c not in used:
            return c
    raise RuntimeError("more than 26 runs with identical settings in one day")


def local_run_id(settings: Mapping[str, Any], folder: str | Path) -> str:
    """Run name for a local run: today's date plus the first sequence letter unused in ``folder``."""
    date = datetime.datetime.now(datetime.timezone.utc).date()
    names = [p.name for p in Path(folder).glob("*")] if Path(folder).exists() else []
    parsed = [x[0] for n in names if (x := parse_filename(n))]
    return encode_run_name(settings, date, next_seq(parsed, settings, date))


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
        "salvo_size": int(g["s"]), "detection_time_s": float(g["d"]), "v_bo_km_s": float(g["v"]),
        "a_g": float(g["g"]), "target_missile_burnout_time_s": float(g["b"]),
        "decision_time_s": float(g["k"]), "n_interceptors_per_sat": int(g["i"]),
        "date": g["date"], "date_iso": d.isoformat(), "seq": g["seq"], "hash": g["hash"],
        "key": re.sub(r"-\d{6}[a-z]-", "-", stem),
    }


def parse_filename(filename: str) -> tuple[str, str] | None:
    """('<name>', '<product>') for e.g. 'CN-...-3f9c_result.json' -> (name, 'result.json')."""
    base = Path(filename).name
    m = _FILE_RE.match(base)
    if not m or not _try_decode(m.group("name")):
        return None
    return m.group("name"), (m.group("product") or "")


def matches(info: Mapping[str, Any], **query: Any) -> bool:
    """True if a decoded name satisfies every given field (unspecified fields are wildcards).

    Values may be a scalar or a list/set of acceptable values; numbers compare with a
    1e-9 tolerance.  e.g. matches(info, scenario="CN", detection_time_s=[15, 30])
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
            else:
                ok |= str(have).upper() == str(w).upper() if k == "scenario" else have == w
        if not ok:
            return False
    return True


def find_runs(names: Iterable[str], **query: Any) -> list[dict[str, Any]]:
    """Decode and filter names; partial queries allowed.  Newest first."""
    out = [i for n in set(names) if (i := _try_decode(n)) and matches(i, **query)]
    return sorted(out, key=lambda i: (i["date"], i["seq"]), reverse=True)


def _cli(argv: list[str]) -> int:
    if len(argv) < 2 or argv[0] not in {"encode", "decode"}:
        print(__doc__)
        return 2
    if argv[0] == "decode":
        print(json.dumps(decode_run_name(argv[1]), indent=2))
    else:
        settings = json.loads(Path(argv[1]).read_text())
        print(encode_run_name(settings.get("settings", settings)))
    return 0


if __name__ == "__main__":
    sys.exit(_cli(sys.argv[1:]))
