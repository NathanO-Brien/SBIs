"""Prespecified target-point helpers: parse/validate a named lat/lon list from CSV text or a CSV file."""
from __future__ import annotations

import csv
import io
import math
from pathlib import Path

# Case/whitespace-tolerant column aliases -- real user CSVs (e.g. exported from a spreadsheet) commonly have
# trailing spaces on header names and cell values, and no fixed convention for "Lat" vs "Latitude" etc.
_NAME_ALIASES = ("location", "name", "target", "site", "label")
_LAT_ALIASES = ("lat", "latitude")
_LON_ALIASES = ("lon", "long", "lng", "longitude")

MAX_ROW_ERRORS_SHOWN = 8
MAX_POINTS = 20000
SOFT_POINT_LIMIT = 500   # above this the MILP gets large; a warning is returned


class TargetFileError(ValueError):
    """Target file could not be used. ``problems`` lists every issue found (each phrased as guidance)."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        super().__init__("; ".join(self.problems))


def parse_target_points_text(text: str, source: str = "target file") -> tuple[list[list], list[str]]:
    """Parse and validate target points from CSV text.

    Accepts comma, semicolon or tab separated text with a header row naming a name/location column and
    lat/lon columns (any of the aliases above, case-insensitive), or a header-less 3-column file in the
    order name, lat, lon. Every field is stripped of surrounding whitespace.

    Returns ([[name, lat_deg, lon_deg], ...], warnings). Raises TargetFileError listing ALL problems found
    (empty file, missing columns, unparseable or out-of-range coordinates, too many points).
    Longitudes in (180, 360] are accepted and wrapped to [-180, 180] with a warning.
    """
    text = text.lstrip("﻿")
    if not text.strip():
        raise TargetFileError([f"{source} is empty. Upload a CSV with columns Name, Lat, Lon (one target per row)."])
    first_line = next((ln for ln in text.splitlines() if ln.strip()), "")
    delim = max(",;\t", key=first_line.count) if any(d in first_line for d in ",;\t") else ","
    rows = [r for r in csv.reader(io.StringIO(text), delimiter=delim) if any(c.strip() for c in r)]
    if not rows:
        raise TargetFileError([f"{source} has no data rows."])

    header = [c.strip().lower() for c in rows[0]]

    def _find(aliases: tuple[str, ...]) -> int | None:
        for a in aliases:
            if a in header:
                return header.index(a)
        return None

    i_name, i_lat, i_lon = _find(_NAME_ALIASES), _find(_LAT_ALIASES), _find(_LON_ALIASES)
    warnings: list[str] = []
    if i_lat is not None and i_lon is not None:
        data = rows[1:]
        first_row = 2
    else:
        # header-less? first row must look like name, number, number
        def _num(x: str) -> bool:
            try:
                float(x.strip())
                return True
            except ValueError:
                return False
        if len(rows[0]) >= 3 and _num(rows[0][1]) and _num(rows[0][2]):
            i_name, i_lat, i_lon, data, first_row = 0, 1, 2, rows, 1
            warnings.append("No header row found; read columns as name, lat, lon.")
        else:
            missing = [lbl for lbl, idx in (("latitude (Lat/Latitude)", i_lat), ("longitude (Lon/Long/Longitude)", i_lon)) if idx is None]
            raise TargetFileError([
                f"{source} header {[c.strip() for c in rows[0]]} is missing a " + " and a ".join(missing) + " column.",
                "Use a header such as: Name,Lat,Lon (decimal degrees, positive north/east).",
            ])
    if i_name is None:
        warnings.append("No name/location column found; targets are numbered Target 1, Target 2, ...")

    problems: list[str] = []
    points: list[list] = []
    seen: dict[tuple[float, float], int] = {}
    n_wrapped = 0
    for k, r in enumerate(data):
        row_no = first_row + k
        try:
            lat = float(r[i_lat].strip())
            lon = float(r[i_lon].strip())
        except (IndexError, ValueError):
            problems.append(f"row {row_no}: latitude/longitude not a number ({r!r}). Use decimal degrees, e.g. 39.66,124.70.")
            continue
        if not (math.isfinite(lat) and math.isfinite(lon)):
            problems.append(f"row {row_no}: latitude/longitude must be finite numbers.")
            continue
        if not -90.0 <= lat <= 90.0:
            problems.append(f"row {row_no}: latitude {lat} is outside [-90, 90]. Are Lat and Lon swapped?")
            continue
        if not -180.0 <= lon <= 360.0:
            problems.append(f"row {row_no}: longitude {lon} is outside [-180, 360].")
            continue
        if lon > 180.0:
            lon -= 360.0
            n_wrapped += 1
        name = r[i_name].strip() if i_name is not None and i_name < len(r) and r[i_name].strip() else f"Target {len(points) + 1}"
        key = (round(lat, 5), round(lon, 5))
        if key in seen:
            warnings.append(f"row {row_no}: same coordinates as row {seen[key]} ({name!r}); duplicates add no coverage requirement.")
        seen.setdefault(key, row_no)
        points.append([name, lat, lon])
    if problems:
        shown = problems[:MAX_ROW_ERRORS_SHOWN]
        if len(problems) > MAX_ROW_ERRORS_SHOWN:
            shown.append(f"... and {len(problems) - MAX_ROW_ERRORS_SHOWN} more row problem(s).")
        raise TargetFileError(shown)
    if not points:
        raise TargetFileError([f"{source} has a header but no data rows."])
    if len(points) > MAX_POINTS:
        raise TargetFileError([f"{len(points)} targets exceeds the limit of {MAX_POINTS}. Reduce the set or use country mode."])
    if n_wrapped:
        warnings.append(f"{n_wrapped} longitude(s) above 180 were wrapped to the -180..180 range.")
    if len(points) > SOFT_POINT_LIMIT:
        warnings.append(f"{len(points)} targets is a large problem; solves get slow and need much more memory above ~{SOFT_POINT_LIMIT}.")
    return points, warnings


def load_target_points_csv(path: str | Path) -> list[list]:
    """Load a prespecified target-point list from a CSV file ([[name, lat, lon], ...]).

    Same parsing and validation as parse_target_points_text; raises ValueError (TargetFileError) naming
    the file and the problems found.
    """
    path = Path(path)
    points, _ = parse_target_points_text(path.read_text(encoding="utf-8-sig"), source=str(path))
    return points
