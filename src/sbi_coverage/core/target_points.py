"""Prespecified target-point helpers: loading a named lat/lon list from a CSV
file, and computing a geometry-aware inclination sweep floor from that list.
"""
from __future__ import annotations

import csv
from pathlib import Path

# Case/whitespace-tolerant column aliases -- real user CSVs (e.g. exported
# from a spreadsheet) commonly have trailing spaces on both header names and
# cell values, and no fixed convention for which of "Lat"/"Latitude" etc. is
# used.
_NAME_ALIASES = {"location", "name"}
_LAT_ALIASES = {"lat", "latitude"}
_LON_ALIASES = {"lon", "long", "longitude"}


def load_target_points_csv(path: str | Path) -> list[list]:
    """Load a prespecified target-point list from a CSV file.

    Expected columns (header names are matched case-insensitively, after
    stripping whitespace): a name/location column and a lat/lon pair, e.g.:

        Location,Lat,Lon
        Sohae Satellite Launching Station/Pongdong-ri,39.66026497,124.7050276
        Sinpung-dong Missile Operating Base/Taegwan ,40.301111 ,125.268056

    Every field (header names and cell values) is stripped of surrounding
    whitespace before use -- real exported CSVs routinely have trailing
    spaces on both names and coordinates, as in the example above.

    Returns the same [[name, lat, lon], ...] shape already used everywhere
    else in the repo as a TARGET_POINTS list (list, not tuple, so it's
    directly JSON-serializable without a conversion step).

    Raises ValueError (naming the file/row) if a required column is missing
    or a row's lat/lon can't be parsed as a float.
    """
    path = Path(path)
    with path.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"load_target_points_csv: {path} has no header row")

        header_by_norm = {h.strip().lower(): h for h in reader.fieldnames}

        def _find_column(aliases: set[str], label: str) -> str:
            for alias in aliases:
                if alias in header_by_norm:
                    return header_by_norm[alias]
            raise ValueError(
                f"load_target_points_csv: {path} is missing a {label} column "
                f"(looked for one of {sorted(aliases)} in header {reader.fieldnames!r})"
            )

        name_col = _find_column(_NAME_ALIASES, "name/location")
        lat_col = _find_column(_LAT_ALIASES, "latitude")
        lon_col = _find_column(_LON_ALIASES, "longitude")

        points: list[list] = []
        for row_num, row in enumerate(reader, start=2):  # row 1 is the header
            name = str(row[name_col]).strip()
            try:
                lat = float(str(row[lat_col]).strip())
                lon = float(str(row[lon_col]).strip())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"load_target_points_csv: {path} row {row_num} has an unparseable "
                    f"lat/lon ({row.get(lat_col)!r}, {row.get(lon_col)!r})"
                ) from exc
            points.append([name, lat, lon])

    if not points:
        raise ValueError(f"load_target_points_csv: {path} contains no data rows")
    return points
