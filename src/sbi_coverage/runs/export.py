"""Exporting tables and run data (CSV/JSON) for the GUI: in-memory bytes plus an optional save to a local folder."""
from __future__ import annotations

import datetime
import io
import json
import re
from pathlib import Path
from typing import Any, Iterable

DEFAULT_EXPORT_DIR = Path.home() / "Downloads" / "SBI_exports"


def df_csv_bytes(df, columns: Iterable[str] | None = None) -> bytes:
    """UTF-8 CSV of a DataFrame (optionally a column subset; list/dict cells are written as JSON text)."""
    out = df[list(columns)] if columns is not None else df
    out = out.copy()
    for c in out.columns:
        if out[c].map(lambda v: isinstance(v, (dict, list))).any():
            out[c] = out[c].map(lambda v: json.dumps(v) if isinstance(v, (dict, list)) else v)
    buf = io.StringIO()
    out.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def json_bytes(obj: Any) -> bytes:
    """Pretty JSON bytes; datetimes and numpy values are converted, NaN becomes null."""
    return json.dumps(_clean(obj), indent=2).encode("utf-8")


def _clean(o: Any) -> Any:
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (datetime.datetime, datetime.date)):
        return o.isoformat()
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            return _clean(o.item())
        except Exception:
            return str(o)
    if isinstance(o, float) and (o != o or o in (float("inf"), float("-inf"))):
        return None
    return o


def safe_filename(name: str) -> str:
    """Filesystem-safe file name (keeps letters, digits, . _ - and replaces everything else with _)."""
    return re.sub(r"[^A-Za-z0-9._-]", "_", name).strip("._") or "export"


def validate_export_dir(folder: str) -> tuple[Path | None, str | None]:
    """(path, None) if usable, else (None, guidance). The folder is created if missing."""
    if not folder or not folder.strip():
        return None, "Enter a folder path to save into (e.g. your Downloads folder)."
    p = Path(folder.strip()).expanduser()
    try:
        p.mkdir(parents=True, exist_ok=True)
        probe = p / ".sbi_write_test"
        probe.write_text("ok")
        probe.unlink()
    except OSError as exc:
        return None, f"Cannot write to '{p}': {exc.strerror or exc}. Choose a folder you have permission to write to."
    return p, None


def save_export(folder: str | Path, filename: str, data: bytes) -> Path:
    """Write ``data`` to ``folder/filename`` (folder created if needed); returns the path. Existing files are not overwritten:
    a numeric suffix is added instead."""
    folder = Path(folder).expanduser()
    folder.mkdir(parents=True, exist_ok=True)
    stem, suffix = Path(safe_filename(filename)).stem, Path(safe_filename(filename)).suffix
    target, k = folder / f"{stem}{suffix}", 1
    while target.exists():
        target = folder / f"{stem}_{k}{suffix}"
        k += 1
    target.write_bytes(data)
    return target
