"""Code/library/machine provenance recorded in every result JSON."""
from __future__ import annotations

import os
import platform
import socket
import subprocess
from importlib import metadata
from pathlib import Path
from typing import Any

_LIBS = ("numpy", "scipy", "numba", "gurobipy", "boto3", "pandas", "geopandas", "shapely")


def _git(*args: str) -> str | None:
    repo = Path(__file__).resolve().parents[3]
    try:
        out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def collect_provenance() -> dict[str, Any]:
    """Git commit (``GIT_COMMIT`` env var wins -- set it as a Docker build arg, since the image has no
    .git), library versions, and where/how the run executed (AWS Batch job id when present)."""
    commit = os.environ.get("GIT_COMMIT") or _git("rev-parse", "HEAD")
    dirty = None
    if not os.environ.get("GIT_COMMIT"):
        status = _git("status", "--porcelain", "--untracked-files=no")
        dirty = None if status is None else bool(status)
    libs: dict[str, str | None] = {}
    for name in _LIBS:
        try:
            libs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            libs[name] = None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "python": platform.python_version(),
        "libraries": libs,
        "host": socket.gethostname(),
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "aws_batch_job_id": os.environ.get("AWS_BATCH_JOB_ID"),
        "image": os.environ.get("IMAGE_URI"),
    }
