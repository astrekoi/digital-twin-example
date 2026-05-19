"""Detect process role from argv (Celery --hostname=hw@%h, ai@%h, ext@%h),
TAIPY_ROLE env var, or fall back to 'beat' / 'unknown'.

Used by storage.factory to size Postgres connection pools per role
(POOL_MAX_HW / POOL_MAX_AI / POOL_MAX_EXT / POOL_MAX_TAIPY).
"""

from __future__ import annotations

import os
import sys
from typing import Literal

Role = Literal["hw", "ai", "ext", "taipy", "beat", "unknown"]

_HOSTNAME_PREFIXES: tuple[tuple[str, Role], ...] = (
    ("hw@", "hw"),
    ("ai@", "ai"),
    ("ext@", "ext"),
)


def _argv_hostname() -> str | None:
    argv = sys.argv
    for i, a in enumerate(argv):
        if a == "--hostname" and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith("--hostname="):
            return a.split("=", 1)[1]
    return None


def detect_role() -> Role:
    explicit = os.environ.get("TAIPY_ROLE", "").strip().lower()
    if explicit in ("hw", "ai", "ext", "taipy", "beat"):
        return explicit  # type: ignore[return-value]

    hostname = _argv_hostname() or ""
    for prefix, role in _HOSTNAME_PREFIXES:
        if hostname.startswith(prefix):
            return role

    argv0 = sys.argv[0] if sys.argv else ""
    if "celery" in argv0 and "beat" in " ".join(sys.argv):
        return "beat"
    if "app.web" in " ".join(sys.argv) or "taipy" in argv0.lower():
        return "taipy"

    return "unknown"


def pool_max_for_role(role: Role) -> int:
    """Return Postgres pool max size for the given process role.

    Defaults match .env.example: POOL_MAX_HW=2, POOL_MAX_AI=2,
    POOL_MAX_EXT=4, POOL_MAX_TAIPY=8. beat/unknown use 1.
    """
    key_map: dict[Role, str] = {
        "hw": "POOL_MAX_HW",
        "ai": "POOL_MAX_AI",
        "ext": "POOL_MAX_EXT",
        "taipy": "POOL_MAX_TAIPY",
    }
    default_map: dict[Role, int] = {
        "hw": 2,
        "ai": 2,
        "ext": 4,
        "taipy": 8,
        "beat": 1,
        "unknown": 1,
    }
    if role in key_map:
        raw = os.environ.get(key_map[role], "").strip()
        if raw:
            try:
                return max(1, int(raw))
            except ValueError:
                pass
    return default_map[role]
