"""Environment helpers for optional .env-based configuration.

No .env file is loaded at import time. Call load_project_env() explicitly from
entry points or config loaders. If python-dotenv is absent, os.environ remains
the only source and imports still work.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping

TRUE_VALUES = {"1", "true", "yes", "y", "on"}
FALSE_VALUES = {"0", "false", "no", "n", "off"}


def load_project_env(dotenv_path: str | Path = ".env") -> bool:
    """Load a .env file when python-dotenv is installed.

    Returns True when python-dotenv loaded an existing file. Returns False when
    the dependency or file is absent. This function never raises on a missing
    python-dotenv package.
    """
    path = Path(dotenv_path)
    if not path.exists():
        return False

    try:
        from dotenv import load_dotenv
    except ImportError:
        return False

    return bool(load_dotenv(dotenv_path=path, override=False))


def _getenv(name: str, default: str | None = None, env: Mapping[str, str] | None = None) -> str | None:
    source = os.environ if env is None else env
    value = source.get(name)
    if value is None or value == "":
        return default
    return value


def get_env_str(
    name: str,
    default: str | None = None,
    env: Mapping[str, str] | None = None,
) -> str | None:
    """Return an environment string, treating unset and empty as default."""
    return _getenv(name, default=default, env=env)


def get_env_bool(
    name: str,
    default: bool = False,
    env: Mapping[str, str] | None = None,
) -> bool:
    """Parse bool values from true/false, 1/0, yes/no, on/off."""
    raw = _getenv(name, env=env)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"Invalid boolean env value for {name}: {raw!r}")


def get_env_int(
    name: str,
    default: int | None = None,
    env: Mapping[str, str] | None = None,
) -> int | None:
    """Parse an integer env value."""
    raw = _getenv(name, env=env)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid integer env value for {name}: {raw!r}") from exc


def get_env_path(
    name: str,
    default: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> Path | None:
    """Parse a path env value."""
    raw_default = str(default) if default is not None else None
    raw = _getenv(name, default=raw_default, env=env)
    return Path(raw) if raw is not None else None
