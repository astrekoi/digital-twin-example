"""Shared types for optional IoT/storage integrations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class IntegrationDisabled(RuntimeError):
    """Raised by future adapters when an explicitly requested integration is off."""


@dataclass(frozen=True)
class IntegrationResult:
    status: str
    message: str = ""
    provider: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PinResult:
    cid: str
    provider: str
    metadata: dict[str, Any] = field(default_factory=dict)
