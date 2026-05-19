"""IPFS provider abstraction.

Concrete providers:
- PinataProvider - production: HTTP via httpx, Bearer JWT, retries with backoff.
- Web3StorageProvider - extension point; reports status='disabled' until activated.
- LocalKuboProvider - extension point; reports status='disabled' until activated.

Selection happens in src/integrations/factory.create_ipfs_provider() based on
IPFS_PROVIDER (.env). Switching providers requires no code changes - just a
.env edit and stack restart.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from src.integrations.base import IntegrationResult


class IPFSProvider(ABC):
    """Pluggable IPFS pinning backend."""

    name: str = "abstract"

    @abstractmethod
    def is_configured(self) -> bool:
        """Return True if the provider has all required credentials."""

    @abstractmethod
    def pin_json(self, payload: dict[str, Any], name: str) -> IntegrationResult:
        """Pin a JSON document. Returns IntegrationResult with cid in metadata."""
