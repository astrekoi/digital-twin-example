"""web3.storage IPFS provider - extension point for future use.

The provider is wired into the factory but does not perform real network
calls. Selecting ``IPFS_PROVIDER=web3_storage`` returns ``status='disabled'``
with a clear message; this lets operators flip providers without code edits.
"""
from __future__ import annotations

from typing import Any

from src.integrations.base import IntegrationResult
from src.integrations.ipfs import IPFSProvider


class Web3StorageProvider(IPFSProvider):
    name = "web3_storage"

    def __init__(self, token: str | None) -> None:
        self._token = (token or "").strip()

    def is_configured(self) -> bool:
        return bool(self._token)

    def pin_json(self, payload: dict[str, Any], name: str) -> IntegrationResult:
        return IntegrationResult(
            status="disabled",
            message="web3_storage provider is not active in this build; switch IPFS_PROVIDER to pinata",
            provider=self.name,
        )
