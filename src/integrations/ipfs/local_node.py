"""Local Kubo IPFS node provider - extension point for self-hosted IPFS.

Selecting ``IPFS_PROVIDER=local`` returns ``status='disabled'`` with a clear
message. Operators with a running Kubo daemon can plug in a custom
implementation here without touching the rest of the pipeline.
"""
from __future__ import annotations

from typing import Any

from src.integrations.base import IntegrationResult
from src.integrations.ipfs import IPFSProvider


class LocalKuboProvider(IPFSProvider):
    name = "local"

    def __init__(self, api_url: str = "http://127.0.0.1:5001") -> None:
        self._api_url = api_url.rstrip("/")

    def is_configured(self) -> bool:
        return bool(self._api_url)

    def pin_json(self, payload: dict[str, Any], name: str) -> IntegrationResult:
        return IntegrationResult(
            status="disabled",
            message="local kubo provider is not active in this build; switch IPFS_PROVIDER to pinata",
            provider=self.name,
        )
