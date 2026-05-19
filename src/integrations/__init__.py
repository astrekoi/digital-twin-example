"""Optional integration adapters: IPFS providers and Hedera Consensus Service.

Both gated by ALLOW_EXTERNAL_API_CALLS=true at runtime; default is no-op.
"""
from __future__ import annotations

from .base import IntegrationDisabled, IntegrationResult, PinResult
from .factory import create_hedera_client, create_ipfs_provider
from .hedera import HederaClient
from .ipfs import IPFSProvider

__all__ = [
    "IntegrationDisabled",
    "IntegrationResult",
    "PinResult",
    "IPFSProvider",
    "HederaClient",
    "create_ipfs_provider",
    "create_hedera_client",
]
