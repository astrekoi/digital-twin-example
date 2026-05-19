"""Factories for IPFS providers and Hedera client based on env config."""
from __future__ import annotations

from src.config import HederaConfig, IntegrationConfig, load_env_config
from src.integrations.hedera import HederaClient
from src.integrations.ipfs import IPFSProvider
from src.integrations.ipfs.local_node import LocalKuboProvider
from src.integrations.ipfs.pinata import PinataProvider
from src.integrations.ipfs.web3_storage import Web3StorageProvider


def create_ipfs_provider(
    integrations: IntegrationConfig | None = None,
) -> IPFSProvider | None:
    cfg = integrations or load_env_config().integrations
    if not cfg.ipfs_enabled or cfg.ipfs_provider == "none":
        return None
    if cfg.ipfs_provider == "pinata":
        return PinataProvider(
            jwt=cfg.pinata_jwt,
            api_base=cfg.pinata_api_base,
            allow_external_api_calls=cfg.allow_external_api_calls,
        )
    if cfg.ipfs_provider == "web3_storage":
        return Web3StorageProvider(token=cfg.web3_storage_token)
    if cfg.ipfs_provider == "local":
        return LocalKuboProvider(api_url=cfg.local_ipfs_api_url)
    return None


def create_hedera_client(hedera: HederaConfig | None = None) -> HederaClient | None:
    cfg = hedera or load_env_config().hedera
    if not cfg.enabled:
        return None
    if not (cfg.operator_id and cfg.operator_key):
        return None
    return HederaClient(
        network=cfg.network,
        operator_id=cfg.operator_id,
        operator_key_hex=cfg.operator_key,
        tx_fee_hbar=cfg.tx_fee_hbar,
        mirror_node_url=cfg.mirror_node_url,
    )
