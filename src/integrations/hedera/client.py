"""Hedera Consensus Service client wrapper.

Thin abstraction over hiero-sdk-python: builds Client.for_<network>(),
configures operator, exposes submit_message + balance + create_topic.

Operator key format: HEX (64 chars, no 0x prefix) ECDSA secp256k1, or DER.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class TopicSubmitReceipt:
    """Subset of TransactionReceipt fields we care about for proof-of-existence."""
    topic_id: str
    sequence_number: int
    consensus_timestamp: str | None  # unix-seconds string (mirror node returns "1700.123…")
    transaction_id: str


class HederaClient:
    """Lazy wrapper - defers import of hiero_sdk_python until needed."""

    def __init__(
        self,
        *,
        network: str,
        operator_id: str,
        operator_key_hex: str,
        tx_fee_hbar: int = 2,
        mirror_node_url: str = "https://testnet.mirrornode.hedera.com",
    ) -> None:
        self.network = network.lower()
        self.operator_id = operator_id
        # NB: removeprefix (not lstrip!) - lstrip strips a CHAR-SET, so "0bdd…"
        # would lose its leading "0" and produce a 63-char invalid hex key.
        key = operator_key_hex.strip()
        if key.startswith(("0x", "0X")):
            key = key[2:]
        self.operator_key_hex = key
        self.tx_fee_hbar = tx_fee_hbar
        self.mirror_node_url = mirror_node_url.rstrip("/")
        self._client: Any = None

    def _build_client(self) -> Any:
        if self._client is not None:
            return self._client
        import hiero_sdk_python as h
        if self.network == "testnet":
            client = h.Client.for_testnet()
        elif self.network == "mainnet":
            client = h.Client.for_mainnet()
        elif self.network == "previewnet":
            client = h.Client.for_previewnet()
        else:
            raise ValueError(f"unknown HEDERA_NETWORK: {self.network}")
        op_id = h.AccountId.from_string(self.operator_id)
        op_key = h.PrivateKey.from_string_ecdsa(self.operator_key_hex)
        client.set_operator(op_id, op_key)
        self._client = client
        return client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                logger.exception("hedera client close failed")
            self._client = None

    def get_balance_tinybars(self) -> int:
        import hiero_sdk_python as h
        client = self._build_client()
        op_id = h.AccountId.from_string(self.operator_id)
        balance = h.CryptoGetAccountBalanceQuery().set_account_id(op_id).execute(client)
        hbars = getattr(balance, "hbars", None)
        if hbars is None:
            return 0
        return int(hbars.to_tinybars())

    def submit_message(self, topic_id: str, message: str) -> TopicSubmitReceipt:
        """Submit a UTF-8 string to a topic. Returns receipt; raises on error."""
        import hiero_sdk_python as h
        client = self._build_client()
        topic = h.TopicId.from_string(topic_id)
        tx = h.TopicMessageSubmitTransaction().set_topic_id(topic).set_message(message)
        receipt = tx.execute(client)
        seq = int(getattr(receipt, "topic_sequence_number", 0) or 0)
        tx_id = getattr(receipt, "transaction_id", None)
        # Receipt does NOT carry consensus_timestamp - that field comes from a
        # record query or Mirror Node REST. publish_digest_task resolves it via
        # client.fetch_message_consensus_ts(...) immediately after submit.
        return TopicSubmitReceipt(
            topic_id=topic_id,
            sequence_number=seq,
            consensus_timestamp=None,
            transaction_id=str(tx_id) if tx_id else "",
        )

    def fetch_message_consensus_ts(
        self,
        topic_id: str,
        sequence_number: int,
        wait_s: float = 6.0,
    ) -> str | None:
        """Look up consensus_timestamp for a (topic, sequence) via Mirror Node REST.

        Mirror Node has ~3-6s lag; default wait_s=6 is empirically reliable.
        Returns the consensus_timestamp string ("1700123456.123456789") or None.
        """
        import json
        import urllib.request
        if wait_s > 0:
            time.sleep(wait_s)
        url = f"{self.mirror_node_url}/api/v1/topics/{topic_id}/messages?limit=10&order=desc"
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = json.loads(r.read().decode("utf-8"))
        except Exception:
            logger.exception("mirror node lookup failed")
            return None
        for m in data.get("messages", []):
            if m.get("sequence_number") == sequence_number:
                return m.get("consensus_timestamp")
        return None
