"""Topic-level helpers for HCS: create_topic exposed as a stand-alone function
so scripts/hedera_poc.py can use it without instantiating a HederaClient.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.integrations.hedera.client import HederaClient

logger = logging.getLogger(__name__)


def create_topic(client: "HederaClient", memo: str = "plc-twin-thesis") -> str:
    import hiero_sdk_python as h
    sdk_client = client._build_client()  # noqa: SLF001 - we own this layer
    tx = h.TopicCreateTransaction().set_memo(memo)
    receipt = tx.execute(sdk_client)
    topic_id = getattr(receipt, "topic_id", None)
    if topic_id is None:
        raise RuntimeError("TopicCreate receipt has no topic_id")
    return str(topic_id)
