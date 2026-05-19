"""Hedera Consensus Service PoC: balance check, create topic, publish test message.

Run from repo root with the .venv-plc-twin venv active. Reads operator credentials
from .env in the project root.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def _load_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        sys.exit(f"FATAL: {ENV_PATH} not found")
    env: dict[str, str] = {}
    for line in ENV_PATH.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        env[k.strip()] = v.strip()
    return env


def _require(env: dict[str, str], key: str) -> str:
    v = env.get(key, "").strip()
    if not v:
        sys.exit(f"FATAL: {key} is empty in {ENV_PATH}")
    return v


def _build_client(env: dict[str, str]):
    from hiero_sdk_python import AccountId, Client, PrivateKey

    network = env.get("HEDERA_NETWORK", "testnet").lower()
    if network == "testnet":
        client = Client.for_testnet()
    elif network == "mainnet":
        client = Client.for_mainnet()
    elif network == "previewnet":
        client = Client.for_previewnet()
    else:
        sys.exit(f"FATAL: unknown HEDERA_NETWORK={network}")

    op_id = AccountId.from_string(_require(env, "HEDERA_OPERATOR_ID"))
    raw_key = _require(env, "HEDERA_OPERATOR_KEY")
    if raw_key.startswith("0x") or raw_key.startswith("0X"):
        raw_key = raw_key[2:]
    op_key = PrivateKey.from_string_ecdsa(raw_key)
    client.set_operator(op_id, op_key)
    return client, op_id, op_key


def _set_env_var(key: str, value: str) -> None:
    text = ENV_PATH.read_text()
    pattern = re.compile(rf"^{re.escape(key)}=.*$", re.MULTILINE)
    if pattern.search(text):
        text = pattern.sub(f"{key}={value}", text)
    else:
        text = text.rstrip() + f"\n{key}={value}\n"
    ENV_PATH.write_text(text)


def cmd_balance(env: dict[str, str]) -> int:
    from hiero_sdk_python import CryptoGetAccountBalanceQuery

    client, op_id, _ = _build_client(env)
    try:
        balance = CryptoGetAccountBalanceQuery().set_account_id(op_id).execute(client)
        hbars = getattr(balance, "hbars", None)
        tinybars = hbars.to_tinybars() if hbars is not None else 0
        print(f"account     {op_id}")
        print(f"hbars       {hbars}")
        print(f"tinybars    {tinybars}")
        if tinybars <= 0:
            sys.exit("FATAL: balance is non-positive - operator key invalid or out of HBAR")
        print("OK: balance > 0")
        return 0
    finally:
        client.close()


def cmd_create_topic(env: dict[str, str], memo: str) -> int:
    from hiero_sdk_python import TopicCreateTransaction

    client, _, _ = _build_client(env)
    try:
        tx = TopicCreateTransaction().set_memo(memo)
        receipt = tx.execute(client)
        topic_id = receipt.topic_id
        if topic_id is None:
            sys.exit("FATAL: receipt has no topic_id")
        topic_str = str(topic_id)
        print(f"topic_id    {topic_str}")
        print(f"memo        {memo}")
        _set_env_var("HEDERA_TOPIC_ID", topic_str)
        print(f"OK: HEDERA_TOPIC_ID written to {ENV_PATH}")
        return 0
    finally:
        client.close()


def cmd_publish_test(env: dict[str, str], message: str) -> int:
    from hiero_sdk_python import TopicId, TopicMessageSubmitTransaction

    topic_str = _require(env, "HEDERA_TOPIC_ID")
    topic_id = TopicId.from_string(topic_str)
    client, _, _ = _build_client(env)
    try:
        tx = TopicMessageSubmitTransaction().set_topic_id(topic_id).set_message(message)
        receipt = tx.execute(client)
        seq = receipt.topic_sequence_number
        tx_id = receipt.transaction_id
        print(f"topic_id    {topic_str}")
        print(f"sequence    {seq}")
        print(f"transaction {tx_id}")
        print("sleeping 6s for mirror-node lag ...")
        time.sleep(6)

        mirror_url = env.get(
            "HEDERA_MIRROR_NODE_URL", "https://testnet.mirrornode.hedera.com"
        ).rstrip("/")
        url = f"{mirror_url}/api/v1/topics/{topic_str}/messages?limit=5&order=desc"
        with urllib.request.urlopen(url, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        messages = data.get("messages", [])
        match = next((m for m in messages if m.get("sequence_number") == seq), None)
        if match is None:
            sys.exit(
                f"FATAL: sequence {seq} not visible in mirror node yet (last 5: {[m.get('sequence_number') for m in messages]})"
            )
        print(
            f"OK: mirror node confirms sequence={seq}, consensus_timestamp={match.get('consensus_timestamp')}"
        )
        return 0
    finally:
        client.close()


def main() -> int:
    env = _load_env()
    parser = argparse.ArgumentParser(description="Hedera HCS PoC")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("balance", help="Print operator account balance, fail if <=0")
    p_create = sub.add_parser(
        "create_topic", help="Create a topic and write HEDERA_TOPIC_ID to .env"
    )
    p_create.add_argument("--memo", default="plc-twin-thesis", help="topic memo")
    p_publish = sub.add_parser("publish_test", help="Publish a UTF-8 message to HEDERA_TOPIC_ID")
    p_publish.add_argument("message", help="UTF-8 message text")
    args = parser.parse_args()

    if args.cmd == "balance":
        return cmd_balance(env)
    if args.cmd == "create_topic":
        return cmd_create_topic(env, args.memo)
    if args.cmd == "publish_test":
        return cmd_publish_test(env, args.message)
    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
