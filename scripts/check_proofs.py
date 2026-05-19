"""Audit proof-of-existence rows against Hedera Mirror Node + IPFS.

For each local proof row, fetches the HCS message via Mirror REST and
confirms consensus_timestamp matches. With --verify-payload also recomputes
blake2b on the message body and fetches the embedded CID from the public
gateway to confirm structural overlap.

Exit 1 if any audited proof fails; 0 otherwise.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import sys
import urllib.request
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_env_config  # noqa: E402  (sys.path bootstrap above)
from src.env import load_project_env  # noqa: E402
from src.storage.factory import create_telemetry_store  # noqa: E402


def _http_get_json(url: str, timeout: float = 15.0) -> dict[str, Any] | None:
    req = urllib.request.Request(url, headers={"User-Agent": "plc-twin/check_proofs"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except Exception:
        return None


def _http_get_bytes(url: str, timeout: float = 30.0) -> bytes | None:
    req = urllib.request.Request(url, headers={"User-Agent": "plc-twin/check_proofs"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def _audit_one(
    proof: dict,
    *,
    mirror_base: str,
    gateway_base: str,
    verify_payload: bool,
) -> tuple[bool, list[str]]:
    """Return (ok, list_of_status_lines)."""
    out: list[str] = []
    topic = proof.get("hedera_topic_id")
    seq = proof.get("hedera_sequence_number")
    if topic is None or seq is None:
        out.append(f"  {proof['idempotency_key']}: not yet anchored to Hedera (skip)")
        return True, out

    mirror_url = f"{mirror_base}/api/v1/topics/{topic}/messages/{seq}"
    data = _http_get_json(mirror_url)
    if data is None:
        out.append(f"  {proof['idempotency_key']}: MIRROR FAIL - {mirror_url}")
        return False, out

    mirror_consensus = data.get("consensus_timestamp")
    if not mirror_consensus:
        out.append(f"  {proof['idempotency_key']}: mirror returned no consensus_timestamp")
        return False, out

    local_consensus = str(proof.get("hedera_consensus_ts") or "").strip()
    if local_consensus and local_consensus != mirror_consensus:
        out.append(
            f"  {proof['idempotency_key']}: CONSENSUS MISMATCH local={local_consensus} mirror={mirror_consensus}"
        )
        return False, out
    out.append(
        f"  {proof['idempotency_key']}: ✓ topic={topic} seq={seq} consensus={mirror_consensus}"
    )

    if not verify_payload:
        return True, out

    # message body is base64-encoded UTF-8 JSON in mirror node responses
    msg_b64 = data.get("message")
    if not msg_b64:
        out.append("    payload: mirror returned no message body")
        return False, out
    try:
        body_bytes = base64.b64decode(msg_b64)
        body = json.loads(body_bytes.decode("utf-8"))
    except Exception as exc:
        out.append(f"    payload: decode failed: {exc}")
        return False, out

    body_blake2b = body.get("blake2b")
    if not body_blake2b:
        out.append("    payload: message has no blake2b field")
        return False, out

    # src/integrations/hedera/digest.py hashes the dict *before* adding "blake2b" -
    # strip it before recomputing or the result won't match.
    body_for_hash = {k: v for k, v in body.items() if k != "blake2b"}
    canonical = json.dumps(body_for_hash, sort_keys=True, separators=(",", ":")).encode("utf-8")
    recomputed = hashlib.blake2b(canonical, digest_size=32).hexdigest()
    if recomputed != body_blake2b:
        out.append(
            f"    payload: BLAKE2B MISMATCH msg={body_blake2b[:16]}… recomputed={recomputed[:16]}…"
        )
        return False, out
    out.append(f"    payload: ✓ blake2b={body_blake2b[:16]}…")

    cid = body.get("cid") or proof.get("pinata_cid")
    if not cid:
        out.append("    ipfs: no CID to verify (skip)")
        return True, out
    ipfs_url = f"{gateway_base.rstrip('/')}/{cid}"
    ipfs_bytes = _http_get_bytes(ipfs_url)
    if ipfs_bytes is None:
        out.append(f"    ipfs: gateway fetch failed {ipfs_url}")
        return False, out
    try:
        ipfs_payload = json.loads(ipfs_bytes.decode("utf-8"))
    except Exception as exc:
        out.append(f"    ipfs: decoded {len(ipfs_bytes)} bytes - non-JSON ({exc})")
        return False, out
    # The IPFS pinned digest is the pre-cid digest dict; HCS message contains
    # cid + blake2b on top. Confirm structural overlap.
    if ipfs_payload.get("ts") != body.get("ts") or ipfs_payload.get("n") != body.get("n"):
        out.append("    ipfs: structural mismatch (ts/n differ between IPFS and HCS message)")
        return False, out
    out.append(f"    ipfs: ✓ cid={cid} ts={ipfs_payload.get('ts')} n={ipfs_payload.get('n')}")
    return True, out


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit proofs against Hedera Mirror Node + IPFS")
    parser.add_argument("--limit", type=int, default=20, help="number of recent proofs to audit")
    parser.add_argument(
        "--verify-payload",
        action="store_true",
        help="decode HCS body, recompute blake2b, fetch IPFS payload",
    )
    args = parser.parse_args()

    load_project_env(PROJECT_ROOT / ".env")
    env = load_env_config(PROJECT_ROOT / ".env")

    mirror_base = env.hedera.mirror_node_url
    gateway_base = env.integrations.ipfs_gateway_base

    store = create_telemetry_store()
    store.init_schema()
    proofs = store.fetch_proofs(limit=args.limit)
    if not proofs:
        print("No proofs in storage.")
        return 0

    total = len(proofs)
    ok_count = 0
    print(
        f"Auditing {total} most recent proofs against {mirror_base} (verify_payload={args.verify_payload})"
    )
    for proof in proofs:
        ok, lines = _audit_one(
            proof,
            mirror_base=mirror_base,
            gateway_base=gateway_base,
            verify_payload=args.verify_payload,
        )
        for line in lines:
            print(line)
        if ok:
            ok_count += 1

    print()
    print(f"Result: {ok_count}/{total} proofs pass audit")
    return 0 if ok_count == total else 1


if __name__ == "__main__":
    sys.exit(main())
