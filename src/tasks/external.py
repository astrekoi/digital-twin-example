"""External-side Celery tasks: Pinata pin + Hedera HCS publish + digest chain.

publish_digest_task is the production hot-path:
  1. fetch readings in [window_start, window_start+period) from storage
  2. build digest
  3. pin to IPFS (Pinata) -> cid
  4. submit cid+digest+blake2b to Hedera topic -> receipt
  5. resolve consensus_timestamp via Mirror Node REST
  6. insert_proof (UNIQUE on idempotency_key)

Idempotency: same window_start -> same idempotency_key -> publish_digest_task is
a no-op on re-run; publish_pinata_task / publish_hedera_task also accept
idempotency_key to support manual on-demand publishing.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from src.config import load_env_config
from src.env import load_project_env
from src.integrations.factory import create_hedera_client, create_ipfs_provider
from src.integrations.hedera.digest import (
    attach_cid_and_hash,
    build_digest,
    build_hedera_payload,
    floor_window,
)
from src.storage.factory import create_telemetry_store
from src.tasks.celery_app import celery_app

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_project_env(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)


def _store():
    s = create_telemetry_store()
    s.init_schema()
    return s


def _fail(message: str, *, stage: str) -> dict[str, Any]:
    logger.error("digest publish failed at %s: %s", stage, message)
    return {"status": "error", "stage": stage, "error": message}


# ---------------------------------------------------------------------------
# Pinata pin (on-demand or chained)
# ---------------------------------------------------------------------------


@celery_app.task(name="src.tasks.external.publish_pinata_task", bind=True, max_retries=3)
def publish_pinata_task(self, payload: dict[str, Any], idempotency_key: str) -> dict:
    """Pin a JSON payload via Pinata. Idempotent on idempotency_key.

    On success: writes proof row (pinata_cid only, hedera_* unset). If a proof
    with the same key already exists - returns it, no network call.
    """
    store = _store()
    existing = store.fetch_proof_by_idempotency_key(idempotency_key)
    if existing is not None:
        return {"status": "skipped", "reason": "exists", "proof": existing}

    provider = create_ipfs_provider()
    if provider is None:
        return {"status": "disabled", "reason": "ipfs provider not configured"}
    res = provider.pin_json(payload, name=idempotency_key)
    if res.status != "ok":
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=min(60 * (self.request.retries + 1), 300))
        return {"status": res.status, "message": res.message}
    cid = res.metadata.get("cid")
    if not cid:
        return {"status": "error", "message": "pinata returned no cid"}

    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib
    blake2b = hashlib.blake2b(canonical, digest_size=32).hexdigest()
    try:
        store.insert_proof(
            idempotency_key=idempotency_key,
            digest_ts_unix=time.time(),
            payload_blake2b=blake2b,
            pinata_cid=cid,
        )
    except Exception as exc:
        if "UNIQUE" in str(exc) or "duplicate" in str(exc).lower():
            return {"status": "skipped", "reason": "race"}
        raise
    return {"status": "ok", "cid": cid}


# ---------------------------------------------------------------------------
# Hedera publish (on-demand or chained)
# ---------------------------------------------------------------------------


@celery_app.task(name="src.tasks.external.publish_hedera_task", bind=True, max_retries=3)
def publish_hedera_task(self, message: str, idempotency_key: str, cid: str | None = None) -> dict:
    """Submit a string message to the configured HCS topic. Idempotent."""
    store = _store()
    existing = store.fetch_proof_by_idempotency_key(idempotency_key)
    if existing is not None:
        return {"status": "skipped", "reason": "exists", "proof": existing}

    env = load_env_config(PROJECT_ROOT / ".env")
    if not env.integrations.allow_external_api_calls:
        return {"status": "disabled", "reason": "ALLOW_EXTERNAL_API_CALLS=false"}

    client = create_hedera_client(env.hedera)
    if client is None or not env.hedera.topic_id:
        return {"status": "disabled", "reason": "hedera not configured"}

    try:
        receipt = client.submit_message(env.hedera.topic_id, message)
    except Exception as exc:
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=min(60 * (self.request.retries + 1), 300))
        return {"status": "error", "message": str(exc)}

    consensus_ts = client.fetch_message_consensus_ts(
        env.hedera.topic_id, receipt.sequence_number, wait_s=6.0
    )

    import hashlib
    blake2b = hashlib.blake2b(message.encode("utf-8"), digest_size=32).hexdigest()
    try:
        store.insert_proof(
            idempotency_key=idempotency_key,
            digest_ts_unix=time.time(),
            payload_blake2b=blake2b,
            pinata_cid=cid,
            hedera_topic_id=env.hedera.topic_id,
            hedera_consensus_ts=consensus_ts,
            hedera_sequence_number=receipt.sequence_number,
        )
    except Exception as exc:
        if "UNIQUE" in str(exc) or "duplicate" in str(exc).lower():
            return {"status": "skipped", "reason": "race"}
        raise

    return {
        "status": "ok",
        "sequence_number": receipt.sequence_number,
        "consensus_timestamp": consensus_ts,
    }


# ---------------------------------------------------------------------------
# Digest chain - production hot-path (Pinata -> Hedera -> proof)
# ---------------------------------------------------------------------------


@celery_app.task(name="src.tasks.external.publish_digest_task", bind=True, max_retries=3)
def publish_digest_task(self) -> dict:
    """Best-effort independent publish to Pinata + Hedera; write whichever succeeds.

    Pinata and Hedera are published INDEPENDENTLY (each in its own try/except).
    One row is written to the `proofs` table reflecting what made it through:
    - both ok          -> cid + hedera_sequence_number + consensus_ts
    - Pinata only      -> cid (hedera_*=None)
    - Hedera only      -> hedera_sequence_number + consensus_ts (cid=None) -
      proof-of-existence via consensus_timestamp + blake2b is still preserved
    - neither          -> retry (via self.retry, no proof inserted)

    Idempotent on `idem_key = digest:<window_start>`. A repeat call in the
    same window is a no-op and returns the proof status from the DB.
    """
    env = load_env_config(PROJECT_ROOT / ".env")
    period = int(env.celery.digest_period_s)
    window_start = floor_window(time.time() - period, period)
    idem_key = f"digest:{window_start}"

    store = _store()
    existing = store.fetch_proof_by_idempotency_key(idem_key)
    if existing is not None:
        return {"status": "skipped", "reason": "already_published", "key": idem_key,
                "proof": _proof_summary(existing)}

    # 1. fetch readings for the window
    try:
        rows = store.fetch_recent_readings(minutes=int(period / 60) + 1)
    except Exception as exc:
        return _fail(str(exc), stage="fetch")
    rows = [r for r in rows if r.get("ts_unix", 0) >= window_start and r.get("ts_unix", 0) < window_start + period]
    if not rows:
        return {"status": "skipped", "reason": "no_data", "key": idem_key}

    # 2. build digest + canonical hash (computed up-front; same hash regardless of cid)
    digest = build_digest(rows, window_start)

    # 3. publish independently - Pinata
    cid: str | None = None
    pinata_status = "skipped"
    pinata_msg = ""
    provider = create_ipfs_provider()
    if provider is not None:
        try:
            pin_res = provider.pin_json(digest, name=idem_key)
            pinata_status = pin_res.status
            pinata_msg = pin_res.message
            if pin_res.status == "ok":
                cid = pin_res.metadata.get("cid")
        except Exception as exc:
            pinata_status = "error"
            pinata_msg = str(exc)
            logger.exception("pinata publish failed")
    else:
        pinata_status = "disabled"
        pinata_msg = "ipfs provider not configured"

    # 4. publish independently - Hedera (payload includes cid if we have it; else hash-only)
    payload = build_hedera_payload(cid, window_start, digest)
    final_digest, blake2b_hex = attach_cid_and_hash(digest, cid)

    hedera_seq: int | None = None
    hedera_consensus_ts: str | None = None
    hedera_status = "skipped"
    hedera_msg = ""
    if not env.integrations.allow_external_api_calls:
        hedera_status = "disabled"
        hedera_msg = "ALLOW_EXTERNAL_API_CALLS=false"
    else:
        client = create_hedera_client(env.hedera)
        if client is None or not env.hedera.topic_id:
            hedera_status = "disabled"
            hedera_msg = "hedera not configured (HEDERA_ENABLED + topic_id required)"
        else:
            try:
                receipt = client.submit_message(env.hedera.topic_id, payload)
                hedera_seq = receipt.sequence_number
                hedera_consensus_ts = client.fetch_message_consensus_ts(
                    env.hedera.topic_id, receipt.sequence_number, wait_s=6.0
                )
                hedera_status = "ok"
            except Exception as exc:
                hedera_status = "error"
                hedera_msg = str(exc)
                logger.exception("hedera publish failed")

    # 5. neither succeeded -> retry (no proof yet)
    if cid is None and hedera_seq is None:
        if self.request.retries < self.max_retries:
            raise self.retry(countdown=min(60 * (self.request.retries + 1), 300))
        return _fail(
            f"both providers failed: pinata={pinata_status}({pinata_msg}); hedera={hedera_status}({hedera_msg})",
            stage="all",
        )

    # 6. at least one succeeded -> write proof
    try:
        store.insert_proof(
            idempotency_key=idem_key,
            digest_ts_unix=window_start,
            payload_blake2b=blake2b_hex,
            pinata_cid=cid,
            hedera_topic_id=env.hedera.topic_id if hedera_seq is not None else None,
            hedera_consensus_ts=hedera_consensus_ts,
            hedera_sequence_number=hedera_seq,
        )
    except Exception as exc:
        if "UNIQUE" in str(exc) or "duplicate" in str(exc).lower():
            return {"status": "skipped", "reason": "race", "key": idem_key}
        return _fail(str(exc), stage="proof")

    # Map providers' results to overall status
    if cid is not None and hedera_seq is not None:
        overall = "ok"
    elif cid is not None:
        overall = "partial_pinata_only"
    else:
        overall = "partial_hedera_only"

    return {
        "status": overall,
        "key": idem_key,
        "cid": cid,
        "sequence_number": hedera_seq,
        "consensus_timestamp": hedera_consensus_ts,
        "blake2b": blake2b_hex,
        "pinata": {"status": pinata_status, "message": pinata_msg},
        "hedera": {"status": hedera_status, "message": hedera_msg},
    }


def _proof_summary(proof: dict) -> dict:
    """Compact representation of a proof row for return values."""
    return {
        "cid": proof.get("pinata_cid"),
        "sequence_number": proof.get("hedera_sequence_number"),
        "consensus_timestamp": proof.get("hedera_consensus_ts"),
        "blake2b": proof.get("payload_blake2b"),
    }
