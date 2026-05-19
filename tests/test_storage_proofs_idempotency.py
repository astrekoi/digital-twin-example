"""SQLite proofs table: UNIQUE on idempotency_key + fetch_proof_by_idempotency_key."""
from __future__ import annotations

import pytest


def test_insert_proof_creates_row(sqlite_store) -> None:
    pid = sqlite_store.insert_proof(
        idempotency_key="digest:1700000000",
        digest_ts_unix=1700000000.0,
        payload_blake2b="abcd",
        pinata_cid="QmABC",
    )
    assert pid > 0
    found = sqlite_store.fetch_proof_by_idempotency_key("digest:1700000000")
    assert found is not None
    assert found["pinata_cid"] == "QmABC"
    assert found["digest_ts_unix"] == 1700000000.0


def test_duplicate_idempotency_key_raises(sqlite_store) -> None:
    sqlite_store.insert_proof(
        idempotency_key="digest:1700000001",
        digest_ts_unix=1700000001.0,
        payload_blake2b="abcd",
    )
    with pytest.raises(Exception) as exc:
        sqlite_store.insert_proof(
            idempotency_key="digest:1700000001",
            digest_ts_unix=1700000001.0,
            payload_blake2b="efgh",
        )
    assert "UNIQUE" in str(exc.value) or "unique" in str(exc.value).lower()


def test_fetch_proofs_orders_desc(sqlite_store) -> None:
    for i, ts in enumerate([1700000000.0, 1700000060.0, 1700000120.0]):
        sqlite_store.insert_proof(
            idempotency_key=f"digest:{int(ts)}",
            digest_ts_unix=ts,
            payload_blake2b=f"hash{i}",
        )
    rows = sqlite_store.fetch_proofs(limit=5)
    assert len(rows) == 3
    # newest first
    assert rows[0]["digest_ts_unix"] == 1700000120.0
    assert rows[-1]["digest_ts_unix"] == 1700000000.0


def test_fetch_proofs_window(sqlite_store) -> None:
    for ts in [1700000000.0, 1700000060.0, 1700000120.0]:
        sqlite_store.insert_proof(
            idempotency_key=f"digest:{int(ts)}",
            digest_ts_unix=ts,
            payload_blake2b="x",
        )
    rows = sqlite_store.fetch_proofs(since_ts_unix=1700000050.0)
    assert len(rows) == 2
