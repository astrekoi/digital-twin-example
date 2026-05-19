"""Pure-logic tests for hedera digest helpers (no network, no SDK)."""
from __future__ import annotations

import json

from src.integrations.hedera.digest import (
    attach_cid_and_hash,
    build_digest,
    build_hedera_payload,
    floor_window,
)


def _row(ts, **kw):
    base = {"ts_unix": ts, "ts_iso": "x"}
    base.update(kw)
    return base


def test_build_digest_aggregates_minmaxavg():
    rows = [
        _row(100, temp_c=20.0, humidity_pct=50.0),
        _row(101, temp_c=22.0, humidity_pct=55.0),
        _row(102, temp_c=21.0, humidity_pct=None),  # None should be ignored
    ]
    d = build_digest(rows, window_start_unix=100)
    assert d["v"] == 1
    assert d["ts"] == 100
    assert d["n"] == 3
    assert d["stats"]["temp_c"] == [20.0, 21.0, 22.0]
    assert d["stats"]["humidity_pct"] == [50.0, 52.5, 55.0]
    assert d["stats"]["pressure_hpa"] == [None, None, None]


def test_attach_cid_and_hash_is_deterministic():
    d = build_digest([_row(100, temp_c=20.0)], window_start_unix=100)
    out1, h1 = attach_cid_and_hash(d, "QmABC")
    out2, h2 = attach_cid_and_hash(d, "QmABC")
    assert h1 == h2
    assert out1["cid"] == "QmABC"
    assert out1["blake2b"] == h1
    assert len(h1) == 64  # blake2b-256 -> 32 bytes -> 64 hex chars


def test_build_hedera_payload_under_limit():
    d = build_digest([_row(100 + i, temp_c=20 + i * 0.1) for i in range(60)], 100)
    payload = build_hedera_payload("QmTopicCID", 100, d)
    decoded = json.loads(payload)
    assert decoded["cid"] == "QmTopicCID"
    assert "blake2b" in decoded
    # HCS message limit is 1024 bytes
    assert len(payload.encode("utf-8")) < 1024


def test_floor_window():
    # 1700000123 falls inside the [1700000100, 1700000400) bucket of size 300
    # because 1700000100 % 300 == 0.
    assert floor_window(1700000100.0, 300) == 1700000100
    assert floor_window(1700000123.5, 300) == 1700000100
    assert floor_window(1700000399.99, 300) == 1700000100
    assert floor_window(1700000400.0, 300) == 1700000400
