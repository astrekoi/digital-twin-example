"""Build a compact telemetry digest for proof-of-existence on Hedera.

Layout (<=1024 byte payload limit per HCS message):
  {
    "v": 1,
    "ts": <window_start_unix>,
    "cid": "<pinata_cid>",
    "n": <reading_count>,
    "stats": {
      "temp_c": [min, mean, max],
      "humidity_pct": [...],
      "pressure_hpa": [...],
      "lux": [...],
      "current_ma": [...]
    },
    "blake2b": "<hex digest of the full window JSON>"
  }
"""
from __future__ import annotations

import hashlib
import json
from typing import Any


def _stats_triplet(values: list[float]) -> list[float | None]:
    if not values:
        return [None, None, None]
    n = len(values)
    return [
        round(min(values), 3),
        round(sum(values) / n, 3),
        round(max(values), 3),
    ]


_STAT_FIELDS = ("temp_c", "humidity_pct", "pressure_hpa", "lux", "current_ma")


def build_digest(rows: list[dict], window_start_unix: float) -> dict[str, Any]:
    """Aggregate readings rows into a compact digest dict."""
    by_field: dict[str, list[float]] = {f: [] for f in _STAT_FIELDS}
    for row in rows:
        for f in _STAT_FIELDS:
            v = row.get(f)
            if v is not None:
                try:
                    by_field[f].append(float(v))
                except (TypeError, ValueError):
                    continue
    stats = {f: _stats_triplet(by_field[f]) for f in _STAT_FIELDS}
    return {
        "v": 1,
        "ts": int(window_start_unix),
        "n": len(rows),
        "stats": stats,
    }


def attach_cid_and_hash(digest: dict[str, Any], cid: str | None) -> tuple[dict[str, Any], str]:
    """Add cid (if available) and content hash to digest; return (final dict, hex hash).

    cid=None - Hedera-only proof (Pinata down or disabled). The hash is still
    computed over the canonical digest form, giving a valid proof-of-existence
    via the Hedera consensus_timestamp.
    """
    out = dict(digest)
    if cid:
        out["cid"] = cid
    canonical = json.dumps(out, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h = hashlib.blake2b(canonical, digest_size=32).hexdigest()
    out["blake2b"] = h
    return out, h


def build_hedera_payload(cid: str | None, window_start_unix: float, digest: dict[str, Any]) -> str:
    """Build the compact UTF-8 string we submit to HCS as the message body.

    cid=None is allowed - payload then contains digest+blake2b only, without
    an IPFS anchor.
    """
    final, _ = attach_cid_and_hash(digest, cid)
    return json.dumps(final, sort_keys=True, separators=(",", ":"))


def floor_window(ts_unix: float, period_s: int) -> int:
    """Round ts_unix down to the start of the current period_s-sized window."""
    return int(ts_unix - (ts_unix % period_s))
