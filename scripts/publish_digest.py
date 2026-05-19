"""Publish telemetry digest to IPFS (Pinata) and/or IOTA.

Reads the last --limit readings from SQLite (read-only), builds a digest with
src.integrations.digest, and publishes via the configured target. External
calls require ALLOW_EXTERNAL_API_CALLS=true plus provider-specific config.

Exit 1 if any publish call returns status="error"; otherwise 0.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.integrations.digest import build_telemetry_digest  # noqa: E402
from src.integrations.factory import create_iota_client, create_pinata_client  # noqa: E402


def _fetch_readings(db_path: str, limit: int) -> list[dict]:
    """Fetch last *limit* readings using read-only SQLite connection."""
    uri = f"file:{db_path}?mode=ro"
    try:
        conn = sqlite3.connect(uri, uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM readings ORDER BY ts_unix DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        # Return in chronological order
        return [dict(r) for r in reversed(rows)]
    except sqlite3.OperationalError as exc:
        print(f"[ERROR] Cannot open DB '{db_path}': {exc}", file=sys.stderr)
        sys.exit(1)


def _publish_pinata(digest: dict, dry_run: bool) -> bool:
    """Publish to Pinata. Returns True on success/disabled/dry-run."""
    client = create_pinata_client()
    if dry_run:
        return True  # payload already printed by caller
    if not client.is_configured():
        print("[INFO] Pinata: not configured (IPFS_ENABLED/IPFS_PROVIDER/PINATA_JWT missing).")
        return True
    if not client.allow_external_api_calls:
        print("[INFO] Pinata: external calls blocked (ALLOW_EXTERNAL_API_CALLS=false).")
        return True
    result = client.pin_json(digest, name="plc-twin-telemetry-digest")
    if result.status == "ok":
        print(
            f"[OK] Pinata: CID={result.metadata.get('cid')}  "
            f"size={result.metadata.get('size')}  "
            f"ts={result.metadata.get('timestamp')}"
        )
        return True
    print(f"[ERROR] Pinata: {result.message}")
    return False


def _publish_iota(digest: dict, dry_run: bool) -> bool:
    """Publish to IOTA Stardust as tagged data block. Returns True on
    success/disabled/dry-run."""
    client = create_iota_client()
    if dry_run:
        return True  # payload already printed by caller
    if not client.is_configured():
        print("[INFO] IOTA: not configured (IOTA_ENABLED/IOTA_NODE_URL missing).")
        return True
    if not client.allow_external_api_calls:
        print("[INFO] IOTA: external calls blocked (ALLOW_EXTERNAL_API_CALLS=false).")
        return True
    result = client.publish_telemetry_digest(digest)
    if result.status == "ok":
        print(
            f"[OK] IOTA: block_id={result.metadata.get('block_id')}  "
            f"explorer={result.metadata.get('explorer_url')}  "
            f"ts={result.metadata.get('timestamp')}"
        )
        return True
    if result.status == "disabled":
        print(f"[INFO] IOTA: {result.message}")
        return True
    print(f"[ERROR] IOTA: {result.message}")
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--db",
        default="data/telemetry.db",
        help="SQLite database path (default: data/telemetry.db)",
    )
    parser.add_argument(
        "--limit", type=int, default=100, help="Number of recent readings to include (default: 100)"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print digest payload without publishing"
    )
    parser.add_argument(
        "--target",
        choices=["pinata", "iota", "both"],
        default="pinata",
        help="Publish target (default: pinata)",
    )
    args = parser.parse_args()

    readings = _fetch_readings(args.db, args.limit)
    digest = build_telemetry_digest(readings, source="sqlite")

    if args.dry_run:
        print("[DRY-RUN] Digest payload (no network call):")
        print(json.dumps(digest, indent=2, ensure_ascii=False))
        print(f"[DRY-RUN] rows={digest['row_count']}  target={args.target}")
        sys.exit(0)

    errors = False
    if args.target in ("pinata", "both"):
        if not _publish_pinata(digest, args.dry_run):
            errors = True
    if args.target in ("iota", "both"):
        if not _publish_iota(digest, args.dry_run):
            errors = True

    sys.exit(1 if errors else 0)


if __name__ == "__main__":
    main()
