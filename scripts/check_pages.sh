#!/usr/bin/env bash
# Multi-stage Taipy page health check via curl.
#
# Stage 1: GET /<page>          - SPA shell (~2.3KB, real content over WebSocket)
# Stage 2: GET /taipy-jsx/<page> - real JSX render (500 = template/state error)
# Stage 3: freshness of last rows in SQLite (readings / predictions / proofs)
#
# `set -e` deliberately omitted: a failing curl should be reported, not abort
# the whole script.
#
# Usage: bash scripts/check_pages.sh

set -u

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5000}"
BASE="http://$HOST:$PORT"
PAGES=(overview charts control forecast events settings)

# Resolve a real client_id via the socket.io handshake.
CLIENT_ID=$(curl -s "$BASE/socket.io/?EIO=4&transport=polling&t=$(date +%s)" 2>/dev/null \
            | python3 -c "import sys,json,re; t=sys.stdin.read(); m=re.search(r'\"sid\":\"([^\"]+)\"', t); print(m.group(1) if m else '')")
[ -z "$CLIENT_ID" ] && CLIENT_ID="cli-smoke-$(date +%s)"

echo "=== STAGE 1: GET /<page> (SPA shell) ==="
for p in "${PAGES[@]}"; do
    code=$(curl -s -o /dev/null -w "%{http_code}" "$BASE/$p")
    size=$(curl -s "$BASE/$p" | wc -c)
    printf "  /%-10s -> HTTP %s  (%s bytes)\n" "$p" "$code" "$size"
done

echo ""
echo "=== STAGE 2: GET /taipy-jsx/<page> (real render) ==="
for p in "${PAGES[@]}"; do
    code=$(curl -s -o /tmp/_jsx_$p -w "%{http_code}" \
           "$BASE/taipy-jsx/$p?client_id=$CLIENT_ID&v=4.1.2")
    size=$(wc -c < /tmp/_jsx_$p)
    if [ "$code" = "200" ]; then
        printf "  /%-10s -> OK   %s  (%s bytes JSX)\n" "$p" "$code" "$size"
    else
        printf "  /%-10s -> FAIL %s  body: %s\n" "$p" "$code" "$(head -c 200 /tmp/_jsx_$p)"
    fi
    rm -f /tmp/_jsx_$p
done

echo ""
echo "=== STAGE 3: DB freshness ==="
DB="${DB:-/home/astropi/new_proj/data/telemetry.db}"
if [ -f "$DB" ]; then
    echo "  readings:"
    sqlite3 "$DB" "SELECT '    last=' || ts_iso || '  count=' || (SELECT count(*) FROM readings) FROM readings ORDER BY ts_unix DESC LIMIT 1"
    echo "  predictions:"
    sqlite3 "$DB" "SELECT '    last=' || datetime(generated_at,'unixepoch','+3 hours') || '  model=' || model_id || '  forecast=' || forecast_json FROM predictions ORDER BY generated_at DESC LIMIT 1"
    echo "  proofs:"
    sqlite3 "$DB" "SELECT '    last=' || idempotency_key || '  cid=' || coalesce(pinata_cid,'NULL') || '  hedera_seq=' || coalesce(hedera_sequence_number,'NULL') FROM proofs ORDER BY digest_ts_unix DESC LIMIT 1"
fi

echo ""
echo "=== Done ==="
