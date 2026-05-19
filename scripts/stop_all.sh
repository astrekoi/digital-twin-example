#!/usr/bin/env bash
# Gracefully stop the PLC-twin stack started by scripts/start_all.sh.
# 1. SIGTERM each PID from data/pids/*.pid (and its process group for Taipy).
# 2. Wait up to 10s for everything to exit.
# 3. Safety net: pkill -TERM celery/app.web by name in case any leaked.
# 4. SIGKILL stragglers.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIDS_DIR="$PROJECT_ROOT/data/pids"

if [[ ! -d "$PIDS_DIR" ]]; then
    PIDS_DIR=""
fi

# Phase 1: SIGTERM by pidfile (group-kill for taipy.pid via -PGID)
if [[ -n "$PIDS_DIR" ]]; then
    for pid_file in "$PIDS_DIR"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        name="$(basename "$pid_file" .pid)"
        pid="$(cat "$pid_file" 2>/dev/null || true)"
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "Stopping $name (PID $pid)…"
            if [[ "$name" == "taipy" ]]; then
                # Taipy was started via setsid -> its PID == PGID; -SIGNAL kills the whole group.
                kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
            else
                kill -TERM "$pid" 2>/dev/null || true
            fi
        fi
    done
fi

# Phase 2: wait up to 10s
for _ in {1..20}; do
    any_alive=0
    if [[ -n "$PIDS_DIR" ]]; then
        for pid_file in "$PIDS_DIR"/*.pid; do
            [[ -f "$pid_file" ]] || continue
            pid="$(cat "$pid_file" 2>/dev/null || true)"
            [[ -n "$pid" ]] || continue
            if kill -0 "$pid" 2>/dev/null; then
                any_alive=1
                break
            fi
        done
    fi
    [[ $any_alive -eq 0 ]] && break
    sleep 0.5
done

# Phase 3: safety net - pkill anything Celery/Taipy of ours that we missed
pkill -TERM -f "src\\.tasks\\.celery_app" 2>/dev/null || true
pkill -TERM -f "app\\.web" 2>/dev/null || true
sleep 1

# Phase 4: SIGKILL stragglers
if [[ -n "$PIDS_DIR" ]]; then
    for pid_file in "$PIDS_DIR"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        name="$(basename "$pid_file" .pid)"
        pid="$(cat "$pid_file" 2>/dev/null || true)"
        [[ -n "$pid" ]] || continue
        if kill -0 "$pid" 2>/dev/null; then
            echo "Force-killing $name (PID $pid)…"
            if [[ "$name" == "taipy" ]]; then
                kill -KILL -"$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
            else
                kill -KILL "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$pid_file"
    done
fi

# Final cleanup of leaked processes by pattern (non-fatal: pkill returns 1
# when nothing matched, which is exactly the desired healthy state).
pkill -KILL -f "src\\.tasks\\.celery_app" 2>/dev/null || true
pkill -KILL -f "app\\.web" 2>/dev/null || true

echo "All stopped."
exit 0
