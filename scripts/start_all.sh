#!/usr/bin/env bash
# Start the full PLC-twin stack: Redis (system), 3 Celery workers, Celery beat, Taipy.
# All workers run as background processes; PIDs in data/pids/, logs in data/logs/.
# Use scripts/stop_all.sh for clean shutdown.
#
# Prerequisites:
#   - sudo apt install redis-server     (one-time)
#   - source ~/.venv-plc-twin/bin/activate

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

VENV_PY="${VENV_PY:-/home/astropi/.venv-plc-twin/bin/python}"
CELERY="${CELERY:-/home/astropi/.venv-plc-twin/bin/celery}"
PIDS_DIR="$PROJECT_ROOT/data/pids"
LOGS_DIR="$PROJECT_ROOT/data/logs"
mkdir -p "$PIDS_DIR" "$LOGS_DIR"

# 0. Pre-flight: nothing of ours already running, port 5000 free.
if pgrep -af "src.tasks.celery_app" >/dev/null 2>&1; then
    echo "ERROR: another celery -A src.tasks.celery_app process is running. Run scripts/stop_all.sh first."
    pgrep -af "src.tasks.celery_app" | head -5
    exit 1
fi
if pgrep -af "app.web" >/dev/null 2>&1; then
    echo "ERROR: another app.web (Taipy) process is running. Run scripts/stop_all.sh first."
    pgrep -af "app.web" | head -3
    exit 1
fi
if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE ':5000$'; then
    echo "ERROR: port 5000 already bound. Free it before starting Taipy."
    exit 1
fi

# 1. Redis
if ! command -v redis-cli >/dev/null 2>&1; then
    echo "ERROR: redis-cli not found. Install via: sudo apt install redis-server"
    exit 1
fi
if ! redis-cli ping >/dev/null 2>&1; then
    echo "Starting redis-server (system service)…"
    sudo systemctl start redis-server
    sleep 1
fi
redis-cli ping >/dev/null || { echo "redis ping failed"; exit 1; }
echo "Redis: OK"

# Helper: celery --detach forks asynchronously; pidfile may not exist for ~1 sec.
# Wait up to 5 sec then echo "<name> started (PID <pid>)" - fail if absent.
wait_pid_and_echo() {
    local name="$1"
    local pid_file="$PIDS_DIR/$name.pid"
    for _ in {1..50}; do
        if [[ -s "$pid_file" ]]; then
            echo "$name started (PID $(cat "$pid_file"))"
            return 0
        fi
        sleep 0.1
    done
    echo "ERROR: $name pidfile $pid_file did not appear within 5s"
    return 1
}

# 2. Celery hw-worker  (--pool=solo, queues hardware_priority + hardware)
# Clean stale pidfiles from previous run (if stop_all left any).
rm -f "$PIDS_DIR"/*.pid

"$CELERY" -A src.tasks.celery_app worker \
    -Q hardware_priority,hardware \
    --pool=solo \
    --hostname=hw@%h \
    --loglevel=info \
    --detach \
    --pidfile="$PIDS_DIR/hw.pid" \
    --logfile="$LOGS_DIR/hw.log"
wait_pid_and_echo hw

# 3. Celery ai-worker  (--pool=prefork --concurrency=1 --max-tasks-per-child=100)
"$CELERY" -A src.tasks.celery_app worker \
    -Q ai \
    --pool=prefork --concurrency=1 --max-tasks-per-child=100 \
    --hostname=ai@%h \
    --loglevel=info \
    --detach \
    --pidfile="$PIDS_DIR/ai.pid" \
    --logfile="$LOGS_DIR/ai.log"
wait_pid_and_echo ai

# 4. Celery ext-worker  (--pool=threads --concurrency=2)
"$CELERY" -A src.tasks.celery_app worker \
    -Q external \
    --pool=threads --concurrency=2 \
    --hostname=ext@%h \
    --loglevel=info \
    --detach \
    --pidfile="$PIDS_DIR/ext.pid" \
    --logfile="$LOGS_DIR/ext.log"
wait_pid_and_echo ext

# 5. Celery beat
"$CELERY" -A src.tasks.celery_app beat \
    --schedule=data/celerybeat-schedule.db \
    --loglevel=info \
    --detach \
    --pidfile="$PIDS_DIR/beat.pid" \
    --logfile="$LOGS_DIR/beat.log"
wait_pid_and_echo beat

# 6. Taipy GUI - capture the child Python PID via setsid so we can SIGTERM
# the whole process group later. Without setsid, gevent's worker forks orphan.
setsid nohup "$VENV_PY" -m app.web \
    > "$LOGS_DIR/taipy.log" 2>&1 < /dev/null &
TAIPY_PID=$!
echo $TAIPY_PID > "$PIDS_DIR/taipy.pid"
echo "taipy started (PGID $TAIPY_PID)"

# Brief health check
sleep 4
echo
echo "--- HEALTH ---"
if "$CELERY" -A src.tasks.celery_app inspect ping --timeout=5 2>&1 | tail -8; then
    :
fi

if curl -s -o /dev/null -w "Taipy /overview HTTP %{http_code}\n" http://127.0.0.1:5000/overview 2>&1; then
    :
fi

echo
echo "All processes started. Logs in $LOGS_DIR/, PIDs in $PIDS_DIR/."
echo "Stop with: bash scripts/stop_all.sh"
