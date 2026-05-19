#!/usr/bin/env bash
# Idempotent troykahat patch for Pi 5.
# Replaces the wiringpi-based gpio_expander.py with the smbus2 version.
# Safe to run repeatedly: the backup is created only once.
#
# Usage:
#   bash scripts/patch_troykahat_for_pi5.sh
#   VENV=/custom/path bash scripts/patch_troykahat_for_pi5.sh

set -euo pipefail

VENV="${VENV:-/home/astropi/.venv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PATCH_SRC="$PROJECT_ROOT/src/patches/gpio_expander.py"

# 1. Verify venv (must exist before we resolve site-packages).
if [[ ! -x "$VENV/bin/python" ]]; then
    echo "ERROR: venv not found: $VENV" >&2
    echo "  Create it: python3 -m venv ~/.venv --system-site-packages" >&2
    exit 1
fi

# Resolve site-packages from the venv's own Python - no hardcoded version.
# Works for Python 3.11 / 3.13 / any future version.
SITE_PACKAGES="$("$VENV/bin/python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
SITE_PKG="$SITE_PACKAGES/troykahat"
TARGET="$SITE_PKG/gpio_expander.py"
BACKUP="$SITE_PKG/gpio_expander.py.bak"

echo "=== patch_troykahat_for_pi5.sh ==="
echo "  venv:       $VENV"
echo "  site-pkgs:  $SITE_PACKAGES"
echo "  patch src:  $PATCH_SRC"
echo "  target:     $TARGET"

# 2. Verify troykahat is installed.
if [[ ! -d "$SITE_PKG" ]]; then
    echo "ERROR: troykahat not found: $SITE_PKG" >&2
    echo "  Install it: pip install troykahat" >&2
    exit 1
fi

if [[ ! -f "$TARGET" ]]; then
    echo "ERROR: gpio_expander.py not found: $TARGET" >&2
    exit 1
fi

# 3. Verify smbus2.
if ! "$VENV/bin/python" -c "import smbus2" 2>/dev/null; then
    echo "ERROR: smbus2 not found in venv." >&2
    echo "  Install manually: pip install smbus2" >&2
    exit 1
fi
echo "smbus2 available"

# 4. Verify the patch source file.
if [[ ! -f "$PATCH_SRC" ]]; then
    echo "ERROR: patch source not found: $PATCH_SRC" >&2
    exit 1
fi

# 5. Backup - only once (idempotency).
if [[ ! -f "$BACKUP" ]]; then
    cp "$TARGET" "$BACKUP"
    echo "Backup saved: $BACKUP"
else
    echo "Backup already exists: $BACKUP (skipping)"
fi

# 6. Check whether the patch is already applied (look for the smbus2 marker).
if grep -q "from smbus2 import" "$TARGET" 2>/dev/null; then
    echo "Patch already applied (smbus2 found in target)"
else
    cp "$PATCH_SRC" "$TARGET"
    echo "Patch applied: $TARGET"
fi

# 7. Class import check (no hardware touched - GpioExpander is not instantiated).
echo "Importing GpioExpander..."
if "$VENV/bin/python" -c "
from troykahat.gpio_expander import GpioExpander
public = [m for m in dir(GpioExpander) if not m.startswith('_')]
expected = {'pinMode', 'analogRead', 'analogWrite', 'digitalRead', 'digitalWrite'}
missing = expected - set(public)
if missing:
    raise AssertionError(f'Missing methods: {missing}')
print('Public API OK:', sorted(expected))
"; then
    echo "=== Patch applied and verified ==="
else
    echo "ERROR: import check failed." >&2
    echo "  Restore the original: cp '$BACKUP' '$TARGET'" >&2
    exit 1
fi
