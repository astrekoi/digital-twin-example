"""Health-check for hardware and AI runtime dependencies.

Verifies that required packages are importable and that the troykahat Pi 5
patch is in place. Does NOT initialise GPIO, I²C, or any hardware.

Exit 1 on any FAIL; 0 otherwise (OK + WARN count as success).
"""

from __future__ import annotations

import re
import sys
import sysconfig
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


_results: list[tuple[str, str, str]] = []  # (status, label, detail)


def _ok(label: str, detail: str = "") -> None:
    _results.append(("OK  ", label, detail))


def _warn(label: str, detail: str) -> None:
    _results.append(("WARN", label, detail))


def _fail(label: str, detail: str) -> None:
    _results.append(("FAIL", label, detail))


def _try_import(module: str, label: str, *, warn_on_fail: bool = False) -> bool:
    """Try to import *module*. Returns True if successful."""
    try:
        __import__(module)
        _ok(label, f"import {module}")
        return True
    except ImportError as exc:
        msg = str(exc)
        if warn_on_fail:
            _warn(label, f"import {module}: {msg}")
        else:
            _fail(label, f"import {module}: {msg}")
        return False
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if warn_on_fail:
            _warn(label, f"import {module}: {msg}")
        else:
            _fail(label, f"import {module}: {msg}")
        return False


# 1. Core project packages
_try_import("yaml", "PyYAML")
_try_import("pydantic", "pydantic")
_try_import("pandas", "pandas")
_try_import("numpy", "numpy")
_try_import("joblib", "joblib")
_try_import("sklearn", "scikit-learn")
_try_import("xgboost", "xgboost")
_try_import("prophet", "prophet")
_try_import("tensorflow", "tensorflow", warn_on_fail=True)

# 2. Hardware deps (WARN on fail - hardware may be physically absent in CI)
_try_import("gpiozero", "gpiozero", warn_on_fail=True)
_try_import("lgpio", "lgpio (rpi-lgpio)", warn_on_fail=True)
_try_import("smbus2", "smbus2", warn_on_fail=True)
_try_import("troykahat", "troykahat", warn_on_fail=True)
_try_import("adafruit_bme280", "adafruit-circuitpython-bme280", warn_on_fail=True)
_try_import("adafruit_bh1750", "adafruit-circuitpython-bh1750", warn_on_fail=True)
_try_import("adafruit_ina219", "adafruit-circuitpython-ina219", warn_on_fail=True)
_try_import("adafruit_bmp280", "adafruit-circuitpython-bmp280", warn_on_fail=True)
_try_import("luma.oled", "luma.oled", warn_on_fail=True)
_try_import("serial", "pyserial", warn_on_fail=True)

# 3. Dashboard / AI / integrations
_try_import("reflex", "reflex")
_try_import("src.db", "src.db")
_try_import("src.config", "src.config")
_try_import("src.ai.features", "src.ai.features")
_try_import("src.ai.loader", "src.ai.loader")
_try_import("src.ai.predictor", "src.ai.predictor")
_try_import("src.integrations.pinata", "src.integrations.pinata")
_try_import("src.integrations.iota", "src.integrations.iota")

# plotly: WARN - needed for charts but not for core operation
_try_import("plotly", "plotly (charts)", warn_on_fail=True)

# 4. troykahat Pi 5 patch verification
_SITE_PACKAGES = Path(sysconfig.get_paths()["purelib"])
_VENV_GPIO_EXP = _SITE_PACKAGES / "troykahat" / "gpio_expander.py"
_REPO_GPIO_EXP = _PROJECT_ROOT / "src/patches/gpio_expander.py"

_ACTIVE_IMPORT_RE = re.compile(
    r"^\s*(?:import|from)\s+wiringpi",
    re.MULTILINE,
)


def _check_patch(path: Path, label: str) -> None:
    if not path.exists():
        _fail(label, f"file not found: {path}")
        return
    content = path.read_text(encoding="utf-8", errors="replace")
    issues: list[str] = []
    if "smbus2" not in content:
        issues.append("smbus2 not present - patch may be missing")
    if _ACTIVE_IMPORT_RE.search(content):
        issues.append("active wiringpi import found - patch may be lost")
    if issues:
        _fail(label, "; ".join(issues))
    else:
        _ok(label, "smbus2 present, no active wiringpi import")


_check_patch(_VENV_GPIO_EXP, "troykahat patch (venv)")
_check_patch(_REPO_GPIO_EXP, "troykahat patch (repo copy)")

# Check backup exists
_BAK = _SITE_PACKAGES / "troykahat" / "gpio_expander.py.bak"
if _BAK.exists():
    _ok("troykahat backup (.bak)", str(_BAK))
else:
    _warn("troykahat backup (.bak)", f"not found at {_BAK}; run patch_troykahat_for_pi5.sh")

# 5. Print results
fails = [r for r in _results if r[0] == "FAIL"]
warns = [r for r in _results if r[0] == "WARN"]
oks = [r for r in _results if r[0] == "OK  "]

print(f"\n=== assert_runtime_imports - {len(_results)} checks ===\n")

for status, label, detail in _results:
    suffix = f"  ({detail})" if detail else ""
    print(f"  [{status}] {label}{suffix}")

print()
print(f"  OK: {len(oks)}   WARN: {len(warns)}   FAIL: {len(fails)}")

if fails:
    print("\nFAIL items:")
    for _, label, detail in fails:
        print(f"  • {label}: {detail}")
    print()
    sys.exit(1)
else:
    print("\nAll checks passed (OK + WARN).")
    sys.exit(0)
