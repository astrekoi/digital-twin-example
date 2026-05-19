"""Validate model artifact metadata before deployment.

Default mode validates file presence and metadata shape only. It does not load
joblib/pickle artifacts unless --allow-load is explicitly passed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.ai.artifacts import find_artifact_pairs, validate_artifact_pair


def _split_messages(messages: list[str]) -> tuple[list[str], list[str]]:
    errors = [msg for msg in messages if msg.startswith("ERROR:")]
    warnings = [msg for msg in messages if msg.startswith("WARNING:")]
    return errors, warnings


_TREE_ALGOS = {"linear", "xgboost", "lightgbm"}
_TREE_REGRESSOR_NAMES = {"Ridge", "XGBRegressor", "LGBMRegressor"}
_LSTM_REQUIRED_KEYS = {
    "state_dict",
    "architecture_config",
    "scaler_X",
    "scaler_y",
    "horizons_min",
    "framework",
}
_NBEATS_REQUIRED_KEYS = {
    "darts_state_dict",
    "architecture_config",
    "horizons_min",
    "framework",
}


def _maybe_load_model(model_path: Path, meta: dict) -> list[str]:
    """Algo-aware payload validation. Skips persistence (intentional bypass)."""
    algo = (meta.get("algo") or "").strip().lower()

    # Persistence: pickle references training-only LastValuePredictor class which
    # is unimportable on the Pi. The Pi predictor reproduces the semantics from
    # meta + the latest reading, so the joblib payload is never touched at
    # inference time. Validator must mirror that policy.
    if algo == "persistence":
        return [
            "WARNING: persistence load skipped "
            "(intentional bypass - joblib payload references training-only class)"
        ]

    if model_path.suffix == ".keras":
        return ["WARNING: --allow-load does not support .keras artifacts in this CLI"]

    try:
        import joblib

        loaded = joblib.load(model_path)
    except Exception as exc:
        return [f"ERROR: joblib.load failed for {model_path}: {exc}"]

    horizons_min = meta.get("horizons_min")

    # ----- Tree algos: dict[int, regressor] -----
    if algo in _TREE_ALGOS:
        if not isinstance(loaded, dict):
            return [
                f"ERROR: {algo} payload must be dict[int, regressor], "
                f"got {type(loaded).__name__}"
            ]
        if horizons_min:
            expected = {int(h) for h in horizons_min}
            actual = {int(k) for k in loaded.keys() if isinstance(k, (int, str))}
            if actual != expected:
                return [
                    f"ERROR: {algo} payload keys {sorted(actual)} != "
                    f"meta.horizons_min {sorted(expected)}"
                ]
        msgs: list[str] = []
        for h, regressor in loaded.items():
            cls_name = type(regressor).__name__
            if cls_name not in _TREE_REGRESSOR_NAMES:
                msgs.append(
                    f"WARNING: {algo}[{h}] regressor class {cls_name!r} not in "
                    f"expected {sorted(_TREE_REGRESSOR_NAMES)}"
                )
        return msgs

    # ----- LSTM (PyTorch): bundle dict -----
    if algo == "lstm":
        if not isinstance(loaded, dict):
            # Could be legacy Keras LSTM via .keras extension, but that's
            # caught by suffix check above; if we get here with a non-dict
            # joblib it's a malformed PyTorch artifact.
            return [f"ERROR: lstm payload must be dict, got {type(loaded).__name__}"]
        missing_keys = _LSTM_REQUIRED_KEYS - set(loaded.keys())
        if missing_keys:
            return [f"ERROR: lstm payload missing required keys: {sorted(missing_keys)}"]
        framework = (loaded.get("framework") or "").strip().lower()
        if framework != "pytorch":
            return [f"ERROR: lstm payload framework={framework!r}, expected 'pytorch'"]
        return []

    # ----- N-BEATS (Darts): bundle dict -----
    if algo == "nbeats":
        if not isinstance(loaded, dict):
            return [f"ERROR: nbeats payload must be dict, got {type(loaded).__name__}"]
        missing_keys = _NBEATS_REQUIRED_KEYS - set(loaded.keys())
        if missing_keys:
            return [f"ERROR: nbeats payload missing required keys: {sorted(missing_keys)}"]
        framework = (loaded.get("framework") or "").strip().lower()
        if framework != "darts":
            return [f"ERROR: nbeats payload framework={framework!r}, expected 'darts'"]
        return []

    # ----- Prophet: single Prophet model -----
    if algo == "prophet":
        if not hasattr(loaded, "make_future_dataframe"):
            return [
                f"ERROR: prophet payload {type(loaded).__name__} lacks "
                "make_future_dataframe method"
            ]
        return []

    # ----- Unknown / legacy algo: just confirm joblib.load succeeded -----
    return []


def _read_meta(meta_path: Path) -> dict:
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open(encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def validate_pairs(
    pairs: list[tuple[Path, Path]],
    *,
    allow_load: bool = False,
) -> list[dict]:
    results: list[dict] = []
    for model_path, meta_path in pairs:
        messages = validate_artifact_pair(model_path, meta_path)
        if (
            allow_load
            and model_path.exists()
            and not any(msg.startswith("ERROR:") for msg in messages)
        ):
            meta = _read_meta(meta_path)
            messages.extend(_maybe_load_model(model_path, meta))
        errors, warnings = _split_messages(messages)
        results.append(
            {
                "model": str(model_path),
                "meta": str(meta_path),
                "status": "error" if errors else "ok",
                "errors": errors,
                "warnings": warnings,
            }
        )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate model artifact metadata")
    parser.add_argument(
        "--models-dir", default="models", help="Directory to scan when --model/--meta are absent"
    )
    parser.add_argument("--model", default=None, help="Specific model file path")
    parser.add_argument("--meta", default=None, help="Specific .meta.json file path")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    parser.add_argument("--strict", action="store_true", help="Treat warnings as non-zero exit")
    parser.add_argument(
        "--allow-load",
        action="store_true",
        help="Explicitly try joblib.load for supported artifacts (default: false)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if bool(args.model) != bool(args.meta):
        parser.error("--model and --meta must be provided together")

    if args.model and args.meta:
        pairs = [(Path(args.model), Path(args.meta))]
    else:
        pairs = find_artifact_pairs(args.models_dir)

    results = validate_pairs(pairs, allow_load=args.allow_load)
    if not pairs:
        payload = {"status": "empty", "models_dir": args.models_dir, "results": []}
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"No model artifacts found in {args.models_dir}")
        return 0

    has_errors = any(item["errors"] for item in results)
    has_warnings = any(item["warnings"] for item in results)
    exit_code = 1 if has_errors or (args.strict and has_warnings) else 0

    if args.json:
        print(json.dumps({"status": "error" if exit_code else "ok", "results": results}, indent=2))
    else:
        for item in results:
            print(f"{item['model']} -> {item['status']}")
            for msg in [*item["errors"], *item["warnings"]]:
                print(f"  {msg}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
