"""ML artifact registry: scans models/, returns the list of available models.

Recognised artifact extensions (in priority order for tie-breaking):
- .joblib  - sklearn / xgboost / prophet pickle
- .keras   - TensorFlow / Keras 3 model
- .pkl     - legacy pickle (still loaded but discouraged)

For each model file, an adjacent ``<stem>.meta.json`` is expected.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# Required meta.json fields per CLAUDE.md. Missing fields -> warning only.
_REQUIRED_META_FIELDS = {
    "model_id", "algo", "trained_at", "features", "target",
    "metrics", "train_rows", "python_version",
}

_MODEL_EXTENSIONS = (".joblib", ".keras", ".pkl")


def _load_meta(meta_path: Path) -> dict:
    """Read meta.json; return {} on any error (with warning)."""
    try:
        with meta_path.open(encoding="utf-8") as f:
            meta = json.load(f)
    except Exception as exc:
        logger.warning("Failed to read %s: %s", meta_path, exc)
        return {}

    missing = _REQUIRED_META_FIELDS - set(meta)
    if missing:
        logger.warning("%s: missing meta fields %s", meta_path.name, sorted(missing))
    return meta


def list_available_models(
    models_dir: Path | str = "models",
) -> list[dict]:
    """Return all models in models_dir as a list[dict].

    Each entry:
        model_id     - from meta or the artifact filename stem
        model_path   - Path to the model file (.joblib / .keras / .pkl)
        joblib_path  - alias of model_path (kept for backward compat)
        meta_path    - Path to .meta.json (or None)
        meta         - contents of .meta.json (or {})
        trained_at   - meta["trained_at"] or None
        mtime        - os.path.getmtime(model_path) for fallback ordering
    """
    models_dir = Path(models_dir)
    if not models_dir.exists():
        logger.warning("models_dir not found: %s", models_dir)
        return []

    entries: list[dict] = []
    for child in sorted(models_dir.iterdir()):
        if not child.is_file() or child.suffix not in _MODEL_EXTENSIONS:
            continue
        model_path = child
        meta_path: Path | None = model_path.with_suffix(".meta.json")
        if not meta_path.exists():
            alt = model_path.parent / (model_path.stem + ".meta.json")
            meta_path = alt if alt.exists() else None

        meta = _load_meta(meta_path) if meta_path else {}

        entries.append({
            "model_id": meta.get("model_id", model_path.stem),
            "model_path": model_path,
            "joblib_path": model_path,
            "meta_path": meta_path,
            "meta": meta,
            "trained_at": meta.get("trained_at"),
            "mtime": os.path.getmtime(model_path),
        })

    return entries


def get_latest_model(
    models_dir: Path | str = "models",
) -> dict | None:
    """Return the freshest model entry, or None if none exist.

    Selection: by meta["trained_at"] (ISO8601 sorts lexicographically);
    falls back to file mtime.
    """
    entries = list_available_models(models_dir)
    if not entries:
        return None

    with_trained_at = [e for e in entries if e["trained_at"]]
    if with_trained_at:
        return max(with_trained_at, key=lambda e: e["trained_at"])

    logger.warning(
        "get_latest_model: trained_at missing on all models - using mtime"
    )
    return max(entries, key=lambda e: e["mtime"])
