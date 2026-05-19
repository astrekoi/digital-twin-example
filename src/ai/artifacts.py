"""Safe model artifact metadata validation.

This module validates file presence and metadata shape only. It deliberately
does not call joblib.load by default because model deserialization is a runtime
operation for trusted artifacts, not a safe readiness check.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "model_id",
    "algo",
    "trained_at",
    "features",
    "target",
    "metrics",
    "train_rows",
}

ALLOWED_MODEL_EXTENSIONS = {".joblib", ".pkl", ".keras"}

# Hard caps on horizons_min list shape. 12 horizons covers typical
# forecasting timelines (1m..1h..24h) without exploding inference cost on
# Pi 5; max horizon 1440min = 24h is far beyond any realistic short-term
# forecasting need on this stand.
_HORIZONS_MAX_LEN = 12
_HORIZON_MIN_VALUE = 1
_HORIZON_MAX_VALUE = 1440


@dataclass(frozen=True)
class ModelArtifactMetadata:
    model_id: str
    algo: str
    trained_at: str
    features: list[str]
    target: str
    metrics: dict[str, float]
    train_rows: int
    python_version: str | None = None
    sklearn_version: str | None = None
    created_by: str | None = None
    notes: str | None = None
    horizon_min: int | None = None
    source_data: str | None = None
    feature_pipeline_version: str | None = None
    model_file: str | None = None
    # Multi-horizon + sequence-model fields (added 2026-05-01 for the 7-algo
    # trainer family). All optional - legacy single-horizon artifacts are still
    # accepted by the validator.
    horizons_min: list[int] | None = None
    feature_columns: list[str] | None = None
    artifact_type: str | None = None  # "torch" | "darts_torch"
    framework: str | None = None  # "pytorch" | "darts" - declared INSIDE joblib payload
    xgboost_version: str | None = None
    lightgbm_version: str | None = None
    torch_version: str | None = None
    prophet_version: str | None = None
    darts_version: str | None = None
    pandas_version: str | None = None
    numpy_version: str | None = None


def _coerce_metadata(data: dict[str, Any]) -> ModelArtifactMetadata:
    horizons_min_val = data.get("horizons_min")
    feature_columns_val = data.get("feature_columns")
    return ModelArtifactMetadata(
        model_id=str(data.get("model_id", "")),
        algo=str(data.get("algo", "")),
        trained_at=str(data.get("trained_at", "")),
        features=list(data.get("features") or []),
        target=str(data.get("target", "")),
        metrics=dict(data.get("metrics") or {}),
        train_rows=int(data.get("train_rows") or 0),
        python_version=data.get("python_version"),
        sklearn_version=data.get("sklearn_version"),
        created_by=data.get("created_by"),
        notes=data.get("notes"),
        horizon_min=data.get("horizon_min"),
        source_data=data.get("source_data"),
        feature_pipeline_version=data.get("feature_pipeline_version"),
        model_file=data.get("model_file"),
        horizons_min=list(horizons_min_val) if isinstance(horizons_min_val, list) else None,
        feature_columns=(
            list(feature_columns_val) if isinstance(feature_columns_val, list) else None
        ),
        artifact_type=data.get("artifact_type"),
        framework=data.get("framework"),
        xgboost_version=data.get("xgboost_version"),
        lightgbm_version=data.get("lightgbm_version"),
        torch_version=data.get("torch_version"),
        prophet_version=data.get("prophet_version"),
        darts_version=data.get("darts_version"),
        pandas_version=data.get("pandas_version"),
        numpy_version=data.get("numpy_version"),
    )


def load_metadata(path: Path | str) -> ModelArtifactMetadata:
    """Load a .meta.json file and return typed metadata.

    Raises JSONDecodeError or OSError for unreadable/broken files. Structural
    validation is handled by validate_metadata_dict().
    """
    meta_path = Path(path)
    with meta_path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{meta_path} must contain a JSON object")
    return _coerce_metadata(data)


def validate_metadata_dict(data: dict) -> list[str]:
    """Return validation messages for a metadata dict. Empty list means OK."""
    messages: list[str] = []

    missing = sorted(REQUIRED_FIELDS - set(data))
    for field in missing:
        messages.append(f"ERROR: missing required field: {field}")

    features = data.get("features")
    if "features" in data and (not isinstance(features, list) or not features):
        messages.append("ERROR: features must be a non-empty list")

    metrics = data.get("metrics")
    if "metrics" in data and (not isinstance(metrics, dict) or not metrics):
        messages.append("ERROR: metrics must be a non-empty object")

    train_rows = data.get("train_rows")
    if "train_rows" in data:
        try:
            if int(train_rows) <= 0:
                messages.append("ERROR: train_rows must be > 0")
        except (TypeError, ValueError):
            messages.append("ERROR: train_rows must be an integer > 0")

    # Multi-horizon validation (optional field; old single-horizon artifacts skip).
    if "horizons_min" in data:
        h_val = data["horizons_min"]
        if not isinstance(h_val, list) or not h_val:
            messages.append("ERROR: horizons_min must be a non-empty list of ints")
        elif len(h_val) > _HORIZONS_MAX_LEN:
            messages.append(
                f"ERROR: horizons_min length {len(h_val)} exceeds max {_HORIZONS_MAX_LEN}"
            )
        else:
            non_int = [h for h in h_val if not isinstance(h, int) or isinstance(h, bool)]
            if non_int:
                messages.append(f"ERROR: horizons_min contains non-int entries: {non_int}")
            else:
                out_of_range = [
                    h for h in h_val
                    if h < _HORIZON_MIN_VALUE or h > _HORIZON_MAX_VALUE
                ]
                if out_of_range:
                    messages.append(
                        "ERROR: horizons_min values out of range "
                        f"[{_HORIZON_MIN_VALUE}, {_HORIZON_MAX_VALUE}]: {out_of_range}"
                    )
                if h_val != sorted(h_val):
                    messages.append(
                        f"ERROR: horizons_min must be sorted ascending; got {h_val}"
                    )
                if len(set(h_val)) != len(h_val):
                    messages.append(f"ERROR: horizons_min contains duplicates: {h_val}")
        # Mutually exclusive with the legacy scalar.
        if "horizon_min" in data and data.get("horizon_min") is not None:
            messages.append(
                "ERROR: horizons_min and horizon_min are mutually exclusive - "
                "use horizons_min for multi-horizon, horizon_min for legacy single-output"
            )

    return messages


def validate_artifact_pair(model_path: Path | str, meta_path: Path | str) -> list[str]:
    """Validate a model file and adjacent metadata without loading the model."""
    model = Path(model_path)
    meta = Path(meta_path)
    messages: list[str] = []

    if not model.exists():
        messages.append(f"ERROR: model file does not exist: {model}")
    if not meta.exists():
        messages.append(f"ERROR: metadata file does not exist: {meta}")
        return messages

    if model.suffix not in ALLOWED_MODEL_EXTENSIONS:
        messages.append(f"ERROR: unsupported model extension: {model.suffix}")
    elif model.suffix == ".pkl":
        messages.append("WARNING: .pkl is legacy; prefer .joblib for deployed Pi artifacts")

    with meta.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{meta} must contain a JSON object")

    messages.extend(validate_metadata_dict(data))

    declared_model_file = data.get("model_file")
    if declared_model_file and declared_model_file != model.name:
        messages.append(
            f"ERROR: meta.model_file {declared_model_file!r} does not match {model.name!r}"
        )

    return messages


def find_artifact_pairs(models_dir: Path | str = "models") -> list[tuple[Path, Path]]:
    """Find model/meta pairs by conventional names in models_dir."""
    root = Path(models_dir)
    if not root.exists():
        return []

    pairs: list[tuple[Path, Path]] = []
    for model_path in sorted(root.iterdir()):
        if not model_path.is_file() or model_path.suffix not in ALLOWED_MODEL_EXTENSIONS:
            continue
        meta_path = model_path.with_suffix(".meta.json")
        pairs.append((model_path, meta_path))
    return pairs
