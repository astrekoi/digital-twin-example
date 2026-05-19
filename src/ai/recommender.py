"""Read-only recommendations over recent telemetry and optional prediction.

This module does not train models, write files, change runtime config, or touch
hardware. It only describes data/model readiness and operator-facing warnings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import find_artifact_pairs, load_metadata, validate_artifact_pair


@dataclass(frozen=True)
class Recommendation:
    category: str
    severity: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RecommenderResult:
    status: str
    recommendations: list[Recommendation]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "recommendations": [
                {
                    "category": item.category,
                    "severity": item.severity,
                    "message": item.message,
                    "details": item.details,
                }
                for item in self.recommendations
            ],
        }


def _none_fields(row: dict[str, Any]) -> list[str]:
    return [key for key, value in row.items() if value is None]


def build_recommendations(
    recent_readings: list[dict[str, Any]],
    prediction: dict[str, Any] | None = None,
    *,
    model_available: bool = False,
    artifact_errors: list[str] | None = None,
    min_rows_for_model: int = 120,
) -> RecommenderResult:
    """Return read-only AI/data-quality recommendations.

    No automatic control action is ever produced. Operator approval and separate
    runtime/hardware permissions are required for all actuator changes.
    """
    recommendations: list[Recommendation] = [
        Recommendation(
            category="safety",
            severity="info",
            message="Recommendations are read-only and must not change hardware automatically.",
        )
    ]

    if not recent_readings:
        recommendations.append(
            Recommendation(
                category="data_quality",
                severity="warning",
                message="No recent readings available; collect telemetry before forecasting.",
            )
        )
        return RecommenderResult(status="no_data", recommendations=recommendations)

    if len(recent_readings) < min_rows_for_model:
        recommendations.append(
            Recommendation(
                category="data_quality",
                severity="warning",
                message="Need more data for stable model features.",
                details={"rows": len(recent_readings), "min_rows_for_model": min_rows_for_model},
            )
        )

    latest = recent_readings[-1]
    missing = _none_fields(latest)
    if missing:
        recommendations.append(
            Recommendation(
                category="sensor_status",
                severity="warning",
                message="Latest reading contains missing sensor values.",
                details={"fields": missing},
            )
        )

    ts_values = [
        row.get("ts_unix")
        for row in recent_readings
        if isinstance(row.get("ts_unix"), (int, float))
    ]
    if len(ts_values) >= 2:
        max_gap = max(b - a for a, b in zip(ts_values, ts_values[1:]))
        if max_gap > 120:
            recommendations.append(
                Recommendation(
                    category="data_quality",
                    severity="warning",
                    message="Telemetry has a gap larger than 120 seconds.",
                    details={"max_gap_s": max_gap},
                )
            )

    if model_available and artifact_errors:
        recommendations.append(
            Recommendation(
                category="model",
                severity="warning",
                message="Model artifact validation has warnings/errors; do not run predictor until reviewed.",
                details={"artifact_errors": artifact_errors},
            )
        )
        return RecommenderResult(status="model_invalid", recommendations=recommendations)

    if not model_available:
        recommendations.append(
            Recommendation(
                category="model",
                severity="info",
                message="Model not available; deploy real artifacts into models/ before forecasting.",
            )
        )
        return RecommenderResult(status="no_model", recommendations=recommendations)

    if prediction is None:
        recommendations.append(
            Recommendation(
                category="model",
                severity="warning",
                message="Model is available but no prediction result was provided.",
            )
        )
        return RecommenderResult(status="no_prediction", recommendations=recommendations)

    recommendations.append(
        Recommendation(
            category="model",
            severity="info",
            message="Prediction available; review forecast before any manual control action.",
            details={"model_id": prediction.get("model_id"), "horizon_min": prediction.get("horizon_min")},
        )
    )
    return RecommenderResult(status="ok", recommendations=recommendations)


def build_model_readiness(models_dir: Path | str = "models") -> dict[str, Any]:
    """Summarize deployed model metadata without loading model files."""
    root = Path(models_dir)
    pairs = find_artifact_pairs(root)
    if not pairs:
        return {
            "status": "no_model",
            "models_dir": str(root),
            "available_count": 0,
            "latest_model_id": None,
            "artifact_errors": [],
        }

    artifact_errors: list[str] = []
    metadata: list[dict[str, Any]] = []
    for model_path, meta_path in pairs:
        try:
            messages = validate_artifact_pair(model_path, meta_path)
        except Exception as exc:
            messages = [f"ERROR: {meta_path}: {exc}"]
        artifact_errors.extend(f"{model_path.name}: {message}" for message in messages)

        try:
            meta = load_metadata(meta_path)
            metadata.append(
                {
                    "model_id": meta.model_id,
                    "trained_at": meta.trained_at,
                    "model_file": model_path.name,
                }
            )
        except Exception:
            continue

    latest = max(metadata, key=lambda item: item.get("trained_at") or "") if metadata else None
    has_errors = any(": ERROR:" in item or item.startswith("ERROR:") for item in artifact_errors)
    return {
        "status": "model_invalid" if has_errors else "ok",
        "models_dir": str(root),
        "available_count": len(pairs),
        "latest_model_id": latest["model_id"] if latest else None,
        "artifact_errors": artifact_errors,
    }
