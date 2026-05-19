"""AI infrastructure: feature engineering, model loading, and inference."""
from __future__ import annotations

from .artifacts import (
    ModelArtifactMetadata,
    find_artifact_pairs,
    load_metadata,
    validate_artifact_pair,
    validate_metadata_dict,
)
from .features import build_features, latest_feature_row
from .loader import ModelLoadError, ModelNotFoundError, load_model
from .predictor import Predictor
from .recommender import (
    Recommendation,
    RecommenderResult,
    build_model_readiness,
    build_recommendations,
)
from .registry import get_latest_model, list_available_models

__all__ = [
    "build_features",
    "latest_feature_row",
    "ModelArtifactMetadata",
    "load_metadata",
    "validate_metadata_dict",
    "validate_artifact_pair",
    "find_artifact_pairs",
    "load_model",
    "ModelNotFoundError",
    "ModelLoadError",
    "Predictor",
    "Recommendation",
    "RecommenderResult",
    "build_recommendations",
    "build_model_readiness",
    "list_available_models",
    "get_latest_model",
]
