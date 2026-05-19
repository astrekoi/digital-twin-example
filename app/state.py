"""Global state object for the Taipy GUI.

Taipy 4.x state is per-session - each browser tab gets its own state. This
module declares the variables that ``app/web.py`` exposes via the ``state``
namespace, plus light-weight initialisation routines.
"""
from __future__ import annotations

from typing import Any

# Live values (overview page)
latest: dict[str, Any] = {}
latest_age_s: float | None = None
data_status: str = "loading"

# Predictions / forecast
last_prediction: dict[str, Any] | None = None
prediction_value: float | None = None
prediction_horizon_min: int | None = None
prediction_model_id: str | None = None
prediction_at_iso: str | None = None

# Events / proofs
events_recent: list[dict[str, Any]] = []
proofs_recent: list[dict[str, Any]] = []

# Charts
chart_range_minutes: int = 60
chart_figure: dict[str, Any] = {"data": [], "layout": {}}

# Control
relay_states: list[bool] = [False, False, False, False]
servo_angle: float = 90.0
servo_channel: int = 1
last_command_at: float = 0.0

# Settings (read-only display from .env / runtime config)
ai_enabled: bool = False
ipfs_provider: str = "none"
hedera_enabled: bool = False
hedera_topic_id: str = ""
allow_external_api_calls: bool = False

# Misc
last_error: str = ""
