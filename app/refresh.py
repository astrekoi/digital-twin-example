"""Daemon-thread refresher for the Taipy GUI.

Periodically calls ``gui.broadcast_callback(on_live_tick, [snapshot])`` which
pushes the latest snapshot to every connected session.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING

from app import data_access

if TYPE_CHECKING:
    from taipy.gui import Gui

logger = logging.getLogger(__name__)

_stop = threading.Event()
_thread: threading.Thread | None = None


def _refresh_loop(gui: "Gui", period_s: float) -> None:
    # Import here to avoid circular import at module load.
    from app.web import on_live_tick
    # CRITICAL: module_context tells Taipy in which module's namespace
    # to resolve `state.<var>` assignments. Caller frame is app.refresh
    # (daemon-thread) where state vars are NOT defined -> must point to app.web.
    module_ctx = on_live_tick.__module__  # = "app.web"
    while not _stop.is_set():
        try:
            snapshot = data_access.build_live_snapshot()
            gui.broadcast_callback(on_live_tick, [snapshot], module_context=module_ctx)
        except Exception:
            logger.exception("refresh tick failed")
        # Sleep in 250 ms slices so stop() returns within ~250 ms.
        slept = 0.0
        while slept < period_s and not _stop.is_set():
            time.sleep(0.25)
            slept += 0.25


def start_refresh(gui: "Gui", period_s: float = 1.0) -> None:
    global _thread
    if _thread is not None and _thread.is_alive():
        return
    _stop.clear()
    _thread = threading.Thread(
        target=_refresh_loop,
        args=(gui, period_s),
        name="taipy-refresh",
        daemon=True,
    )
    _thread.start()
    logger.info("refresh thread started: period=%.1fs", period_s)


def stop_refresh() -> None:
    _stop.set()
