"""Celery application: queues, task_routes, beat schedule, lifecycle signals.

Run from project root:

    # 1 worker per role + 1 beat
    celery -A src.tasks.celery_app worker -Q hardware_priority,hardware --pool=solo --hostname=hw@%h
    celery -A src.tasks.celery_app worker -Q ai --pool=prefork --concurrency=1 --max-tasks-per-child=100 --hostname=ai@%h
    celery -A src.tasks.celery_app worker -Q external --pool=threads --concurrency=2 --hostname=ext@%h
    celery -A src.tasks.celery_app beat --schedule=data/celerybeat-schedule.db
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from celery import Celery
from celery.signals import worker_init, worker_process_init, worker_shutdown

from src.config import load_env_config
from src.env import load_project_env

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_project_env(PROJECT_ROOT / ".env")
_env = load_env_config(PROJECT_ROOT / ".env")

logger = logging.getLogger(__name__)

celery_app = Celery(
    "plc_twin",
    broker=_env.celery.redis_url,
    backend=_env.celery.redis_url,
    include=[
        "src.tasks.hardware",
        "src.tasks.ai",
        "src.tasks.external",
    ],
)

celery_app.conf.update(
    timezone=_env.celery.timezone,
    enable_utc=False,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    broker_connection_retry_on_startup=True,
    task_routes={
        "src.tasks.hardware.kill_switch_task": {"queue": "hardware_priority"},
        "src.tasks.hardware.retention_cleanup_task": {"queue": "hardware"},
        "src.tasks.ai.predict_task": {"queue": "ai"},
        "src.tasks.external.publish_pinata_task": {"queue": "external"},
        "src.tasks.external.publish_hedera_task": {"queue": "external"},
        "src.tasks.external.publish_digest_task": {"queue": "external"},
    },
    beat_schedule={
        "predict": {
            "task": "src.tasks.ai.predict_task",
            "schedule": float(_env.celery.predict_period_s),
        },
        "publish-digest": {
            "task": "src.tasks.external.publish_digest_task",
            "schedule": float(_env.celery.digest_period_s),
        },
        "retention-cleanup": {
            "task": "src.tasks.hardware.retention_cleanup_task",
            "schedule": float(_env.celery.retention_period_s),
        },
    },
)


@worker_init.connect
def _on_worker_init(sender=None, **_kwargs):
    """Runs in solo/threads workers (hw, ext) before consuming tasks.

    For ai-worker (prefork), worker_process_init handles per-fork init instead.
    """
    hostname = (sender.hostname if sender else "") or ""
    logger.info("worker_init: hostname=%s", hostname)
    if hostname.startswith("hw@"):
        try:
            from src.tasks.hardware import init_hardware_resources, start_apscheduler

            if not _env.safety.allow_hardware_runtime:
                logger.warning(
                    "ALLOW_HARDWARE_RUNTIME is false - hw-worker will refuse to touch GPIO. "
                    "Set ALLOW_HARDWARE_RUNTIME=true in .env when stand is wired."
                )
                return
            init_hardware_resources()
            start_apscheduler(
                sample_period_s=_env.celery.sample_period_s,
                apply_commands_period_s=_env.celery.apply_commands_period_s,
            )
        except Exception:
            logger.exception("hw-worker init failed")


@worker_process_init.connect
def _on_worker_process_init(sender=None, **_kwargs):
    """Runs per fork in prefork pool (ai-worker)."""
    hostname = ""
    if sender is not None:
        hostname = getattr(sender, "hostname", "") or ""
    if not hostname:
        hostname = os.environ.get("CELERY_HOSTNAME", "")
    logger.info("worker_process_init: hostname=%s", hostname)
    if hostname.startswith("ai@"):
        try:
            from src.tasks.ai import init_ai_resources

            init_ai_resources()
        except Exception:
            logger.exception("ai-worker init failed (lazy retry on first task)")


@worker_shutdown.connect
def _on_worker_shutdown(sender=None, **_kwargs):
    hostname = (sender.hostname if sender else "") or ""
    logger.info("worker_shutdown: hostname=%s", hostname)
    if hostname.startswith("hw@"):
        try:
            from src.tasks.hardware import cleanup_hardware_resources, stop_apscheduler

            stop_apscheduler()
            cleanup_hardware_resources()
        except Exception:
            logger.exception("hw-worker shutdown cleanup failed")
