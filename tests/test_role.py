"""Role detection from argv hostname / TAIPY_ROLE env."""
from __future__ import annotations

import sys

import pytest

from src.role import detect_role, pool_max_for_role


@pytest.mark.parametrize("hostname,expected", [
    ("hw@host", "hw"),
    ("ai@host", "ai"),
    ("ext@host", "ext"),
    ("worker@host", "unknown"),
])
def test_argv_hostname(monkeypatch, hostname, expected):
    monkeypatch.setattr(sys, "argv", ["celery", "worker", "--hostname", hostname])
    monkeypatch.delenv("TAIPY_ROLE", raising=False)
    assert detect_role() == expected


def test_taipy_role_env(monkeypatch):
    monkeypatch.setenv("TAIPY_ROLE", "taipy")
    assert detect_role() == "taipy"


def test_pool_max_defaults():
    assert pool_max_for_role("hw") == 2
    assert pool_max_for_role("ai") == 2
    assert pool_max_for_role("ext") == 4
    assert pool_max_for_role("taipy") == 8
    assert pool_max_for_role("beat") == 1


def test_pool_max_env_override(monkeypatch):
    monkeypatch.setenv("POOL_MAX_TAIPY", "16")
    assert pool_max_for_role("taipy") == 16
    monkeypatch.setenv("POOL_MAX_HW", "garbage")
    assert pool_max_for_role("hw") == 2  # falls back to default
