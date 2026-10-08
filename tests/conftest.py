from __future__ import annotations

from pathlib import Path

import pytest

from featurepipe.registry import Registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def registry() -> Registry:
    return Registry.from_directory(ROOT / "definitions")


@pytest.fixture
def account_view(registry):
    return registry.view("account_activity")


@pytest.fixture
def merchant_view(registry):
    return registry.view("merchant_activity")
