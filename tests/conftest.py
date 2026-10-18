from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from featurepipe.clients import memory_client
from featurepipe.clock import ManualClock
from featurepipe.offline.frames import load_events, load_spine
from featurepipe.online.store import FeatureStore
from featurepipe.registry import Registry
from featurepipe.spec import FeatureSpec

ROOT = Path(__file__).resolve().parents[1]
EVENTS = str(ROOT / "examples" / "events.csv")

AS_OF = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
LATER = datetime(2026, 3, 2, 12, 0, tzinfo=UTC)


@pytest.fixture
def registry() -> Registry:
    return Registry.from_directory(ROOT / "definitions")


@pytest.fixture
def account_view(registry):
    return registry.view("account_activity")


@pytest.fixture
def merchant_view(registry):
    return registry.view("merchant_activity")


@pytest.fixture
def account_events(account_view):
    return load_events(EVENTS, account_view)


@pytest.fixture
def merchant_events(merchant_view):
    return load_events(EVENTS, merchant_view)


@pytest.fixture
def account_spine(account_view):
    return load_spine(str(ROOT / "examples" / "spine_accounts.csv"), account_view)


@pytest.fixture
def merchant_spine(merchant_view):
    return load_spine(str(ROOT / "examples" / "spine_merchants.csv"), merchant_view)


@pytest.fixture
def client():
    return memory_client()


@pytest.fixture
def store(client):
    """A store on the store's own clock, which is what production has."""
    return FeatureStore(client)


@pytest.fixture
def store_clock() -> ManualClock:
    """The store clock, started at AS_OF, for tests that have to move time."""
    return ManualClock(AS_OF)


@pytest.fixture
def timed_store(client, store_clock):
    """A store whose clock the test advances, so an age is exact and not nearly."""
    return FeatureStore(client, clock=store_clock)


@pytest.fixture
def new_feature() -> FeatureSpec:
    return FeatureSpec(
        name="amount_sum_1h",
        agg="sum",
        column="amount",
        window="1h",
        dtype="float",
        empty_window="zero",
    )
