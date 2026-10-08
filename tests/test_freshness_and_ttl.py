"""Freshness, and the ttl arithmetic that keeps a row alive long enough."""

from __future__ import annotations

from datetime import timedelta

import pytest

from featurepipe.online.materialize import materialize
from featurepipe.online.store import StaleFeatures, Status
from featurepipe.spec import FeatureView
from tests.conftest import AS_OF


@pytest.fixture
def filled(account_view, account_events, store):
    materialize(account_view, store, account_events, AS_OF)
    return store


def test_a_row_read_at_its_own_as_of_is_fresh(account_view, filled):
    served = filled.read_vector(account_view, "A1", AS_OF)
    assert served.status is Status.FRESH
    assert served.age_seconds == 0.0
    assert served.require()["txn_count_24h"] == 2


def test_a_row_inside_the_budget_is_fresh(account_view, filled):
    assert account_view.max_staleness_seconds == 1200
    at = AS_OF + timedelta(seconds=1200)
    assert filled.read_vector(account_view, "A1", at).status is Status.FRESH


def test_one_second_past_the_budget_is_stale(account_view, filled):
    served = filled.read_vector(account_view, "A1", AS_OF + timedelta(seconds=1201))
    assert served.status is Status.STALE
    assert served.age_seconds == 1201.0


def test_a_stale_row_still_carries_its_numbers_and_its_age(account_view, filled):
    served = filled.read_vector(account_view, "A1", AS_OF + timedelta(hours=2))
    assert served.values["txn_count_24h"] == 2
    assert served.age_seconds == 7200.0
    assert "budget is 1200s" in served.reason


def test_a_caller_that_needs_fresh_features_gets_an_error_not_old_ones(account_view, filled):
    with pytest.raises(StaleFeatures, match="stale"):
        filled.read_vector(account_view, "A1", AS_OF + timedelta(hours=2)).require()


def test_the_ttl_is_the_views_ttl_and_not_the_staleness_budget(account_view, filled):
    assert filled.ttl_of(account_view, "A1") == 129600
    assert account_view.ttl_seconds > account_view.max_staleness_seconds


def test_the_ttl_outlives_a_materialisation_cycle_in_both_shipped_views(registry):
    for view in registry.views:
        floor = view.materialize_every_seconds + view.max_staleness_seconds
        assert view.ttl_seconds >= floor


def test_a_ttl_that_would_expire_between_runs_cannot_be_declared():
    with pytest.raises(ValueError, match="keys expire between runs"):
        FeatureView(
            name="too_short",
            entity="account_id",
            event_time="event_at",
            recorded_time="recorded_at",
            ttl="5m",
            materialize_every="10m",
            max_staleness="20m",
            features=(
                {
                    "name": "txn_count_24h",
                    "agg": "count",
                    "window": "24h",
                    "dtype": "int",
                    "empty_window": "zero",
                },
            ),
        )


def test_an_expired_row_reads_as_a_miss_rather_than_as_zeros(account_view, filled, client):
    client.delete(filled.key(account_view, "A1"))
    served = filled.read_vector(account_view, "A1", AS_OF)
    assert served.status is Status.MISS
    assert served.values["txn_count_24h"] is None
    assert served.values["txn_count_24h"] != 0


def test_rewriting_a_row_pushes_the_ttl_back_out(account_view, account_events, filled, client):
    client.expire(filled.key(account_view, "A1"), 30)
    assert filled.ttl_of(account_view, "A1") == 30
    materialize(account_view, filled, account_events, AS_OF)
    assert filled.ttl_of(account_view, "A1") == account_view.ttl_seconds


def test_the_merchant_view_keeps_its_own_tighter_budget(merchant_view, merchant_events, store):
    materialize(merchant_view, store, merchant_events, AS_OF)
    assert store.ttl_of(merchant_view, "M1") == 21600
    assert merchant_view.max_staleness_seconds == 300
    inside = AS_OF + timedelta(minutes=4)
    assert store.read_vector(merchant_view, "M1", inside).status is Status.FRESH
    outside = AS_OF + timedelta(minutes=6)
    assert store.read_vector(merchant_view, "M1", outside).status is Status.STALE
    assert store.read_vector(merchant_view, "M9", AS_OF).status is Status.MISS
