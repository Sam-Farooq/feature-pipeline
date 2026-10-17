"""Freshness, the clock it is measured on, and the ttl arithmetic.

The store clock is a `ManualClock` here, started at AS_OF and advanced by the
test. That is not a convenience: an age has to be exact to be asserted on, and
the point of the whole file is which clock the number comes from.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from featurepipe.encoding import AS_OF_FIELD, WRITTEN_AT_FIELD, encode_instant
from featurepipe.online.materialize import materialize
from featurepipe.online.store import StaleFeatures, Status
from featurepipe.spec import FeatureView
from tests.conftest import AS_OF


@pytest.fixture
def filled(account_view, account_events, timed_store):
    materialize(account_view, timed_store, account_events, AS_OF)
    return timed_store


def test_a_row_read_the_moment_it_was_stored_is_fresh(account_view, filled):
    served = filled.read_vector(account_view, "A1")
    assert served.status is Status.FRESH
    assert served.age_seconds == 0.0
    assert served.require()["txn_count_24h"] == 2


def test_a_row_inside_the_budget_is_fresh(account_view, filled, store_clock):
    assert account_view.max_staleness_seconds == 1200
    store_clock.advance(1200)
    assert filled.read_vector(account_view, "A1").status is Status.FRESH


def test_one_second_past_the_budget_is_stale(account_view, filled, store_clock):
    store_clock.advance(1201)
    served = filled.read_vector(account_view, "A1")
    assert served.status is Status.STALE
    assert served.age_seconds == 1201.0


def test_a_stale_row_still_carries_its_numbers_and_its_age(account_view, filled, store_clock):
    store_clock.advance(7200)
    served = filled.read_vector(account_view, "A1")
    assert served.values["txn_count_24h"] == 2
    assert served.age_seconds == 7200.0
    assert "budget is 1200s" in served.reason


def test_a_caller_that_needs_fresh_features_gets_an_error_not_old_ones(
    account_view, filled, store_clock
):
    store_clock.advance(7200)
    with pytest.raises(StaleFeatures, match="stale"):
        filled.read_vector(account_view, "A1").require()


def test_a_row_stamped_by_a_fast_clock_is_not_fresh_forever(
    account_view, filled, client, store_clock
):
    """The row this issue is about: a writer an hour fast stamped it an hour ahead.

    It is read 25 minutes after the store took it, five minutes past the twenty
    minute budget, so STALE is the only correct answer. Dating it by `as_of`
    gave an age of -1500 seconds and the answer FRESH.

    The stamp goes in with a direct HSET because `write_vector` now refuses an
    as_of that far ahead of the store clock. This is the row such a writer
    leaves behind, judged by the read path that has to cope with it.
    """
    ahead = AS_OF + timedelta(hours=1)
    client.hset(filled.key(account_view, "A1"), AS_OF_FIELD, encode_instant(ahead))
    store_clock.advance(25 * 60)
    served = filled.read_vector(account_view, "A1")
    assert served.status is Status.STALE
    assert served.age_seconds == 1500.0
    assert served.as_of == ahead
    assert served.written_at == AS_OF


def test_an_as_of_behind_the_store_clock_does_not_age_the_row(
    account_view, account_events, timed_store
):
    """And what that gives up.

    A job computing an hour-old window writes a row that is new in the store
    and old in event time, and it reads FRESH. A row cannot tell a lagging job
    apart from a slow clock, so the read path judges only what it can see: how
    long the store has held the row. Watching the lag belongs where as_of and
    the store clock are both in hand, which is the write.
    """
    materialize(account_view, timed_store, account_events, AS_OF - timedelta(hours=1))
    served = timed_store.read_vector(account_view, "A1")
    assert served.status is Status.FRESH
    assert served.age_seconds == 0.0
    assert served.as_of == AS_OF - timedelta(hours=1)


def test_a_row_with_no_store_write_time_is_a_miss(account_view, filled, client):
    """A row written by the version before this one cannot be dated.

    It reads as a MISS until the next materialisation, which is ten minutes for
    account_activity, rather than being judged on the writer's clock again.
    """
    client.hdel(filled.key(account_view, "A1"), WRITTEN_AT_FIELD)
    served = filled.read_vector(account_view, "A1")
    assert served.status is Status.MISS
    assert WRITTEN_AT_FIELD in served.reason
    assert served.values["txn_count_24h"] is None


def test_a_row_whose_store_write_time_is_unreadable_is_a_miss(account_view, filled, client):
    client.hset(filled.key(account_view, "A1"), WRITTEN_AT_FIELD, "half past four")
    served = filled.read_vector(account_view, "A1")
    assert served.status is Status.MISS
    assert "unreadable" in served.reason


def test_a_store_clock_that_steps_backwards_says_so_rather_than_serving(
    account_view, filled, store_clock
):
    """An NTP correction can move the clock the row was stamped on.

    Inside the skew allowance that is noise and the row is still served. Beyond
    it the age is not a number anyone should act on, so the answer is STALE
    with the reason, not a FRESH row from a negative age.
    """
    store_clock.advance(-3)
    assert filled.read_vector(account_view, "A1").status is Status.FRESH
    store_clock.advance(-7200)
    served = filled.read_vector(account_view, "A1")
    assert served.status is Status.STALE
    assert "two clocks disagree" in served.reason
    assert served.values["txn_count_24h"] == 2


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
    served = filled.read_vector(account_view, "A1")
    assert served.status is Status.MISS
    assert served.values["txn_count_24h"] is None
    assert served.values["txn_count_24h"] != 0


def test_rewriting_a_row_pushes_the_ttl_back_out(account_view, account_events, filled, client):
    client.expire(filled.key(account_view, "A1"), 30)
    assert filled.ttl_of(account_view, "A1") == 30
    materialize(account_view, filled, account_events, AS_OF)
    assert filled.ttl_of(account_view, "A1") == account_view.ttl_seconds


def test_rewriting_a_row_restamps_how_long_the_store_has_held_it(
    account_view, account_events, filled, store_clock
):
    store_clock.advance(1800)
    assert filled.read_vector(account_view, "A1").status is Status.STALE
    materialize(account_view, filled, account_events, AS_OF)
    served = filled.read_vector(account_view, "A1")
    assert served.status is Status.FRESH
    assert served.age_seconds == 0.0


def test_the_merchant_view_keeps_its_own_tighter_budget(
    merchant_view, merchant_events, timed_store, store_clock
):
    materialize(merchant_view, timed_store, merchant_events, AS_OF)
    assert timed_store.ttl_of(merchant_view, "M1") == 21600
    assert merchant_view.max_staleness_seconds == 300
    store_clock.advance(4 * 60)
    assert timed_store.read_vector(merchant_view, "M1").status is Status.FRESH
    store_clock.advance(2 * 60)
    assert timed_store.read_vector(merchant_view, "M1").status is Status.STALE
    assert timed_store.read_vector(merchant_view, "M9").status is Status.MISS
