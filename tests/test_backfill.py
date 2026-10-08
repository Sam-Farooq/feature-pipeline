"""Adding a feature to rows that already exist, and to a frame already built."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd
import pytest

from featurepipe.backfill import backfill_offline, backfill_online
from featurepipe.encoding import AS_OF_FIELD
from featurepipe.offline.frames import compute_training_frame
from featurepipe.online.materialize import materialize
from featurepipe.online.store import FeatureStore, Status
from tests.conftest import AS_OF


class ExpiresOnWrite:
    """A client whose key disappears between the as_of read and the field write.

    Two round trips cannot be atomic without a script, so this is the race the
    store has to survive rather than the one it can prevent.
    """

    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def hset(self, key, *args, **kwargs):
        self._inner.delete(key)
        return self._inner.hset(key, *args, **kwargs)


@pytest.fixture
def extended(account_view, new_feature):
    return account_view.with_features((*account_view.features, new_feature))


@pytest.fixture
def filled(account_view, account_events, store):
    materialize(account_view, store, account_events, AS_OF)
    return store


def test_a_new_feature_is_a_miss_before_it_is_backfilled(extended, filled):
    served = filled.read_vector(extended, "A1", AS_OF)
    assert served.status is Status.MISS
    assert served.missing == ("amount_sum_1h",)


def test_backfilling_writes_one_field_on_every_row_that_exists(
    extended, new_feature, account_events, filled
):
    report = backfill_online(extended, new_feature, filled, account_events, AS_OF)
    assert (report.written, report.absent, report.no_history) == (3, 0, 0)
    served = filled.read_vector(extended, "A1", AS_OF)
    assert served.status is Status.FRESH
    assert served.values["amount_sum_1h"] == pytest.approx(10.25)


def test_the_other_features_are_untouched(
    extended, new_feature, account_events, filled, client
):
    key = filled.key(extended, "A1")
    before = client.hgetall(key)
    backfill_online(extended, new_feature, filled, account_events, AS_OF)
    after = client.hgetall(key)
    assert set(after) - set(before) == {new_feature.field}
    assert {name: after[name] for name in before} == before


def test_the_ttl_does_not_move(extended, new_feature, account_events, filled, client):
    client.expire(filled.key(extended, "A1"), 600)
    backfill_online(extended, new_feature, filled, account_events, AS_OF)
    assert filled.ttl_of(extended, "A1") == 600


def test_the_key_is_the_same_key(account_view, extended, filled):
    assert filled.key(extended, "A1") == filled.key(account_view, "A1")


def test_an_absent_row_is_not_created_by_a_backfill(
    extended, new_feature, account_events, filled, client
):
    filled.drop(extended, "A3")
    report = backfill_online(extended, new_feature, filled, account_events, AS_OF, ["A3"])
    assert (report.written, report.absent) == (0, 1)
    assert client.exists(filled.key(extended, "A3")) == 0


def test_an_entity_the_stream_never_saw_is_reported_rather_than_skipped(
    extended, new_feature, account_events, filled
):
    report = backfill_online(extended, new_feature, filled, account_events, AS_OF, ["A9"])
    assert (report.written, report.no_history, report.considered) == (0, 1, 1)


def test_a_row_stamped_at_another_instant_is_left_alone(
    extended, new_feature, account_events, filled, client
):
    """Every field in a row shares one as_of, so a late field is refused."""
    report = backfill_online(
        extended, new_feature, filled, account_events, AS_OF + timedelta(minutes=5)
    )
    assert report.as_of_mismatch == 3
    assert report.written == 0
    assert new_feature.field not in client.hgetall(filled.key(extended, "A1"))


def test_an_undateable_row_is_refused(extended, new_feature, account_events, filled, client):
    client.hset(filled.key(extended, "A1"), AS_OF_FIELD, "wednesday")
    report = backfill_online(extended, new_feature, filled, account_events, AS_OF)
    assert report.unreadable_as_of == 1
    assert report.written == 2


def test_a_row_that_expires_mid_write_is_removed_rather_than_left_immortal(
    extended, new_feature, account_events, client
):
    store = FeatureStore(ExpiresOnWrite(client))
    materialize(extended, store, account_events, AS_OF)
    report = backfill_online(extended, new_feature, store, account_events, AS_OF)
    assert report.expired_mid_write == 3
    assert report.written == 0
    assert client.exists(store.key(extended, "A1")) == 0
    assert store.read_vector(extended, "A1", AS_OF).status is Status.MISS


def test_the_backfilled_value_matches_what_a_full_materialisation_would_write(
    extended, new_feature, account_events, filled, client
):
    backfill_online(extended, new_feature, filled, account_events, AS_OF)
    backfilled = client.hget(filled.key(extended, "A1"), new_feature.field)
    client.flushall()
    materialize(extended, filled, account_events, AS_OF)
    assert client.hget(filled.key(extended, "A1"), new_feature.field) == backfilled


def test_a_training_frame_gains_one_column_and_keeps_the_rest(
    account_view, extended, new_feature, account_events, account_spine
):
    original = compute_training_frame(account_view, account_events, account_spine)
    widened = backfill_offline(extended, new_feature, account_events, original)
    assert list(widened.columns) == [*original.columns, "amount_sum_1h"]
    pd.testing.assert_frame_equal(widened[original.columns], original)
    assert widened["amount_sum_1h"].tolist() == [10.25, 90.0, 0.0, 0.0]


def test_a_column_that_is_already_there_is_refused(account_view, account_events, account_spine):
    original = compute_training_frame(account_view, account_events, account_spine)
    with pytest.raises(ValueError, match="already a column"):
        backfill_offline(
            account_view, account_view.feature("txn_count_24h"), account_events, original
        )


def test_the_offline_backfill_keeps_the_frames_own_index(
    account_view, extended, new_feature, account_events, account_spine
):
    original = compute_training_frame(account_view, account_events, account_spine)
    shifted = original.set_index(pd.Index([10, 11, 12, 13], name="row_id"))
    widened = backfill_offline(extended, new_feature, account_events, shifted)
    assert widened.index.tolist() == [10, 11, 12, 13]
    assert widened.loc[10, "amount_sum_1h"] == pytest.approx(10.25)
