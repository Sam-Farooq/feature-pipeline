"""The serving path: the accumulator, the encoding, and what a read can say."""

from __future__ import annotations

import ast
from datetime import timedelta
from pathlib import Path

import pytest

from featurepipe.encoding import AS_OF_FIELD, decode, encode
from featurepipe.online import accumulator as accumulator_module
from featurepipe.online.accumulator import Accumulator
from featurepipe.online.materialize import accumulator_as_of, materialize
from featurepipe.online.store import StaleFeatures, Status
from tests.conftest import AS_OF, LATER


def _imported_modules(module) -> set[str]:
    tree = ast.parse(Path(module.__file__).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_the_serving_path_holds_its_own_implementation():
    """If this module reached the batch path, the parity check would be circular."""
    imported = _imported_modules(accumulator_module)
    assert "pandas" not in imported
    assert not any(name.startswith("featurepipe.offline") for name in imported)
    assert "featurepipe.windows" in imported


def test_a_vector_round_trips_through_redis(account_view, account_events, store):
    materialize(account_view, store, account_events, AS_OF)
    served = store.read_vector(account_view, "A1")
    assert served.status is Status.FRESH
    assert served.values["txn_count_24h"] == 2
    assert served.values["amount_sum_24h"] == pytest.approx(85.75)
    assert served.values["distinct_countries_7d"] == 1


def test_a_computed_unknown_is_not_a_zero(account_view, account_events, store):
    materialize(account_view, store, account_events, AS_OF)
    served = store.read_vector(account_view, "A3")
    assert served.status is Status.FRESH
    assert served.values["txn_count_24h"] == 0
    assert served.values["amount_mean_7d"] is None
    assert served.missing == ()


def test_an_absent_row_is_a_miss_and_not_a_row_of_empty_values(account_view, store):
    served = store.read_vector(account_view, "A1")
    assert served.status is Status.MISS
    assert set(served.values.values()) == {None}
    assert served.missing == tuple(spec.name for spec in account_view.features)
    assert served.as_of is None


def test_a_miss_raises_rather_than_handing_over_numbers(account_view, store):
    with pytest.raises(StaleFeatures, match="miss"):
        store.read_vector(account_view, "A1").require()


def test_a_partly_written_row_is_a_miss(account_view, account_events, store, client):
    materialize(account_view, store, account_events, AS_OF)
    client.hdel(store.key(account_view, "A1"), account_view.feature("max_amount_7d").field)
    served = store.read_vector(account_view, "A1")
    assert served.status is Status.MISS
    assert served.missing == ("max_amount_7d",)
    assert served.values["txn_count_24h"] == 2


def test_a_row_that_cannot_be_dated_is_a_miss(account_view, account_events, store, client):
    materialize(account_view, store, account_events, AS_OF)
    client.hset(store.key(account_view, "A1"), AS_OF_FIELD, "not-a-timestamp")
    served = store.read_vector(account_view, "A1")
    assert served.status is Status.MISS
    assert "unreadable" in served.reason


def test_a_definition_change_renames_the_field_rather_than_reusing_the_value(
    account_view, account_events, store
):
    materialize(account_view, store, account_events, AS_OF)
    narrowed = account_view.with_features(
        tuple(
            spec.model_copy(update={"window": "12h"}) if spec.name == "txn_count_24h" else spec
            for spec in account_view.features
        )
    )
    served = store.read_vector(narrowed, "A1")
    assert served.status is Status.MISS
    assert served.missing == ("txn_count_24h",)
    assert served.values["amount_sum_24h"] == pytest.approx(85.75)


def test_an_entity_the_stream_has_not_seen_has_no_row(account_view, account_events, store):
    materialize(account_view, store, account_events, AS_OF)
    assert store.read_vector(account_view, "A9").status is Status.MISS


def test_an_entity_keeps_a_row_after_its_last_event_ages_out(account_view, account_events):
    """A3's only transaction by 09:00 is older than the 7d window."""
    accumulator = accumulator_as_of(account_view, account_events, AS_OF)
    assert accumulator.retained("A3") == 1
    assert accumulator.evict(AS_OF) == 1
    assert accumulator.retained("A3") == 0
    assert "A3" in accumulator.entities()
    assert accumulator.values_for("A3", AS_OF)["txn_count_24h"] == 0


def test_only_events_recorded_by_then_reach_the_accumulator(account_view, account_events):
    assert accumulator_as_of(account_view, account_events, AS_OF).ingested == 9
    assert accumulator_as_of(account_view, account_events, LATER).ingested == 11


def test_an_event_older_than_the_longest_window_is_dropped_not_stored(
    account_view, account_events
):
    accumulator = accumulator_as_of(account_view, account_events, LATER)
    before = accumulator.retained("A1")
    kept = accumulator.ingest(
        "A1",
        event_at=LATER - timedelta(days=30),
        recorded_at=LATER,
        values={"amount": 1.0, "country": "PK"},
    )
    assert kept is False
    assert accumulator.dropped_too_old == 1
    assert accumulator.retained("A1") == before


def test_an_event_arriving_out_of_recorded_order_is_still_counted(account_view):
    accumulator = Accumulator(account_view)
    accumulator.ingest("A1", AS_OF, AS_OF, {"amount": 2.0, "country": "PK"})
    accumulator.ingest(
        "A1", AS_OF, AS_OF - timedelta(minutes=5), {"amount": 4.0, "country": "PK"}
    )
    assert accumulator.out_of_order == 1
    assert accumulator.watermark == AS_OF
    assert accumulator.values_for("A1", AS_OF)["txn_count_24h"] == 2


def test_two_events_on_the_same_instant_do_not_collide(account_view, account_events):
    accumulator = accumulator_as_of(account_view, account_events, AS_OF)
    for amount in (1.0, 2.0):
        accumulator.ingest("A7", AS_OF, AS_OF, {"amount": amount, "country": "PK"})
    assert accumulator.values_for("A7", AS_OF)["amount_sum_24h"] == pytest.approx(3.0)


def test_asking_for_an_entity_never_ingested_is_an_error_not_a_zero(
    account_view, account_events
):
    accumulator = accumulator_as_of(account_view, account_events, AS_OF)
    with pytest.raises(KeyError, match="nothing ingested"):
        accumulator.values_for("A9", AS_OF)


@pytest.mark.parametrize("value", [0, 1, 99])
def test_ints_round_trip(account_view, value):
    spec = account_view.feature("txn_count_24h")
    assert decode(encode(value, spec), spec) == value


@pytest.mark.parametrize("value", [0.0, 85.75, 445.75 / 5, 1e-12])
def test_floats_round_trip_exactly(account_view, value):
    spec = account_view.feature("amount_sum_24h")
    assert decode(encode(value, spec), spec) == value


def test_an_unknown_round_trips_as_an_unknown(account_view):
    spec = account_view.feature("amount_mean_7d")
    assert decode(encode(None, spec), spec) is None
    assert encode(None, spec) != ""
    assert encode(None, spec) != encode(0.0, spec)


def test_bytes_from_a_client_without_decode_responses_still_decode(account_view):
    spec = account_view.feature("txn_count_24h")
    assert decode(b"7", spec) == 7


def test_writing_nothing_is_refused(account_view, store):
    with pytest.raises(ValueError, match="nothing to write"):
        store.write_vector(account_view, "A1", {}, AS_OF)


def test_a_row_is_written_with_its_ttl_in_one_transaction(account_view, account_events, store):
    report = materialize(account_view, store, account_events, AS_OF)
    assert report.rows_written == 3
    assert report.fields_written == 15
    assert store.ttl_of(account_view, "A1") == account_view.ttl_seconds


def test_dropping_a_row_leaves_the_others(account_view, account_events, store):
    materialize(account_view, store, account_events, AS_OF)
    assert store.drop(account_view, "A1") == 1
    assert store.read_vector(account_view, "A1").status is Status.MISS
    assert store.read_vector(account_view, "A2").status is Status.FRESH
