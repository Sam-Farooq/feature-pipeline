"""The centrepiece: one definition, two implementations, the same answers.

Both shipped views are run down both paths over every spine row, and the
comparison is made against what the serving code actually hands back, Redis
encoding included.

The second half of this file matters as much as the first. A parity check that
cannot fail proves nothing, so the known ways the two paths can drift are
reintroduced on purpose and the check is required to catch each one.
"""

from __future__ import annotations

import pandas as pd
import pytest

from featurepipe.offline.frames import AS_OF, compute_training_frame
from featurepipe.online.accumulator import apply_aggregation
from featurepipe.parity import check_parity
from featurepipe.windows import window_bounds


def test_the_account_view_agrees_down_both_paths(
    account_view, account_events, account_spine, client
):
    report = check_parity(account_view, account_events, account_spine, client=client)
    assert report.disagreements == ()
    assert report.rows == 4
    assert report.comparisons == 20
    assert report.agreed


def test_the_merchant_view_agrees_down_both_paths(
    merchant_view, merchant_events, merchant_spine, client
):
    report = check_parity(merchant_view, merchant_events, merchant_spine, client=client)
    assert report.disagreements == ()
    assert report.comparisons == 6


def test_parity_covers_the_late_arrival_and_the_empty_window(
    account_view, account_events, account_spine, client
):
    """The two rows worth the whole exercise are in the spine the check runs on."""
    batch = compute_training_frame(account_view, account_events, account_spine)
    late = batch[batch[AS_OF] == pd.Timestamp("2026-03-02T09:00:00Z")]
    assert late[late["account_id"] == "A1"]["txn_count_24h"].item() == 2
    assert pd.isna(late[late["account_id"] == "A3"]["amount_mean_7d"].item())
    assert check_parity(account_view, account_events, account_spine, client=client).agreed


def test_a_window_edge_moved_in_one_path_only_is_caught(
    monkeypatch, account_view, account_events, account_spine, client
):
    """The bug this repo actually had: an inclusive lower bound on one side.

    A1 has a transaction stamped exactly 24h before its 09:00 label, so an
    inclusive edge in the serving path alone turns a count of 2 into 3.
    """

    def inclusive(event_at, as_of, window_seconds):
        lower, upper = window_bounds(as_of, window_seconds)
        return (event_at >= lower) & (event_at <= upper)

    monkeypatch.setattr("featurepipe.online.accumulator.within_window", inclusive)
    report = check_parity(account_view, account_events, account_spine, client=client)
    assert not report.agreed
    caught = {item.feature for item in report.disagreements}
    assert "txn_count_24h" in caught
    assert "amount_sum_24h" in caught


def test_an_empty_window_answered_with_a_zero_instead_of_an_unknown_is_caught(
    monkeypatch, account_view, account_events, account_spine, client
):
    """A3's 7d mean is unknown. A serving path that answered 0.0 would be wrong
    in the one direction a model cannot detect."""

    def zero_instead(spec, rows):
        if not rows:
            return 0 if spec.dtype == "int" else 0.0
        return apply_aggregation(spec, rows)

    monkeypatch.setattr("featurepipe.online.accumulator.apply_aggregation", zero_instead)
    report = check_parity(account_view, account_events, account_spine, client=client)
    assert not report.agreed
    assert {item.feature for item in report.disagreements} == {
        "amount_mean_7d",
        "max_amount_7d",
    }
    assert all(item.entity_id == "A3" for item in report.disagreements)


def test_a_serving_path_that_ignores_the_recorded_time_is_caught(
    monkeypatch, account_view, account_events, account_spine, client
):
    """Point-in-time correctness, from the other side. If the stream were fed
    every event regardless of when it was recorded, A1's 09:00 row would see
    the 500.00 that was not recorded until 11:15."""
    import featurepipe.parity as parity_module
    from featurepipe.online.accumulator import Accumulator

    def everything(view, events, as_of):
        accumulator = Accumulator(view)
        for row in events.sort_values(view.recorded_time).to_dict("records"):
            accumulator.ingest(
                str(row[view.entity]),
                row[view.event_time].to_pydatetime(),
                row[view.recorded_time].to_pydatetime(),
                {column: row[column] for column in view.columns},
            )
        return accumulator

    monkeypatch.setattr(parity_module, "accumulator_as_of", everything)
    report = check_parity(account_view, account_events, account_spine, client=client)
    assert not report.agreed
    at_nine = [item for item in report.disagreements if item.as_of.hour == 9]
    assert {item.feature for item in at_nine} >= {"txn_count_24h", "distinct_countries_7d"}


def test_an_entity_the_stream_has_never_seen_is_reported_not_hidden(
    account_view, account_events, account_spine, client
):
    unseen = pd.concat(
        [
            account_spine,
            pd.DataFrame(
                {
                    "account_id": ["A9"],
                    AS_OF: [pd.Timestamp("2026-03-02T09:00:00Z")],
                    "is_fraud": ["0"],
                }
            ),
        ],
        ignore_index=True,
    )
    report = check_parity(account_view, account_events, unseen, client=client)
    assert not report.agreed
    only = report.disagreements[0]
    assert (only.entity_id, only.feature) == ("A9", "*")
    assert "never seen this entity" in only.note
    assert "A9" in str(only)


def test_a_float_difference_inside_the_tolerance_is_not_a_disagreement(
    monkeypatch, account_view, account_events, account_spine, client
):
    """Both paths add the same numbers in different orders, so float sums can
    differ in the last bits. One part in a billion is called agreement here,
    and a feature where that changes a decision is not one this defends."""

    def nudged(spec, rows):
        value = apply_aggregation(spec, rows)
        if spec.dtype == "float" and value:
            return value * (1 + 1e-12)
        return value

    monkeypatch.setattr("featurepipe.online.accumulator.apply_aggregation", nudged)
    assert check_parity(account_view, account_events, account_spine, client=client).agreed
    strict = check_parity(
        account_view, account_events, account_spine, client=client, rel_tol=1e-15
    )
    assert not strict.agreed


def test_a_disagreement_prints_the_entity_the_instant_and_both_answers(
    monkeypatch, account_view, account_events, account_spine, client
):
    monkeypatch.setattr(
        "featurepipe.online.accumulator.apply_aggregation",
        lambda spec, rows: 999 if spec.dtype == "int" else 999.0,
    )
    report = check_parity(account_view, account_events, account_spine, client=client)
    printed = str(report.disagreements[0])
    assert "account_activity/A1" in printed
    assert "2026-03-02T09:00:00Z" in printed
    assert "999" in printed


def test_the_check_reads_through_the_real_serving_path(
    monkeypatch, account_view, account_events, account_spine, client
):
    """A round trip bug is a parity bug. An encoder that loses an unknown is
    caught here and nowhere else."""
    monkeypatch.setattr("featurepipe.online.store.encode", lambda value, spec: "0")
    report = check_parity(account_view, account_events, account_spine, client=client)
    assert not report.agreed


@pytest.mark.parametrize("view_name", ["account_activity", "merchant_activity"])
def test_every_shipped_view_is_checked(
    view_name, registry, account_events, merchant_events, account_spine, merchant_spine, client
):
    events = account_events if view_name == "account_activity" else merchant_events
    spine = account_spine if view_name == "account_activity" else merchant_spine
    report = check_parity(registry.view(view_name), events, spine, client=client)
    assert report.agreed, [str(item) for item in report.disagreements]
