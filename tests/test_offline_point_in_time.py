"""The batch path, and the leak it exists to stop."""

from __future__ import annotations

import pandas as pd
import pytest

from featurepipe.offline.frames import AS_OF, compute_training_frame, load_events
from tests.conftest import EVENTS


def row(frame: pd.DataFrame, entity: str, at: str) -> pd.Series:
    want = pd.Timestamp(at)
    match = frame[(frame.iloc[:, 0] == entity) & (frame[AS_OF] == want)]
    assert len(match) == 1, f"expected one row for {entity} at {at}, got {len(match)}"
    return match.iloc[0]


@pytest.fixture
def frame(account_view, account_events, account_spine):
    return compute_training_frame(account_view, account_events, account_spine)


def test_one_row_per_spine_row_and_the_spine_columns_survive(frame, account_spine):
    assert len(frame) == len(account_spine)
    assert list(frame.columns)[:3] == ["account_id", AS_OF, "is_fraud"]
    assert frame["is_fraud"].tolist() == ["0", "0", "0", "1"]


def test_an_event_recorded_after_the_label_is_invisible_to_it(frame):
    """A1's 500.00 happened at 22:40 and was recorded at 11:15 the next day.

    At 09:00 it has not been recorded, so it is not in the 24h count, not in
    the sum, and its country is not in the distinct count. Filtering on
    event_at alone would put all three in.
    """
    at_nine = row(frame, "A1", "2026-03-02T09:00:00Z")
    assert at_nine["txn_count_24h"] == 2
    assert at_nine["amount_sum_24h"] == pytest.approx(85.75)
    assert at_nine["distinct_countries_7d"] == 1


def test_the_same_entity_three_hours_later_can_see_it(frame):
    at_noon = row(frame, "A1", "2026-03-02T12:00:00Z")
    assert at_noon["txn_count_24h"] == 3
    assert at_noon["amount_sum_24h"] == pytest.approx(585.75)
    assert at_noon["distinct_countries_7d"] == 2


def test_an_event_on_the_lower_edge_is_outside_the_window(frame):
    """A1's 120.00 is stamped exactly 24h before the 09:00 label.

    It is outside the 24h window and inside the 7d one, where it is still the
    largest amount this account has recorded by then.
    """
    at_nine = row(frame, "A1", "2026-03-02T09:00:00Z")
    assert at_nine["txn_count_24h"] == 2
    assert at_nine["max_amount_7d"] == pytest.approx(200.0)
    assert at_nine["amount_mean_7d"] == pytest.approx(445.75 / 5)


def test_an_event_recorded_exactly_at_the_label_time_counts(frame):
    """A1's 10.25 was recorded at 09:00:00.000. recorded_at <= as_of, so it is in."""
    assert row(frame, "A1", "2026-03-02T09:00:00Z")["amount_sum_24h"] == pytest.approx(85.75)


def test_an_empty_window_is_zero_for_a_count_and_unknown_for_a_mean(frame):
    """A3 has one transaction and it is older than every window here."""
    a3 = row(frame, "A3", "2026-03-02T09:00:00Z")
    assert a3["txn_count_24h"] == 0
    assert a3["amount_sum_24h"] == 0.0
    assert a3["distinct_countries_7d"] == 0
    assert pd.isna(a3["amount_mean_7d"])
    assert pd.isna(a3["max_amount_7d"])


def test_unknown_stays_unknown_rather_than_becoming_a_nan_shaped_zero(frame):
    assert str(frame["amount_mean_7d"].dtype) == "Float64"
    assert str(frame["txn_count_24h"].dtype) == "Int64"
    assert frame["amount_mean_7d"].isna().sum() == 1


def test_a_subset_of_features_can_be_computed_on_its_own(
    account_view, account_events, account_spine
):
    one = compute_training_frame(
        account_view,
        account_events,
        account_spine,
        features=(account_view.feature("txn_count_24h"),),
    )
    assert "amount_sum_24h" not in one.columns
    assert one["txn_count_24h"].tolist() == [2, 3, 0, 3]


def test_a_feature_from_another_view_is_refused(account_view, account_events, account_spine):
    with pytest.raises(KeyError, match="has no feature"):
        compute_training_frame(
            account_view,
            account_events,
            account_spine,
            features=(
                account_view.feature("txn_count_24h").model_copy(update={"name": "borrowed"}),
            ),
        )


def test_a_spine_without_a_decision_time_is_refused(
    account_view, account_events, account_spine
):
    with pytest.raises(ValueError, match="missing column"):
        compute_training_frame(
            account_view, account_events, account_spine.drop(columns=[AS_OF])
        )


def test_an_events_file_missing_an_aggregated_column_is_refused(account_view, tmp_path):
    thin = tmp_path / "events.csv"
    thin.write_text(
        "account_id,event_at,recorded_at,amount\nA1,2026-03-02T08:00:00Z,2026-03-02T08:00:00Z,1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="country"):
        load_events(str(thin), account_view)


def test_comment_lines_in_a_csv_are_not_rows(account_view, account_events, account_spine):
    assert len(account_events) == 11
    assert len(account_spine) == 4
    assert str(account_events["amount"].dtype).startswith("float")
    assert load_events(EVENTS, account_view)["country"].tolist()[0] == "PK"


def test_the_merchant_view_reads_the_same_events(
    merchant_view, merchant_events, merchant_spine
):
    frame = compute_training_frame(merchant_view, merchant_events, merchant_spine)
    assert frame["merchant_txn_count_1h"].tolist() == [2, 1, 0]
    assert frame["merchant_amount_mean_1h"].iloc[0] == pytest.approx(45.0)
    assert frame["merchant_amount_mean_1h"].iloc[1] == pytest.approx(10.25)
    assert pd.isna(frame["merchant_amount_mean_1h"].iloc[2])
