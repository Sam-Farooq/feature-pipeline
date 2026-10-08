"""One definition, two implementations, the same answers.

The batch path is a vectorised pandas join over a whole spine. The serving
path is a row-at-a-time accumulator with no pandas in it. Comparing an
implementation against itself would prove nothing, so the only thing worth
testing is whether these two agree.
"""

from __future__ import annotations

from featurepipe.parity import check_parity


def test_the_merchant_view_agrees_down_both_paths(
    merchant_view, merchant_events, merchant_spine, client
):
    report = check_parity(merchant_view, merchant_events, merchant_spine, client=client)
    assert report.disagreements == ()
    assert report.rows == 3
    assert report.comparisons == 6
    assert report.agreed


def test_a_serving_path_that_computes_something_else_is_caught(
    monkeypatch, merchant_view, merchant_events, merchant_spine, client
):
    """A check that cannot fail is worth as little as no check."""
    monkeypatch.setattr(
        "featurepipe.online.accumulator.apply_aggregation",
        lambda spec, rows: 7 if spec.dtype == "int" else 7.0,
    )
    report = check_parity(merchant_view, merchant_events, merchant_spine, client=client)
    assert not report.agreed
    assert "merchant_txn_count_1h" in {item.feature for item in report.disagreements}
