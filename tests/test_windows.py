from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from featurepipe.windows import format_duration, parse_duration, window_bounds, within_window

AS_OF = datetime(2026, 3, 2, 9, 0, tzinfo=UTC)
DAY = 86400


@pytest.mark.parametrize(
    ("text", "seconds"),
    [("30s", 30), ("15m", 900), ("1h", 3600), ("24h", DAY), ("7d", 7 * DAY)],
)
def test_durations_parse(text, seconds):
    assert parse_duration(text) == seconds


@pytest.mark.parametrize("text", ["", "h", "24", "24x", "1.5h", "-1h", "0h", "7 d", None])
def test_junk_is_refused(text):
    with pytest.raises(ValueError):
        parse_duration(text)


@pytest.mark.parametrize(("seconds", "text"), [(7 * DAY, "7d"), (3600, "1h"), (90, "90s")])
def test_durations_print_back(seconds, text):
    assert format_duration(seconds) == text


def test_the_lower_edge_is_exclusive_and_the_upper_edge_is_not():
    lower, upper = window_bounds(AS_OF, DAY)
    assert lower == AS_OF - timedelta(days=1)
    assert upper == AS_OF
    assert within_window(upper, AS_OF, DAY) is True
    assert within_window(lower, AS_OF, DAY) is False


def test_a_24h_window_cannot_hold_24h_and_one_event():
    """Both edges inclusive would make the window one event longer than it says."""
    inside = AS_OF - timedelta(days=1) + timedelta(microseconds=1)
    assert within_window(inside, AS_OF, DAY) is True
    assert within_window(AS_OF + timedelta(microseconds=1), AS_OF, DAY) is False
