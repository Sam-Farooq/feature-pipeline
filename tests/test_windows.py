from __future__ import annotations

import pytest

from featurepipe.windows import format_duration, parse_duration

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
