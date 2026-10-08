"""Durations and window bounds.

The only module both paths import that has no pandas and no redis in it. If
the batch path and the serving path ever disagree about whether an event is
inside a window, they disagree here, so there is one function and both call
it.
"""

from __future__ import annotations

from datetime import timedelta
from typing import TypeVar

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}

T = TypeVar("T")


def parse_duration(text: str) -> int:
    """Seconds in a duration written as 30s, 15m, 24h or 7d."""
    if not isinstance(text, str) or len(text) < 2:
        raise ValueError(f"{text!r} is not a duration; write it as 30s, 15m, 24h or 7d")
    digits, unit = text[:-1], text[-1]
    if unit not in _UNITS or not digits.isdigit():
        raise ValueError(f"{text!r} is not a duration; write it as 30s, 15m, 24h or 7d")
    seconds = int(digits) * _UNITS[unit]
    if seconds <= 0:
        raise ValueError(f"{text!r} is a zero-length window")
    return seconds


def format_duration(seconds: int) -> str:
    """The shortest exact spelling of a whole number of seconds."""
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if seconds % size == 0:
            return f"{seconds // size}{unit}"
    return f"{seconds}s"


def window_bounds(as_of: T, window_seconds: int) -> tuple[T, T]:
    """The (lower, upper) edges of the window ending at as_of.

    Works on a single timestamp and on a pandas Series of them, because the
    batch path needs one bound per spine row and the serving path needs one.
    """
    return as_of - timedelta(seconds=window_seconds), as_of


def within_window(event_at, as_of, window_seconds: int):
    """Is event_at inside the window ending at as_of?

    Lower edge exclusive, upper edge inclusive. An event stamped exactly
    as_of counts; an event stamped exactly as_of minus the window does not.
    One rule, so a 24h window can never hold 24h and one event.

    Scalars in, bool out. Series in, boolean Series out.
    """
    lower, upper = window_bounds(as_of, window_seconds)
    return (event_at > lower) & (event_at <= upper)
