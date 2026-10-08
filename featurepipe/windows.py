"""Durations, written the way a definition writes them."""

from __future__ import annotations

_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


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
