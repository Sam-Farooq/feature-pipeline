"""The clock a row's freshness is measured on.

Freshness used to be the reader's `now` minus the writer's `as_of`, which
trusted a timestamp from a machine the reader never talks to. A writer an hour
fast stamped every row an hour in the future, so a row computed at 09:00 and
read at 09:25 came back FRESH against a twenty minute budget, with the age
reported as -2100 seconds and no counter moving. A slow writer fails the other
way and triggers a recompute that was not needed.

So a row is dated by a clock the reader and the writer both reach: the store's
own. Redis answers `TIME`, the write stamp and the comparison instant both come
from there, and neither machine's clock enters the arithmetic. `as_of` is still
written and still means what it meant, the instant the features were computed
over, which is what the batch path joins on and what a backfill matches.

`ManualClock` is here because the demo and the tests have to move time without
waiting for it, and because `featpipe serve --at` lets an operator name the
instant a row is judged against.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Protocol


class Clock(Protocol):
    """Whatever the store asks for the time."""

    def now(self) -> datetime: ...


class RedisClock:
    """The store's own clock, read with TIME. One round trip per reading."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def now(self) -> datetime:
        seconds, microseconds = self._client.time()
        return datetime.fromtimestamp(int(seconds) + int(microseconds) / 1_000_000, tz=UTC)


class ManualClock:
    """A clock that only moves when it is told to.

    `advance` takes negative seconds too, which is how a store clock stepping
    backwards gets tested without waiting for an NTP correction to happen.
    """

    def __init__(self, at: datetime) -> None:
        self._at = at

    def now(self) -> datetime:
        return self._at

    def advance(self, seconds: float) -> datetime:
        self._at += timedelta(seconds=seconds)
        return self._at
