"""The serving path's own computation.

This is deliberately not the batch path's code. A feature store that computes
the online value by calling the offline function has not proved anything: the
whole risk is that two implementations of one definition drift apart, and you
cannot test for drift between a thing and itself.

So this is what a stream consumer would be. Events arrive in the order they
are recorded, each one is inserted into its entity's window, and a vector is
read off the retained events at materialisation time. No pandas in here.

Memory: one entry per entity ever seen, holding the events of the longest
window in the view. The entity entry outlives its events on purpose. An
account whose last transaction aged out still has a 24h count, and that count
is 0, which is a different statement from "we have no idea". Dropping the
entity would make serving answer MISS where the batch path answers 0, and the
parity check is how that was found.
"""

from __future__ import annotations

import bisect
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

from featurepipe.spec import FeatureSpec, FeatureView
from featurepipe.windows import within_window

Value = int | float | None


def apply_aggregation(spec: FeatureSpec, rows: list[Mapping[str, Any]]) -> Value:
    if not rows:
        return spec.empty_value
    if spec.agg == "count":
        return len(rows)
    column = [row[spec.column] for row in rows]
    if spec.agg == "sum":
        return float(sum(column))
    if spec.agg == "mean":
        return float(sum(column)) / len(column)
    if spec.agg == "max":
        return float(max(column))
    return len(set(column))


class Accumulator:
    """Per-entity event windows, fed in the order events are recorded."""

    def __init__(self, view: FeatureView) -> None:
        self._view = view
        self._events: dict[str, list[tuple[datetime, int, Mapping[str, Any]]]] = {}
        self._sequence = 0
        self.watermark: datetime | None = None
        self.ingested = 0
        self.dropped_too_old = 0
        self.out_of_order = 0

    def ingest(
        self,
        entity_id: str,
        event_at: datetime,
        recorded_at: datetime,
        values: Mapping[str, Any],
    ) -> bool:
        """Take one event. False means it was dropped as unusable.

        An event older than the longest window measured from the watermark can
        never appear in a vector, so it is counted and discarded rather than
        stored. Because the watermark never runs ahead of the as_of a vector is
        built at, nothing droppable here was going to be in that vector.
        """
        if self.watermark is None or recorded_at > self.watermark:
            self.watermark = recorded_at
        elif recorded_at < self.watermark:
            self.out_of_order += 1

        horizon = self.watermark - timedelta(seconds=self._view.longest_window_seconds)
        bucket = self._events.setdefault(entity_id, [])
        if event_at <= horizon:
            self.dropped_too_old += 1
            return False

        self._sequence += 1
        bisect.insort(bucket, (event_at, self._sequence, dict(values)))
        self.ingested += 1
        return True

    def evict(self, as_of: datetime) -> int:
        """Drop events that no window ending at as_of can reach."""
        horizon = as_of - timedelta(seconds=self._view.longest_window_seconds)
        dropped = 0
        for entity_id, bucket in self._events.items():
            cut = bisect.bisect_right(bucket, (horizon, self._sequence + 1, {}))
            if cut:
                dropped += cut
                self._events[entity_id] = bucket[cut:]
        return dropped

    def entities(self) -> tuple[str, ...]:
        return tuple(sorted(self._events))

    def retained(self, entity_id: str) -> int:
        return len(self._events.get(entity_id, ()))

    def values_for(self, entity_id: str, as_of: datetime) -> dict[str, Value]:
        """The vector for one entity as of one instant."""
        if entity_id not in self._events:
            raise KeyError(f"{self._view.name}: nothing ingested for entity {entity_id!r}")
        bucket = self._events[entity_id]
        out: dict[str, Value] = {}
        for spec in self._view.features:
            rows = [
                row
                for event_at, _, row in bucket
                if within_window(event_at, as_of, spec.window_seconds)
            ]
            out[spec.name] = apply_aggregation(spec, rows)
        return out
