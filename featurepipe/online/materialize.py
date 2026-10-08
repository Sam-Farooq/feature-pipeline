"""Feeding the accumulator from a frame, and writing what it holds.

In production the accumulator is fed by a consumer and this module does not
exist. Replaying a frame row at a time is how the same code gets exercised in
a test and from the command line, and it is the slowest possible way to do it.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from featurepipe.online.accumulator import Accumulator
from featurepipe.online.store import FeatureStore
from featurepipe.spec import FeatureView


@dataclass(frozen=True)
class MaterializeReport:
    view: str
    as_of: datetime
    rows_written: int
    fields_written: int
    events_ingested: int
    events_dropped_too_old: int
    events_evicted: int


def accumulator_as_of(view: FeatureView, events: pd.DataFrame, as_of: datetime) -> Accumulator:
    """An accumulator holding exactly what the stream had recorded by as_of.

    The recorded_at filter is the whole point. Feeding it the full history
    would let a vector see an event that had not been recorded yet, which is
    the leak the batch path is careful about, reintroduced on the other side.
    """
    accumulator = Accumulator(view)
    visible = events[events[view.recorded_time] <= as_of]
    visible = visible.sort_values(view.recorded_time, kind="stable")
    for row in visible.to_dict("records"):
        accumulator.ingest(
            entity_id=str(row[view.entity]),
            event_at=row[view.event_time].to_pydatetime(),
            recorded_at=row[view.recorded_time].to_pydatetime(),
            values={column: row[column] for column in view.columns},
        )
    return accumulator


def materialize(
    view: FeatureView,
    store: FeatureStore,
    events: pd.DataFrame,
    as_of: datetime,
) -> MaterializeReport:
    """Compute every entity's vector at as_of and write it with the view's ttl."""
    accumulator = accumulator_as_of(view, events, as_of)
    evicted = accumulator.evict(as_of)
    rows = 0
    fields = 0
    for entity_id in accumulator.entities():
        values = accumulator.values_for(entity_id, as_of)
        store.write_vector(view, entity_id, values, as_of)
        rows += 1
        fields += len(values)
    return MaterializeReport(
        view=view.name,
        as_of=as_of,
        rows_written=rows,
        fields_written=fields,
        events_ingested=accumulator.ingested,
        events_dropped_too_old=accumulator.dropped_too_old,
        events_evicted=evicted,
    )
