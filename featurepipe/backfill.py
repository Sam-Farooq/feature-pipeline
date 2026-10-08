"""Adding one feature to a store and to a training frame.

Both halves rest on the same property: a feature's identity lives in the hash
field name and a row's identity lives in the key, so a new feature is a new
field on rows that already exist. Nothing is recomputed, nothing is rewritten
and the ttl does not move.

The online half computes the new value with the serving path's own
accumulator rather than with the batch path, because a backfilled value that
came from the other implementation would be a value the parity check never
looked at.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from featurepipe.offline.frames import AS_OF, compute_training_frame
from featurepipe.online.materialize import accumulator_as_of
from featurepipe.online.store import FeatureStore
from featurepipe.spec import FeatureSpec, FeatureView


@dataclass(frozen=True)
class BackfillReport:
    view: str
    feature: str
    as_of: datetime
    written: int = 0
    absent: int = 0
    as_of_mismatch: int = 0
    unreadable_as_of: int = 0
    no_history: int = 0

    @property
    def considered(self) -> int:
        return (
            self.written
            + self.absent
            + self.as_of_mismatch
            + self.unreadable_as_of
            + self.no_history
        )


def backfill_online(
    view: FeatureView,
    spec: FeatureSpec,
    store: FeatureStore,
    events: pd.DataFrame,
    as_of: datetime,
    entity_ids: Iterable[str] | None = None,
) -> BackfillReport:
    """Compute one feature and add it to the rows already in the store."""
    narrowed = view.with_features((spec,))
    accumulator = accumulator_as_of(narrowed, events, as_of)
    accumulator.evict(as_of)
    known = set(accumulator.entities())
    targets = sorted(known) if entity_ids is None else [str(entity) for entity in entity_ids]

    counts = {
        "written": 0,
        "absent": 0,
        "as_of_mismatch": 0,
        "unreadable_as_of": 0,
        "no_history": 0,
    }
    for entity_id in targets:
        if entity_id not in known:
            counts["no_history"] += 1
            continue
        value = accumulator.values_for(entity_id, as_of)[spec.name]
        write = store.write_field(view, spec, entity_id, value, as_of)
        counts[write.outcome] += 1
    return BackfillReport(view=view.name, feature=spec.name, as_of=as_of, **counts)


def backfill_offline(
    view: FeatureView,
    spec: FeatureSpec,
    events: pd.DataFrame,
    training_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Add one column to a training frame, leaving every other column alone."""
    if spec.name in training_frame.columns:
        raise ValueError(f"{spec.name} is already a column of this frame")
    spine = training_frame[[view.entity, AS_OF]]
    computed = compute_training_frame(view.with_features((spec,)), events, spine)
    column = computed[spec.name]
    column.index = training_frame.index
    out = training_frame.copy()
    out[spec.name] = column
    return out
