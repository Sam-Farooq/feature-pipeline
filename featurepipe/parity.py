"""Running one definition down both paths and comparing the answers.

This is the only reason to believe the two implementations agree. Everything
else in the repo is one path or the other.

For each spine row the check computes the batch value, builds the serving
path's accumulator out of the events recorded by that instant, writes the
vector, and reads it back through the real serving code. The comparison is
therefore against what a caller would actually be handed, encoding and
freshness included, not against an in-memory dict.

It is quadratic in the spine: every row rebuilds the stream from the start.
That is affordable because this is a check and not a job.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd

from featurepipe.encoding import Value
from featurepipe.offline.frames import AS_OF, compute_training_frame
from featurepipe.online.materialize import accumulator_as_of
from featurepipe.online.store import FeatureStore, Status
from featurepipe.spec import FeatureSpec, FeatureView

REL_TOL = 1e-9


@dataclass(frozen=True)
class Disagreement:
    view: str
    entity_id: str
    as_of: datetime
    feature: str
    offline: Value
    online: Value
    note: str = ""

    def __str__(self) -> str:
        where = f"{self.view}/{self.entity_id} at {self.as_of:%Y-%m-%dT%H:%M:%SZ}"
        line = f"{where} {self.feature}: batch {self.offline!r}, serving {self.online!r}"
        return f"{line} ({self.note})" if self.note else line


@dataclass(frozen=True)
class ParityReport:
    view: str
    rows: int
    comparisons: int
    disagreements: tuple[Disagreement, ...]

    @property
    def agreed(self) -> bool:
        return not self.disagreements


def _expected(raw: Any, spec: FeatureSpec) -> Value:
    if raw is None or pd.isna(raw):
        return None
    return int(raw) if spec.dtype == "int" else float(raw)


def _matches(offline: Value, online: Value, spec: FeatureSpec, rel_tol: float) -> bool:
    if offline is None or online is None:
        return offline is None and online is None
    if spec.dtype == "int":
        return int(offline) == int(online)
    return math.isclose(float(offline), float(online), rel_tol=rel_tol, abs_tol=0.0)


def check_parity(
    view: FeatureView,
    events: pd.DataFrame,
    spine: pd.DataFrame,
    client: Any,
    rel_tol: float = REL_TOL,
) -> ParityReport:
    """Compare the batch answer and the served answer for every spine row."""
    batch = compute_training_frame(view, events, spine)
    store = FeatureStore(client, namespace="parity")
    disagreements: list[Disagreement] = []
    comparisons = 0

    for row in batch.to_dict("records"):
        as_of = row[AS_OF].to_pydatetime()
        entity_id = str(row[view.entity])
        accumulator = accumulator_as_of(view, events, as_of)
        accumulator.evict(as_of)

        if entity_id not in accumulator.entities():
            disagreements.append(
                Disagreement(
                    view=view.name,
                    entity_id=entity_id,
                    as_of=as_of,
                    feature="*",
                    offline="a row",
                    online=None,
                    note="the stream has never seen this entity, so serving can only "
                    "answer MISS while the batch path filled its empty windows",
                )
            )
            continue

        store.write_vector(view, entity_id, accumulator.values_for(entity_id, as_of), as_of)
        served = store.read_vector(view, entity_id, as_of)
        if served.status is not Status.FRESH:
            disagreements.append(
                Disagreement(
                    view=view.name,
                    entity_id=entity_id,
                    as_of=as_of,
                    feature="*",
                    offline="a fresh row",
                    online=str(served.status),
                    note=served.reason or "",
                )
            )
            continue

        for spec in view.features:
            comparisons += 1
            offline = _expected(row[spec.name], spec)
            online = served.values[spec.name]
            if not _matches(offline, online, spec, rel_tol):
                disagreements.append(
                    Disagreement(
                        view=view.name,
                        entity_id=entity_id,
                        as_of=as_of,
                        feature=spec.name,
                        offline=offline,
                        online=online,
                    )
                )

    return ParityReport(
        view=view.name,
        rows=len(batch),
        comparisons=comparisons,
        disagreements=tuple(disagreements),
    )
