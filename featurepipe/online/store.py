"""Redis as the serving store, and the three answers it can give.

FRESH, STALE and MISS are different, and collapsing any two of them is how
training/serving skew gets into production without anyone seeing it:

* MISS is "no value". It is never filled in with the empty-window value. A
  key that expired looks exactly like an account with no transactions if the
  read path answers 0 for both, and the model cannot tell them apart either.
* STALE is "a value, computed too long ago". The numbers come back with the
  age attached and the caller decides, because a 40 second old count is fine
  for one model and not for another.
* a partly written row is a MISS, not a row with holes in it.

One row has one as_of. Every field in a key was computed at the same instant
or the write is refused, so there is one number to compare against the
staleness budget. Per-field timestamps would allow a mixed-freshness row;
being able to say when a row is from was worth more.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from featurepipe.encoding import AS_OF_FIELD, Value, as_text, decode, encode
from featurepipe.spec import FeatureView


class Status(StrEnum):
    FRESH = "fresh"
    STALE = "stale"
    MISS = "miss"


class StaleFeatures(RuntimeError):
    """A caller asked for features it could use and did not get them."""


@dataclass(frozen=True)
class ServedVector:
    view: str
    entity_id: str
    status: Status
    values: Mapping[str, Value]
    as_of: datetime | None = None
    age_seconds: float | None = None
    missing: tuple[str, ...] = ()
    reason: str | None = None

    @property
    def usable(self) -> bool:
        return self.status is Status.FRESH

    def require(self) -> Mapping[str, Value]:
        if not self.usable:
            raise StaleFeatures(
                f"{self.view}/{self.entity_id} is {self.status}"
                + (f": {self.reason}" if self.reason else "")
            )
        return self.values


class FeatureStore:
    """Reads and writes one view's rows. Takes any redis-py compatible client."""

    def __init__(self, client: Any, namespace: str = "fp") -> None:
        self._client = client
        self._namespace = namespace

    def key(self, view: FeatureView, entity_id: str) -> str:
        return f"{self._namespace}:{view.name}:{view.version}:{entity_id}"

    def write_vector(
        self,
        view: FeatureView,
        entity_id: str,
        values: Mapping[str, Value],
        as_of: datetime,
    ) -> str:
        """Write a whole row and stamp it, in one transaction.

        HSET then EXPIRE as two round trips would leave a window where the row
        exists with no ttl, which is a row that never expires. One MULTI.
        """
        mapping = {
            spec.field: encode(values[spec.name], spec)
            for spec in view.features
            if spec.name in values
        }
        if not mapping:
            raise ValueError(f"{view.name}: nothing to write for entity {entity_id!r}")
        mapping[AS_OF_FIELD] = repr(as_of.timestamp())
        key = self.key(view, entity_id)
        pipe = self._client.pipeline(transaction=True)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, view.ttl_seconds)
        pipe.execute()
        return key

    def read_vector(
        self,
        view: FeatureView,
        entity_id: str,
        now: datetime,
    ) -> ServedVector:
        """One HGETALL, so a reader cannot see half a row."""
        key = self.key(view, entity_id)
        raw = {as_text(name): payload for name, payload in self._client.hgetall(key).items()}
        empty = {spec.name: None for spec in view.features}
        if not raw:
            return ServedVector(
                view=view.name,
                entity_id=entity_id,
                status=Status.MISS,
                values=empty,
                missing=tuple(spec.name for spec in view.features),
                reason="no row for this entity at this view version",
            )
        try:
            as_of = datetime.fromtimestamp(float(as_text(raw.get(AS_OF_FIELD))), tz=UTC)
        except (TypeError, ValueError):
            return ServedVector(
                view=view.name,
                entity_id=entity_id,
                status=Status.MISS,
                values=empty,
                missing=tuple(spec.name for spec in view.features),
                reason=f"{AS_OF_FIELD} is absent or unreadable, so the row cannot be dated",
            )

        values: dict[str, Value] = {}
        missing: list[str] = []
        for spec in view.features:
            if spec.field in raw:
                values[spec.name] = decode(raw[spec.field], spec)
            else:
                values[spec.name] = None
                missing.append(spec.name)

        age = (now - as_of).total_seconds()
        if missing:
            status = Status.MISS
            reason = "no field at the current definition version for: " + ", ".join(missing)
        elif age > view.max_staleness_seconds:
            status = Status.STALE
            reason = f"computed {age:.0f}s ago, budget is {view.max_staleness_seconds}s"
        else:
            status = Status.FRESH
            reason = None
        return ServedVector(
            view=view.name,
            entity_id=entity_id,
            status=status,
            values=values,
            as_of=as_of,
            age_seconds=age,
            missing=tuple(missing),
            reason=reason,
        )

    def drop(self, view: FeatureView, entity_id: str) -> int:
        """Remove one row. Rows left by an older view version are not touched."""
        return int(self._client.delete(self.key(view, entity_id)))

    def ttl_of(self, view: FeatureView, entity_id: str) -> int:
        """Seconds left, -1 for no expiry, -2 for no key."""
        return int(self._client.ttl(self.key(view, entity_id)))
