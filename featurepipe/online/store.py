"""Redis as the serving store, and the three answers it can give.

FRESH, STALE and MISS are different, and collapsing any two of them is how
training/serving skew gets into production without anyone seeing it:

* MISS is "no value". It is never filled in with the empty-window value. A
  key that expired looks exactly like an account with no transactions if the
  read path answers 0 for both, and the model cannot tell them apart either.
* STALE is "a value that cannot be called current", which is either a value
  computed too long ago or one whose age two clocks disagree about. The
  numbers come back with the age attached and the caller decides, because a 40
  second old count is fine for one model and not for another.
* a partly written row is a MISS, not a row with holes in it.

One row has one as_of. Every field in a key was computed at the same instant
or the write is refused, so there is one number to compare against the
staleness budget. Per-field timestamps would allow a mixed-freshness row;
being able to say when a row is from was worth more.

A row carries two timestamps and they answer different questions. `as_of` is
the instant the features were computed over: it is event time, the batch path
joins on it, and a backfill matches against it. `_written_at` is when the store
accepted the row, read from the store's own clock, and it is the only one
freshness is measured against. Dating a row by `as_of` trusts the clock of a
machine the reader never talks to, and a writer running an hour fast stamps
rows that read as fresh an hour past their budget with no counter moving.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from featurepipe.clock import Clock, RedisClock
from featurepipe.encoding import (
    AS_OF_FIELD,
    WRITTEN_AT_FIELD,
    Value,
    as_text,
    decode,
    decode_instant,
    encode,
    encode_instant,
)
from featurepipe.spec import FeatureSpec, FeatureView

MAX_SKEW_SECONDS = 5.0


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
    written_at: datetime | None = None
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


@dataclass(frozen=True)
class FieldWrite:
    key: str
    outcome: str
    stored_as_of: datetime | None = None


class FeatureStore:
    """Reads and writes one view's rows. Takes any redis-py compatible client.

    `clock` is where both the write stamp and the freshness comparison come
    from, and it defaults to the store's own clock so that reader and writer
    share it. `max_skew_seconds` is how far the two readings of that clock may
    disagree before a read says so instead of answering.
    """

    def __init__(
        self,
        client: Any,
        namespace: str = "fp",
        clock: Clock | None = None,
        max_skew_seconds: float = MAX_SKEW_SECONDS,
    ) -> None:
        self._client = client
        self._namespace = namespace
        self._clock = clock if clock is not None else RedisClock(client)
        self._max_skew_seconds = max_skew_seconds

    def key(self, view: FeatureView, entity_id: str) -> str:
        return f"{self._namespace}:{view.name}:{view.version}:{entity_id}"

    def write_vector(
        self,
        view: FeatureView,
        entity_id: str,
        values: Mapping[str, Value],
        as_of: datetime,
    ) -> str:
        """Write a whole row, stamp it twice, and give it its ttl.

        HSET then EXPIRE as two round trips would leave a window where the row
        exists with no ttl, which is a row that never expires. One MULTI.

        Reading the store clock first is one round trip more than this used to
        cost. It buys a row that can be dated without trusting the machine that
        wrote it, and TIME cannot go inside the MULTI because its answer is
        needed to build the HSET.
        """
        mapping = {
            spec.field: encode(values[spec.name], spec)
            for spec in view.features
            if spec.name in values
        }
        if not mapping:
            raise ValueError(f"{view.name}: nothing to write for entity {entity_id!r}")
        mapping[AS_OF_FIELD] = encode_instant(as_of)
        mapping[WRITTEN_AT_FIELD] = encode_instant(self._clock.now())
        key = self.key(view, entity_id)
        pipe = self._client.pipeline(transaction=True)
        pipe.hset(key, mapping=mapping)
        pipe.expire(key, view.ttl_seconds)
        pipe.execute()
        return key

    def write_field(
        self,
        view: FeatureView,
        spec: FeatureSpec,
        entity_id: str,
        value: Value,
        as_of: datetime,
    ) -> FieldWrite:
        """Add or replace one field on a row that already exists.

        This is the backfill primitive, and it is this shape because of two
        Redis behaviours:

        * HSET on an existing key leaves the other fields and the ttl alone,
          which is exactly what adding a feature should do.
        * HSET on a *missing* key creates it with no ttl at all. A backfill
          that touched an expired entity would leave an immortal row holding
          one feature. So a row that is not there is reported absent and not
          created.

        The EXISTS and the HSET are two round trips, so the key can expire in
        between. The ttl is read back after the write and a key that came out
        persistent is deleted, which turns a rare lost row into a miss rather
        than into a row that never expires and never updates.

        `_written_at` is left alone. The row's values were all computed at one
        as_of and the row is as old as that write, so a backfill that moved the
        stamp would hand a caller an old vector reading as newly stored.
        """
        key = self.key(view, entity_id)
        stored = as_text(self._client.hget(key, AS_OF_FIELD))
        if stored is None:
            return FieldWrite(key, "absent")
        try:
            stored_at = decode_instant(stored)
        except ValueError:
            return FieldWrite(key, "unreadable_as_of")
        if abs(stored_at.timestamp() - as_of.timestamp()) > 1e-6:
            return FieldWrite(key, "as_of_mismatch", stored_as_of=stored_at)
        self._client.hset(key, spec.field, encode(value, spec))
        if self._client.ttl(key) == -1:
            self._client.delete(key)
            return FieldWrite(key, "expired_mid_write")
        return FieldWrite(key, "written", stored_as_of=stored_at)

    def read_vector(self, view: FeatureView, entity_id: str) -> ServedVector:
        """One HGETALL, so a reader cannot see half a row.

        There is no `now` argument. The age is the store's clock minus the
        store's own record of when it took the row, so neither the writer nor
        the reader can pick the clock the answer is judged on. The clock is
        read after the row, which can only make the age larger.
        """
        key = self.key(view, entity_id)
        raw = {as_text(name): payload for name, payload in self._client.hgetall(key).items()}
        if not raw:
            return self._miss(view, entity_id, "no row for this entity at this view version")
        try:
            as_of = decode_instant(raw.get(AS_OF_FIELD))
        except (TypeError, ValueError):
            return self._miss(
                view,
                entity_id,
                f"{AS_OF_FIELD} is absent or unreadable, so the row cannot be dated",
            )
        try:
            written_at = decode_instant(raw.get(WRITTEN_AT_FIELD))
        except (TypeError, ValueError):
            return self._miss(
                view,
                entity_id,
                f"{WRITTEN_AT_FIELD} is absent or unreadable, so the row cannot be dated "
                "on a clock the reader shares with the writer",
            )

        values: dict[str, Value] = {}
        missing: list[str] = []
        for spec in view.features:
            if spec.field in raw:
                values[spec.name] = decode(raw[spec.field], spec)
            else:
                values[spec.name] = None
                missing.append(spec.name)

        age = (self._clock.now() - written_at).total_seconds()
        if missing:
            status = Status.MISS
            reason = "no field at the current definition version for: " + ", ".join(missing)
        elif age < -self._max_skew_seconds:
            status = Status.STALE
            reason = (
                f"the store took this row {-age:.0f}s after the instant it is being judged "
                "at, so two clocks disagree and its age is not knowable"
            )
        elif age > view.max_staleness_seconds:
            status = Status.STALE
            reason = f"stored {age:.0f}s ago, budget is {view.max_staleness_seconds}s"
        else:
            status = Status.FRESH
            reason = None
        return ServedVector(
            view=view.name,
            entity_id=entity_id,
            status=status,
            values=values,
            as_of=as_of,
            written_at=written_at,
            age_seconds=age,
            missing=tuple(missing),
            reason=reason,
        )

    def _miss(self, view: FeatureView, entity_id: str, reason: str) -> ServedVector:
        """No values at all, and the reason. Never the empty-window value."""
        return ServedVector(
            view=view.name,
            entity_id=entity_id,
            status=Status.MISS,
            values={spec.name: None for spec in view.features},
            missing=tuple(spec.name for spec in view.features),
            reason=reason,
        )

    def drop(self, view: FeatureView, entity_id: str) -> int:
        """Remove one row. Rows left by an older view version are not touched."""
        return int(self._client.delete(self.key(view, entity_id)))

    def ttl_of(self, view: FeatureView, entity_id: str) -> int:
        """Seconds left, -1 for no expiry, -2 for no key."""
        return int(self._client.ttl(self.key(view, entity_id)))
