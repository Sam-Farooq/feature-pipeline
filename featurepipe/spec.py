"""What a feature is, written down once.

Two hashes come out of this module and they are not the same thing:

* a view version, over the fields that decide what a stored row *is*: the
  entity column and the two time columns. Change one and every stored value
  is suspect, so the version goes in the Redis key and the old keyspace ages
  out on its own ttl.
* a feature version, over one feature's aggregation, column, window and empty
  policy. It goes in the hash *field* name. Narrowing a window renames one
  field, so serving misses it instead of answering with the old number, and
  the other features on the same row are untouched.

Putting the feature set in the key instead would mean adding a feature
invalidates every row, which is the thing a backfill exists to avoid.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from featurepipe.windows import parse_duration

AggKind = Literal["count", "sum", "mean", "max", "distinct_count"]
EmptyWindow = Literal["zero", "unknown"]

_NEEDS_COLUMN = ("sum", "mean", "max", "distinct_count")
_INT_AGGS = ("count", "distinct_count")
_NO_ZERO = ("mean", "max")


def _digest(payload: dict[str, Any], length: int) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()[:length]


class FeatureSpec(BaseModel):
    """One number, and the window it is computed over."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    agg: AggKind
    window: str
    dtype: Literal["int", "float"]
    empty_window: EmptyWindow
    column: str | None = None

    @field_validator("name")
    @classmethod
    def _usable_as_a_column(cls, value: str) -> str:
        if not value.isidentifier():
            raise ValueError(f"feature name {value!r} is not usable as a column name")
        return value

    @field_validator("window")
    @classmethod
    def _window_parses(cls, value: str) -> str:
        parse_duration(value)
        return value

    @model_validator(mode="after")
    def _the_parts_agree(self) -> FeatureSpec:
        if self.agg == "count" and self.column is not None:
            raise ValueError(f"{self.name}: count takes no column, got {self.column!r}")
        if self.agg in _NEEDS_COLUMN and self.column is None:
            raise ValueError(f"{self.name}: {self.agg} needs a column to aggregate")
        if self.agg in _INT_AGGS and self.dtype != "int":
            raise ValueError(f"{self.name}: a {self.agg} is an int, not a {self.dtype}")
        if self.agg == "mean" and self.dtype != "float":
            raise ValueError(f"{self.name}: a mean is a float, not an {self.dtype}")
        if self.agg in _NO_ZERO and self.empty_window == "zero":
            raise ValueError(
                f"{self.name}: there is no {self.agg} of nothing, "
                'so empty_window has to be "unknown"'
            )
        return self

    @property
    def window_seconds(self) -> int:
        return parse_duration(self.window)

    @property
    def empty_value(self) -> int | float | None:
        """What a computation over an empty window returns.

        This is not a default for a cache miss. It means the window was looked
        at and held nothing, which for a count is 0 and for a mean is unknown.
        Serving never substitutes it.
        """
        if self.empty_window == "unknown":
            return None
        return 0 if self.dtype == "int" else 0.0

    @property
    def version(self) -> str:
        return _digest(
            {
                "agg": self.agg,
                "column": self.column,
                "dtype": self.dtype,
                "empty_window": self.empty_window,
                "name": self.name,
                "window_seconds": self.window_seconds,
            },
            8,
        )

    @property
    def field(self) -> str:
        """The Redis hash field this feature is stored under."""
        return f"{self.name}@{self.version}"


class FeatureView(BaseModel):
    """A set of features over one entity, materialised on one schedule."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    entity: str
    event_time: str
    recorded_time: str
    ttl: str
    materialize_every: str
    max_staleness: str
    features: tuple[FeatureSpec, ...]

    @field_validator("features")
    @classmethod
    def _at_least_one_and_no_repeats(cls, value: tuple[FeatureSpec, ...]):
        if not value:
            raise ValueError("a view with no features has nothing to serve")
        names = [spec.name for spec in value]
        repeats = sorted({name for name in names if names.count(name) > 1})
        if repeats:
            raise ValueError(f"features declared twice: {', '.join(repeats)}")
        return value

    @model_validator(mode="after")
    def _ttl_outlives_a_cycle(self) -> FeatureView:
        for field in ("ttl", "materialize_every", "max_staleness"):
            parse_duration(getattr(self, field))
        floor = self.materialize_every_seconds + self.max_staleness_seconds
        if self.ttl_seconds < floor:
            raise ValueError(
                f"{self.name}: ttl {self.ttl} is shorter than one materialisation "
                f"cycle plus the staleness allowance ({self.materialize_every} + "
                f"{self.max_staleness}), so keys expire between runs and every read "
                "becomes a miss"
            )
        return self

    @property
    def ttl_seconds(self) -> int:
        return parse_duration(self.ttl)

    @property
    def materialize_every_seconds(self) -> int:
        return parse_duration(self.materialize_every)

    @property
    def max_staleness_seconds(self) -> int:
        return parse_duration(self.max_staleness)

    @property
    def longest_window_seconds(self) -> int:
        return max(spec.window_seconds for spec in self.features)

    @property
    def columns(self) -> tuple[str, ...]:
        """Event columns the aggregations read, deduplicated, in order."""
        seen: list[str] = []
        for spec in self.features:
            if spec.column is not None and spec.column not in seen:
                seen.append(spec.column)
        return tuple(seen)

    @property
    def version(self) -> str:
        return _digest(
            {
                "entity": self.entity,
                "event_time": self.event_time,
                "name": self.name,
                "recorded_time": self.recorded_time,
            },
            10,
        )

    def feature(self, name: str) -> FeatureSpec:
        for spec in self.features:
            if spec.name == name:
                return spec
        known = ", ".join(spec.name for spec in self.features)
        raise KeyError(f"{self.name} has no feature {name!r}; it has {known}")

    def with_features(self, features: tuple[FeatureSpec, ...]) -> FeatureView:
        """The same view over a different feature set, revalidated."""
        return FeatureView.model_validate({**self.model_dump(), "features": features})
