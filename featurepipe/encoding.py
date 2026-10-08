"""Turning a feature value into a Redis string and back.

Redis hash fields are bytes, so every value becomes text. Two rules:

* a computed unknown is stored as a sentinel, never as an empty string and
  never omitted, so it round trips as None rather than as a miss. A count of
  0 and an unknown count are different answers and they stay different.
* floats go through repr, which round trips exactly on CPython. A served
  value is the computed value, not a shortened one.
"""

from __future__ import annotations

from featurepipe.spec import FeatureSpec

UNKNOWN = "\x00unknown"
AS_OF_FIELD = "_as_of"

Value = int | float | None


def as_text(raw: str | bytes | None) -> str | None:
    if raw is None:
        return None
    return raw.decode("utf-8") if isinstance(raw, bytes) else raw


def encode(value: Value, spec: FeatureSpec) -> str:
    if value is None:
        return UNKNOWN
    if spec.dtype == "int":
        return str(int(value))
    return repr(float(value))


def decode(raw: str | bytes | None, spec: FeatureSpec) -> Value:
    text = as_text(raw)
    if text is None or text == UNKNOWN:
        return None
    return int(text) if spec.dtype == "int" else float(text)
