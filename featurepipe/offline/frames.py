"""Point-in-time aggregation over a frame of history.

Two timestamps per event and both of them matter. event_at says when it
happened; recorded_at says when this pipeline could first have known about
it. A training row at 09:00 is allowed to see an event that happened at 03:00
and was recorded at 03:01. It is not allowed to see one that happened at
22:40 the night before and was recorded at 11:15, because the model that uses
this feature will not have it at 09:00 either.

Filtering on event_at alone is the single commonest way a feature table leaks
the future into the training set, and it leaks silently: the offline numbers
get better and production does not move.
"""

from __future__ import annotations

import pandas as pd

from featurepipe.spec import FeatureSpec, FeatureView
from featurepipe.windows import within_window

AS_OF = "as_of"

_NUMERIC_AGGS = ("sum", "mean", "max")


def _numeric_columns(view: FeatureView) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            spec.column
            for spec in view.features
            if spec.column is not None and spec.agg in _NUMERIC_AGGS
        )
    )


def _require(frame: pd.DataFrame, columns: tuple[str, ...], path: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing column(s): {', '.join(missing)}")


def load_events(path: str, view: FeatureView) -> pd.DataFrame:
    """Read an event history. Lines starting with # are comments, not data."""
    raw = pd.read_csv(path, comment="#", dtype=str)
    needed = (view.entity, view.event_time, view.recorded_time, *view.columns)
    _require(raw, needed, path)
    frame = pd.DataFrame({view.entity: raw[view.entity].astype(str)})
    for column in (view.event_time, view.recorded_time):
        frame[column] = pd.to_datetime(raw[column], utc=True, format="ISO8601")
    numeric = _numeric_columns(view)
    for column in view.columns:
        frame[column] = (
            pd.to_numeric(raw[column]) if column in numeric else raw[column].astype(str)
        )
    return frame


def load_spine(path: str, view: FeatureView) -> pd.DataFrame:
    """Read the (entity, decision time) pairs a training frame is built on."""
    raw = pd.read_csv(path, comment="#", dtype=str)
    _require(raw, (view.entity, AS_OF), path)
    frame = raw.copy()
    frame[view.entity] = frame[view.entity].astype(str)
    frame[AS_OF] = pd.to_datetime(frame[AS_OF], utc=True, format="ISO8601")
    return frame


def _aggregate(rows: pd.DataFrame, spec: FeatureSpec) -> pd.Series:
    grouped = rows.groupby("_row", sort=True)
    if spec.agg == "count":
        return grouped.size()
    column = grouped[spec.column]
    if spec.agg == "sum":
        return column.sum()
    if spec.agg == "mean":
        return column.mean()
    if spec.agg == "max":
        return column.max()
    return column.nunique()


def compute_training_frame(
    view: FeatureView,
    events: pd.DataFrame,
    spine: pd.DataFrame,
    features: tuple[FeatureSpec, ...] | None = None,
) -> pd.DataFrame:
    """One row per spine row, one column per feature, plus the spine's own.

    Pass `features` to compute a subset, which is what a backfill of a single
    new column does rather than rebuilding the whole table.
    """
    specs = view.features if features is None else tuple(features)
    for spec in specs:
        view.feature(spec.name)
    _require(spine, (view.entity, AS_OF), "the spine")

    spine = spine.reset_index(drop=True)
    keys = spine[[view.entity, AS_OF]].copy()
    keys["_row"] = pd.RangeIndex(len(spine))

    pairs = keys.merge(events, on=view.entity, how="left")
    known = pairs[view.recorded_time] <= pairs[AS_OF]

    out = spine.copy()
    for spec in specs:
        inside = within_window(pairs[view.event_time], pairs[AS_OF], spec.window_seconds)
        rows = pairs[known & inside]
        series = _aggregate(rows, spec) if len(rows) else pd.Series(dtype="float64")
        series = series.reindex(keys["_row"]).astype("float64")
        if spec.empty_value is not None:
            series = series.fillna(spec.empty_value)
        out[spec.name] = series.to_numpy()
        out[spec.name] = out[spec.name].astype("Int64" if spec.dtype == "int" else "Float64")
    return out
