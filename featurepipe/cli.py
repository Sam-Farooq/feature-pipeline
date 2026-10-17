"""featpipe: inspect the definitions, build a training frame, materialise,
serve, back a feature in, and check that the two paths agree.

Exit codes:
  0  nothing went wrong
  2  refused: a bad definition, a missing file, an unparseable flag
  5  the two paths disagreed
  6  a served row was not usable and --strict was given
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

from featurepipe import __version__
from featurepipe.backfill import backfill_online
from featurepipe.clients import MEMORY, client_for, memory_client
from featurepipe.clock import ManualClock
from featurepipe.offline.frames import compute_training_frame, load_events, load_spine
from featurepipe.online.materialize import materialize
from featurepipe.online.store import FeatureStore
from featurepipe.parity import check_parity
from featurepipe.registry import DefinitionError, Registry
from featurepipe.spec import FeatureSpec, FeatureView
from featurepipe.windows import format_duration

REFUSED = 2
DISAGREED = 5
UNUSABLE = 6

DEFAULT_DEFINITIONS = "definitions"
DEMO_EVENTS = "examples/events.csv"
DEMO_AS_OF = "2026-03-02T09:00:00Z"


def _timestamp(text: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"{text!r} is not an ISO 8601 instant, for example 2026-03-02T09:00:00Z"
        ) from error
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _show(value: object) -> str:
    """A computed unknown. Different from a field that is not there at all."""
    return "unknown" if value is None else str(value)


def _print_defs(registry: Registry) -> None:
    for view in registry.views:
        print(f"{view.name}  key fp:{view.name}:{view.version}:<{view.entity}>")
        print(
            f"  every {format_duration(view.materialize_every_seconds)}"
            f", stale after {format_duration(view.max_staleness_seconds)}"
            f", ttl {format_duration(view.ttl_seconds)}"
        )
        width = max(len(spec.name) for spec in view.features)
        for spec in view.features:
            column = spec.column or "-"
            print(
                f"  {spec.name:<{width}}  {spec.agg:<14} {column:<8}"
                f" {spec.window:>4}  empty={spec.empty_window:<8} field={spec.field}"
            )
        print()


def _print_vector(view: FeatureView, entity_id: str, served) -> None:
    age = "-" if served.age_seconds is None else f"{served.age_seconds:.0f}s"
    print(f"{view.name}/{entity_id}  {served.status.upper()}  age {age}")
    if served.reason:
        print(f"  {served.reason}")
    width = max(len(name) for name in served.values)
    for name, value in served.values.items():
        shown = "no value" if name in served.missing else _show(value)
        print(f"  {name:<{width}}  {shown}")


def _registry(args: argparse.Namespace) -> Registry:
    return Registry.from_directory(args.definitions)


def _cmd_defs(args: argparse.Namespace) -> int:
    _print_defs(_registry(args))
    return 0


def _cmd_offline(args: argparse.Namespace) -> int:
    view = _registry(args).view(args.view)
    events = load_events(args.events, view)
    spine = load_spine(args.spine, view)
    frame = compute_training_frame(view, events, spine)
    print(frame.to_string(index=False))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(f"\n{len(frame)} rows written to {args.out}")
    return 0


def _cmd_parity(args: argparse.Namespace) -> int:
    view = _registry(args).view(args.view)
    events = load_events(args.events, view)
    spine = load_spine(args.spine, view)
    report = check_parity(view, events, spine, client=memory_client())
    print(
        f"{report.view}: {report.rows} spine rows, {report.comparisons} values compared, "
        f"{len(report.disagreements)} disagreements"
    )
    for item in report.disagreements:
        print(f"  {item}")
    return 0 if report.agreed else DISAGREED


def _cmd_materialize(args: argparse.Namespace) -> int:
    view = _registry(args).view(args.view)
    events = load_events(args.events, view)
    store = FeatureStore(client_for(args.redis))
    report = materialize(view, store, events, args.as_of)
    print(
        f"{report.view} at {_stamp(report.as_of)}: {report.rows_written} rows, "
        f"{report.fields_written} fields, {report.events_ingested} events ingested, "
        f"{report.events_dropped_too_old} dropped as too old, "
        f"{report.events_evicted} evicted"
    )
    if args.redis == MEMORY:
        print("the memory client is per process, so nothing survives this command")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    view = _registry(args).view(args.view)
    clock = ManualClock(args.at) if args.at else None
    store = FeatureStore(client_for(args.redis), clock=clock)
    served = store.read_vector(view, args.entity)
    _print_vector(view, args.entity, served)
    if args.strict and not served.usable:
        return UNUSABLE
    return 0


def _cmd_backfill(args: argparse.Namespace) -> int:
    registry = _registry(args)
    view = registry.view(args.view)
    spec = view.feature(args.feature)
    events = load_events(args.events, view)
    store = FeatureStore(client_for(args.redis))
    report = backfill_online(view, spec, store, events, args.as_of)
    print(
        f"{report.view}.{report.feature} at {_stamp(report.as_of)}: "
        f"{report.written} written, {report.absent} rows absent, "
        f"{report.as_of_mismatch} stamped at another instant, "
        f"{report.expired_mid_write} expired mid write"
    )
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    """The whole cycle in one process, against an in-memory Redis."""
    registry = Registry.from_directory(args.definitions)
    view = registry.view("account_activity")
    events = load_events(args.events, view)
    as_of = args.as_of
    client = memory_client()
    store = FeatureStore(client, clock=ManualClock(as_of))

    print(f"1. materialise {view.name} at {_stamp(as_of)}")
    report = materialize(view, store, events, as_of)
    print(
        f"   {report.rows_written} rows, {report.fields_written} fields, "
        f"ttl {format_duration(view.ttl_seconds)}"
    )
    print(
        f"   {report.events_ingested} events recorded by then, "
        f"{report.events_evicted} evicted as older than the "
        f"{format_duration(view.longest_window_seconds)} window"
    )

    print("\n2. a row computed a moment ago")
    _print_vector(view, "A1", store.read_vector(view, "A1"))

    print(f"\n3. the same row read {format_duration(25 * 60)} later, by a reader whose")
    print("   store clock has moved on. Nothing about the row changed")
    later = FeatureStore(client, clock=ManualClock(as_of + timedelta(minutes=25)))
    _print_vector(view, "A1", later.read_vector(view, "A1"))

    print("\n4. an entity the stream has never seen: a miss, not a set of zeros")
    _print_vector(view, "A9", store.read_vector(view, "A9"))

    print("\n5. an entity whose history has all aged out: zeros where zero is the")
    print("   answer, unknown where there is nothing to average")
    _print_vector(view, "A3", store.read_vector(view, "A3"))

    narrowed = view.with_features(
        tuple(
            spec.model_copy(update={"window": "12h"}) if spec.name == "txn_count_24h" else spec
            for spec in view.features
        )
    )
    print("\n6. txn_count_24h redefined over 12h. Its field name moves with its")
    print("   definition, so the old number is not served under the new meaning")
    _print_vector(narrowed, "A1", store.read_vector(narrowed, "A1"))

    added = FeatureSpec(
        name="amount_sum_1h",
        agg="sum",
        column="amount",
        window="1h",
        dtype="float",
        empty_window="zero",
    )
    extended = view.with_features((*view.features, added))
    before = store.ttl_of(view, "A1")
    print(f"\n7. a new feature {added.name} arrives. Before the backfill it is a miss")
    _print_vector(extended, "A1", store.read_vector(extended, "A1"))
    filled = backfill_online(extended, added, store, events, as_of)
    after = store.ttl_of(view, "A1")
    print(
        f"   backfilled: {filled.written} rows written, {filled.absent} absent, "
        f"ttl {before}s before and {after}s after"
    )
    _print_vector(extended, "A1", store.read_vector(extended, "A1"))

    print("\n8. A3's row expires, then the same backfill runs again. HSET on a")
    print("   missing key would create one with no ttl at all, holding a single")
    print("   feature and never expiring, so an absent row is left absent")
    store.drop(view, "A3")
    again = backfill_online(extended, added, store, events, as_of, entity_ids=["A3", "A9"])
    print(
        f"   {again.written} written, {again.absent} absent, "
        f"{again.no_history} unknown to the stream, "
        f"ttl of A3 now {store.ttl_of(view, 'A3')} (-2 is no key)"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="featpipe",
        description="One feature definition, a batch path, a serving path, "
        "and a check that they agree",
    )
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument(
        "--definitions",
        default=DEFAULT_DEFINITIONS,
        help=f"directory of .toml definitions (default: {DEFAULT_DEFINITIONS})",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    defs = subparsers.add_parser("defs", help="print every view, its key and its fields")
    defs.set_defaults(handler=_cmd_defs)

    offline = subparsers.add_parser("offline", help="build a point-in-time training frame")
    offline.add_argument("--view", required=True)
    offline.add_argument("--events", required=True)
    offline.add_argument("--spine", required=True)
    offline.add_argument("--out", help="also write the frame to this csv")
    offline.set_defaults(handler=_cmd_offline)

    parity = subparsers.add_parser("parity", help="compare both paths over a spine")
    parity.add_argument("--view", required=True)
    parity.add_argument("--events", required=True)
    parity.add_argument("--spine", required=True)
    parity.set_defaults(handler=_cmd_parity)

    materialise = subparsers.add_parser("materialize", help="write every entity's row")
    materialise.add_argument("--view", required=True)
    materialise.add_argument("--events", required=True)
    materialise.add_argument("--as-of", required=True, type=_timestamp, dest="as_of")
    materialise.add_argument("--redis", default="redis://localhost:6379/0")
    materialise.set_defaults(handler=_cmd_materialize)

    serve = subparsers.add_parser("serve", help="read one entity's row")
    serve.add_argument("--view", required=True)
    serve.add_argument("--entity", required=True)
    serve.add_argument(
        "--at",
        type=_timestamp,
        help="judge freshness against this instant instead of the store's own clock",
    )
    serve.add_argument("--redis", default="redis://localhost:6379/0")
    serve.add_argument("--strict", action="store_true", help=f"exit {UNUSABLE} unless fresh")
    serve.set_defaults(handler=_cmd_serve)

    backfill = subparsers.add_parser("backfill", help="add one feature to existing rows")
    backfill.add_argument("--view", required=True)
    backfill.add_argument("--feature", required=True)
    backfill.add_argument("--events", required=True)
    backfill.add_argument("--as-of", required=True, type=_timestamp, dest="as_of")
    backfill.add_argument("--redis", default="redis://localhost:6379/0")
    backfill.set_defaults(handler=_cmd_backfill)

    demo = subparsers.add_parser(
        "demo", help="the full cycle in one process, no Redis and no network"
    )
    demo.add_argument("--events", default=DEMO_EVENTS)
    demo.add_argument("--as-of", default=DEMO_AS_OF, type=_timestamp, dest="as_of")
    demo.set_defaults(handler=_cmd_demo)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (DefinitionError, KeyError, ValueError) as error:
        print(f"featpipe: {error}", file=sys.stderr)
        return REFUSED
    except FileNotFoundError as error:
        print(f"featpipe: {error.filename}: no such file", file=sys.stderr)
        return REFUSED


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
