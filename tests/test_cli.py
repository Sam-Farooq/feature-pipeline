"""The commands in the README, and the exit codes a job would branch on."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from featurepipe.cli import main
from tests.conftest import EVENTS, ROOT

ACCOUNTS = str(ROOT / "examples" / "spine_accounts.csv")
MERCHANTS = str(ROOT / "examples" / "spine_merchants.csv")
DEFS = str(ROOT / "definitions")


def run(capsys, *argv):
    """Exit code, stdout and stderr. One readouterr per call, or it empties."""
    code = main(["--definitions", DEFS, *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_defs_prints_both_views_their_keys_and_their_fields(capsys):
    code, out, _ = run(capsys, "defs")
    assert code == 0
    assert "fp:account_activity:" in out
    assert "fp:merchant_activity:" in out
    assert "field=txn_count_24h@" in out
    assert "ttl 36h" in out


def test_offline_prints_a_row_per_spine_row(capsys):
    code, out, _ = run(
        capsys, "offline", "--view", "account_activity", "--events", EVENTS, "--spine", ACCOUNTS
    )
    assert code == 0
    assert out.count("A1") == 2
    assert "85.75" in out
    assert "<NA>" in out


def test_offline_writes_the_frame_where_it_is_told(capsys, tmp_path):
    out_path = tmp_path / "nested" / "training.csv"
    code, out, _ = run(
        capsys,
        "offline",
        "--view",
        "account_activity",
        "--events",
        EVENTS,
        "--spine",
        ACCOUNTS,
        "--out",
        str(out_path),
    )
    assert code == 0
    assert f"4 rows written to {out_path}" in out
    written = pd.read_csv(out_path)
    assert list(written.columns)[-1] == "distinct_countries_7d"
    assert written["txn_count_24h"].tolist() == [2, 3, 0, 3]


@pytest.mark.parametrize(
    ("view", "spine", "values"),
    [("account_activity", ACCOUNTS, 20), ("merchant_activity", MERCHANTS, 6)],
)
def test_parity_exits_zero_on_both_views(capsys, view, spine, values):
    code, out, _ = run(capsys, "parity", "--view", view, "--events", EVENTS, "--spine", spine)
    assert code == 0
    assert f"{values} values compared, 0 disagreements" in out


def test_parity_exits_five_when_the_paths_disagree(capsys, monkeypatch, tmp_path):
    monkeypatch.setattr(
        "featurepipe.online.accumulator.apply_aggregation",
        lambda spec, rows: 7 if spec.dtype == "int" else 7.0,
    )
    code, out, _ = run(
        capsys, "parity", "--view", "account_activity", "--events", EVENTS, "--spine", ACCOUNTS
    )
    assert code == 5
    assert "disagreements" in out
    assert "batch 2, serving 7" in out


def test_the_demo_runs_the_whole_cycle_without_redis(capsys):
    code, out, _ = run(capsys, "demo", "--events", EVENTS)
    assert code == 0
    assert "FRESH" in out and "STALE" in out and "MISS" in out
    assert "ttl 129600s before and 129600s after" in out
    assert "amount_sum_1h          10.25" in out
    assert "-2 is no key" in out


def test_the_demo_ends_by_refusing_a_fast_writers_write(capsys):
    """The last step of the demo is the guard being fed what it exists to
    reject, so a reader of the output sees the refusal and not a claim."""
    code, out, _ = run(capsys, "demo", "--events", EVENTS)
    assert code == 0
    assert "refused: account_activity/A1: as_of is 3600.0s ahead of the store clock" in out


def test_materialising_with_an_as_of_in_the_future_is_refused(capsys):
    """Exit 2, the code for anything refused, rather than a store full of rows
    that read as fresh until the clock is fixed."""
    ahead = datetime.now(tz=UTC) + timedelta(hours=1)
    code, _, err = run(
        capsys,
        "materialize",
        "--view",
        "account_activity",
        "--events",
        EVENTS,
        "--as-of",
        ahead.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "--redis",
        "memory",
    )
    assert code == 2
    assert "ahead of the store clock" in err


def test_serve_reads_nothing_from_an_empty_store(capsys):
    code, out, _ = run(
        capsys,
        "serve",
        "--view",
        "account_activity",
        "--entity",
        "A1",
        "--at",
        "2026-03-02T09:00:00Z",
        "--redis",
        "memory",
    )
    assert code == 0
    assert "MISS" in out


def test_serve_can_be_told_to_fail_on_an_unusable_row(capsys):
    code, out, _ = run(
        capsys,
        "serve",
        "--view",
        "account_activity",
        "--entity",
        "A1",
        "--redis",
        "memory",
        "--strict",
        "--at",
        "2026-03-02T09:00:00Z",
    )
    assert code == 6
    assert "no row for this entity" in out


def test_materialize_against_the_memory_client_says_it_is_throwaway(capsys):
    code, out, _ = run(
        capsys,
        "materialize",
        "--view",
        "account_activity",
        "--events",
        EVENTS,
        "--as-of",
        "2026-03-02T09:00:00Z",
        "--redis",
        "memory",
    )
    assert code == 0
    assert "3 rows, 15 fields" in out
    assert "per process" in out


def test_an_unknown_view_is_refused_with_the_names_that_exist(capsys):
    code, _, err = run(
        capsys, "parity", "--view", "nope", "--events", EVENTS, "--spine", ACCOUNTS
    )
    assert code == 2
    assert "account_activity, merchant_activity" in err


def test_an_unknown_feature_is_refused(capsys):
    code, _, err = run(
        capsys,
        "backfill",
        "--view",
        "account_activity",
        "--feature",
        "nope",
        "--events",
        EVENTS,
        "--as-of",
        "2026-03-02T09:00:00Z",
        "--redis",
        "memory",
    )
    assert code == 2
    assert "has no feature" in err


def test_a_missing_definitions_directory_is_refused(capsys):
    code = main(["--definitions", "/nowhere", "defs"])
    assert code == 2
    assert "not a directory" in capsys.readouterr().err


def test_a_missing_events_file_names_it(capsys):
    code = main(
        [
            "--definitions",
            DEFS,
            "offline",
            "--view",
            "account_activity",
            "--events",
            "/nowhere/events.csv",
            "--spine",
            ACCOUNTS,
        ]
    )
    assert code == 2
    assert "/nowhere/events.csv" in capsys.readouterr().err


def test_a_timestamp_that_is_not_a_timestamp_is_refused(capsys):
    with pytest.raises(SystemExit) as exit_info:
        main(
            [
                "--definitions",
                DEFS,
                "materialize",
                "--view",
                "account_activity",
                "--events",
                EVENTS,
                "--as-of",
                "wednesday",
                "--redis",
                "memory",
            ]
        )
    assert exit_info.value.code == 2
    assert "ISO 8601" in capsys.readouterr().err


def test_a_naive_timestamp_is_read_as_utc(capsys):
    code, out, _ = run(
        capsys,
        "materialize",
        "--view",
        "account_activity",
        "--events",
        EVENTS,
        "--as-of",
        "2026-03-02T09:00:00",
        "--redis",
        "memory",
    )
    assert code == 0
    assert "at 2026-03-02T09:00:00Z" in out
