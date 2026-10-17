"""Which clock a row is dated by, and that it is the store's and not ours."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from featurepipe.clock import ManualClock, RedisClock
from featurepipe.encoding import WRITTEN_AT_FIELD, decode_instant
from featurepipe.online.materialize import materialize
from featurepipe.online.store import FeatureStore, Status
from tests.conftest import AS_OF

SERVER_TIME = datetime(2026, 7, 4, 15, 30, 45, 500_000, tzinfo=UTC)


class ServerAt:
    """A client whose TIME answers an instant of the test's choosing.

    Every other command goes to the real in-memory client. fakeredis answers
    TIME from this process, so with the two agreeing there is nothing to tell
    apart: the only way to show the store asks the server is to make the server
    say something this process never would.
    """

    def __init__(self, inner, moment: datetime):
        self._inner = inner
        self._moment = moment

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def time(self):
        whole = int(self._moment.timestamp())
        return (whole, round((self._moment.timestamp() - whole) * 1_000_000))


def test_the_redis_clock_reads_the_server_rather_than_the_process():
    assert RedisClock(ServerAt(None, SERVER_TIME)).now() == SERVER_TIME


def test_the_redis_clock_keeps_the_microseconds_time_returns():
    """TIME is two integers. Dropping the second one would round an age to a
    second, and the merchant view's budget is five minutes."""
    clock = RedisClock(ServerAt(None, SERVER_TIME))
    assert clock.now().microsecond == 500_000


def test_the_default_clock_is_the_stores_own(account_view, client):
    """No clock passed, so the store takes the server's. fakeredis answers TIME
    from the real clock, so the stamp lands inside the window of the write."""
    before = datetime.now(tz=UTC)
    store = FeatureStore(client)
    key = store.write_vector(account_view, "A1", {"txn_count_24h": 1}, AS_OF)
    written_at = decode_instant(client.hget(key, WRITTEN_AT_FIELD))
    assert before <= written_at <= datetime.now(tz=UTC)


def test_a_row_is_stamped_with_the_servers_time_and_not_the_writers(
    account_view, account_events, client
):
    store = FeatureStore(ServerAt(client, SERVER_TIME))
    materialize(account_view, store, account_events, AS_OF)
    stamped = decode_instant(client.hget(store.key(account_view, "A1"), WRITTEN_AT_FIELD))
    assert stamped == SERVER_TIME
    assert store.read_vector(account_view, "A1").age_seconds == 0.0
    assert store.read_vector(account_view, "A1").status is Status.FRESH


def test_a_manual_clock_moves_only_when_it_is_told_to():
    clock = ManualClock(AS_OF)
    assert clock.now() == AS_OF
    assert clock.now() == AS_OF
    assert clock.advance(90) == AS_OF + timedelta(seconds=90)
    assert clock.now() == AS_OF + timedelta(seconds=90)


def test_a_manual_clock_can_step_backwards():
    """Which is what an NTP correction does, and the read path has to answer
    for a row stamped after the instant it is being judged at."""
    clock = ManualClock(AS_OF)
    clock.advance(-60)
    assert clock.now() == AS_OF - timedelta(seconds=60)
