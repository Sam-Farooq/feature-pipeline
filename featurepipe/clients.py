"""Getting a Redis client, real or in memory."""

from __future__ import annotations

from typing import Any

MEMORY = "memory"


def memory_client() -> Any:
    """An in-process Redis for tests, the parity check and the demo."""
    try:
        import fakeredis
    except ModuleNotFoundError as error:  # pragma: no cover - import guard
        raise RuntimeError(
            'the in-memory client lives in the dev extra: pip install -e ".[dev]"'
        ) from error
    return fakeredis.FakeStrictRedis(decode_responses=True)


def client_for(target: str) -> Any:
    """`memory`, or any redis:// url."""
    if target == MEMORY:
        return memory_client()
    import redis

    return redis.Redis.from_url(target, decode_responses=True)
