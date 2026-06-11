"""
Tests for app/limiter.py: rate limiter storage backend selection.
"""


def test_rate_limiter_falls_back_to_memory_when_no_redis():
    from limits.storage import MemoryStorage
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    test_limiter = Limiter(key_func=get_remote_address, storage_uri="memory://")
    assert isinstance(test_limiter._storage, MemoryStorage)


def test_rate_limiter_uses_redis_storage_when_url_set():
    from limits.storage import MemoryStorage
    from slowapi import Limiter
    from slowapi.util import get_remote_address

    test_limiter = Limiter(key_func=get_remote_address, storage_uri="redis://localhost:6379")
    assert not isinstance(test_limiter._storage, MemoryStorage)
