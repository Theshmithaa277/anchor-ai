"""
cache.py — Response caching for repeated /ask queries.

Deliberately a *thin* interface (get/set/clear) so the in-memory
implementation here can be swapped for a Redis-backed one later without
touching any call site in app.py or quest.py — just implement the same
three methods against a Redis client instead of a dict.

Cache correctness depends on the caller building the right key. The
convention used by app.py is:

    make_cache_key(query, index_version)

`index_version` is a counter app.py bumps every time the active FAISS
index changes (new upload/ingest). Baking it into the key means a stale
cached answer from before the last ingest is simply never looked up
again — no separate invalidation pass is needed, and app.py additionally
calls `cache.clear()` on every successful ingest to free that memory
immediately rather than let it expire on TTL.
"""

import hashlib
import time
import threading


class ResponseCache:
    """
    A minimal, thread-safe, TTL-based in-memory cache.

    Not shared across multiple processes/machines — fine for a single
    Flask worker (the current deployment target), but this is exactly
    the constraint a future Redis swap removes.
    """

    def __init__(self, ttl_seconds: int = 600, max_entries: int = 500):
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._store: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None

            expires_at, value = entry
            if time.time() > expires_at:
                del self._store[key]
                return None

            return value

    def set(self, key: str, value: dict, ttl_seconds: int | None = None):
        ttl = ttl_seconds if ttl_seconds is not None else self._ttl
        with self._lock:
            if len(self._store) >= self._max_entries:
                # Cheap eviction: drop the oldest-inserted entry. Good
                # enough for a bounded dev/small-deployment cache; a
                # Redis swap would get real LRU semantics for free.
                oldest_key = next(iter(self._store), None)
                if oldest_key is not None:
                    del self._store[oldest_key]

            self._store[key] = (time.time() + ttl, value)

    def clear(self):
        with self._lock:
            self._store.clear()

    def __len__(self):
        with self._lock:
            return len(self._store)


def make_cache_key(query: str, index_version: int) -> str:
    """
    Normalize the query (case/whitespace-insensitive) and combine it with
    the current index version, so the same question against a different
    uploaded document never returns a stale cached answer.
    """
    normalized = " ".join(query.strip().lower().split())
    raw = f"v{index_version}:{normalized}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
