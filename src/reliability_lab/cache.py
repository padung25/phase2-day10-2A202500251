from __future__ import annotations

import hashlib
import re
import time
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# Shared utilities — used by both ResponseCache and SharedRedisCache
# ---------------------------------------------------------------------------

PRIVACY_PATTERNS = re.compile(
    r"\b(balance|password|credit.card|ssn|social.security|user.\d+|account.\d+)\b",
    re.IGNORECASE,
)


def _is_uncacheable(query: str) -> bool:
    """Return True if query contains privacy-sensitive keywords."""
    return bool(PRIVACY_PATTERNS.search(query))


def _looks_like_false_hit(query: str, cached_key: str) -> bool:
    """Return True if query and cached key contain different 4-digit numbers (years, IDs)."""
    nums_q = set(re.findall(r"\b\d{4}\b", query))
    nums_c = set(re.findall(r"\b\d{4}\b", cached_key))
    return bool(nums_q and nums_c and nums_q != nums_c)


# ---------------------------------------------------------------------------
# In-memory cache
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class CacheEntry:
    key: str
    value: str
    created_at: float
    metadata: dict[str, str]


class ResponseCache:
    """In-memory semantic cache with TTL, privacy guardrails, and false-hit detection."""

    def __init__(self, ttl_seconds: int, similarity_threshold: float):
        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self._entries: list[CacheEntry] = []

    def get(self, query: str) -> tuple[str | None, float]:
        """Return (response, score) or (None, best_score) if no match found.

        Privacy queries are never served from cache.
        Year/ID mismatches are rejected as false hits even when token similarity is high.
        """
        if _is_uncacheable(query):
            return None, 0.0

        now = time.time()
        self._entries = [e for e in self._entries if now - e.created_at <= self.ttl_seconds]

        best_entry: CacheEntry | None = None
        best_score = 0.0
        for entry in self._entries:
            score = self.similarity(query, entry.key)
            if score > best_score:
                best_score = score
                best_entry = entry

        if best_score >= self.similarity_threshold and best_entry is not None:
            if _looks_like_false_hit(query, best_entry.key):
                return None, best_score
            return best_entry.value, best_score

        return None, best_score

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        """Store a response. Privacy-sensitive queries are silently skipped."""
        if _is_uncacheable(query):
            return
        self._entries.append(CacheEntry(query, value, time.time(), metadata or {}))

    @staticmethod
    def similarity(a: str, b: str) -> float:
        """Weighted token similarity with character-bigram refinement.

        Exact match returns 1.0 immediately.
        Token Jaccard uses stop-word down-weighting so content words dominate.
        Character bigram overlap adds sensitivity to sub-token differences
        (e.g., distinguishing "2024" from "2026" at the character level).
        The two scores are blended 70/30 in favour of token similarity.
        """
        a_norm = a.lower().strip()
        b_norm = b.lower().strip()
        if a_norm == b_norm:
            return 1.0

        STOP_WORDS = {
            "the", "a", "an", "for", "in", "of", "on", "at", "to", "with",
            "and", "or", "is", "are", "what", "how", "when", "where", "i",
            "me", "my", "do", "does", "did", "please", "give", "tell",
        }

        def token_weights(text: str) -> dict[str, float]:
            return {t: (0.3 if t in STOP_WORDS else 1.0) for t in text.split()}

        wa = token_weights(a_norm)
        wb = token_weights(b_norm)
        if not wa or not wb:
            return 0.0

        all_tokens = set(wa) | set(wb)
        intersection_w = sum(min(wa.get(t, 0.0), wb.get(t, 0.0)) for t in set(wa) & set(wb))
        union_w = sum(max(wa.get(t, 0.0), wb.get(t, 0.0)) for t in all_tokens)
        token_sim = intersection_w / union_w if union_w else 0.0

        # Character bigram overlap for sub-token sensitivity
        def bigrams(s: str) -> set[str]:
            s = s.replace(" ", "")
            return {s[i : i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else {s}

        bg_a = bigrams(a_norm)
        bg_b = bigrams(b_norm)
        if bg_a or bg_b:
            bi_sim = len(bg_a & bg_b) / len(bg_a | bg_b)
        else:
            bi_sim = 0.0

        return 0.7 * token_sim + 0.3 * bi_sim


# ---------------------------------------------------------------------------
# Redis shared cache
# ---------------------------------------------------------------------------


class SharedRedisCache:
    """Redis-backed shared cache for multi-instance deployments.

    Cache entries are stored as Redis Hashes with TTL via EXPIRE.
    Lookup is two-phase:
      1. Exact match on deterministic key hash.
      2. Similarity scan over all prefixed keys using ResponseCache.similarity().

    Privacy guardrails and false-hit detection mirror the in-memory cache.
    Redis errors are caught and return (None, 0.0) so the gateway keeps serving.
    """

    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int,
        similarity_threshold: float,
        prefix: str = "rl:cache:",
    ):
        import redis as redis_lib

        self.ttl_seconds = ttl_seconds
        self.similarity_threshold = similarity_threshold
        self.prefix = prefix
        self.false_hit_log: list[dict[str, object]] = []
        self._redis: Any = redis_lib.Redis.from_url(redis_url, decode_responses=True)

    def ping(self) -> bool:
        """Check Redis connectivity."""
        try:
            return bool(self._redis.ping())
        except Exception:
            return False

    def get(self, query: str) -> tuple[str | None, float]:
        """Look up a cached response from Redis.

        1. Return (None, 0.0) if privacy-sensitive.
        2. Try exact-match key; return (response, 1.0) if found.
        3. Scan all prefixed keys for best similarity match.
        4. Reject year/ID mismatches as false hits (logged to false_hit_log).
        """
        if _is_uncacheable(query):
            return None, 0.0

        try:
            # --- exact match ---
            key = f"{self.prefix}{self._query_hash(query)}"
            response = self._redis.hget(key, "response")
            if response is not None:
                return response, 1.0

            # --- similarity scan ---
            best_score = 0.0
            best_response: str | None = None
            best_cached_query: str | None = None

            for redis_key in self._redis.scan_iter(f"{self.prefix}*"):
                cached_query = self._redis.hget(redis_key, "query")
                if cached_query is None:
                    continue
                score = ResponseCache.similarity(query, cached_query)
                if score > best_score:
                    best_score = score
                    best_response = self._redis.hget(redis_key, "response")
                    best_cached_query = cached_query

            if best_score >= self.similarity_threshold and best_response is not None:
                if _looks_like_false_hit(query, best_cached_query or ""):
                    self.false_hit_log.append(
                        {
                            "query": query,
                            "cached_key": best_cached_query,
                            "score": best_score,
                        }
                    )
                    return None, best_score
                return best_response, best_score

            return None, best_score

        except Exception:
            return None, 0.0

    def set(self, query: str, value: str, metadata: dict[str, str] | None = None) -> None:
        """Store a response in Redis with TTL. Privacy queries are silently skipped."""
        if _is_uncacheable(query):
            return
        try:
            key = f"{self.prefix}{self._query_hash(query)}"
            self._redis.hset(key, mapping={"query": query, "response": value})
            self._redis.expire(key, self.ttl_seconds)
        except Exception:
            pass  # Redis unavailable — degrade gracefully

    def flush(self) -> None:
        """Remove all entries with this cache prefix (for testing)."""
        for key in self._redis.scan_iter(f"{self.prefix}*"):
            self._redis.delete(key)

    def close(self) -> None:
        """Close Redis connection."""
        if self._redis is not None:
            self._redis.close()

    @staticmethod
    def _query_hash(query: str) -> str:
        """Deterministic short hash for a query string."""
        return hashlib.md5(query.lower().strip().encode()).hexdigest()[:12]
