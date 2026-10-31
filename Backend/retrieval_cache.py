"""Optional Redis retrieval cache. Redis outages always fall back to Qdrant."""

from functools import lru_cache
import hashlib
import json
import os
from types import SimpleNamespace


@lru_cache(maxsize=1)
def get_redis():
    url = os.getenv("REDIS_URL")
    if not url:
        return None
    try:
        import redis
        return redis.Redis.from_url(url, decode_responses=True,
            socket_connect_timeout=0.25, socket_timeout=0.25, health_check_interval=30)
    except Exception:
        return None


def _cache_key(client, role: str, query: str) -> str | None:
    try:
        epoch = client.get("securerag:retrieval:epoch")
        if epoch is None:
            client.set("securerag:retrieval:epoch", "1", nx=True)
            epoch = client.get("securerag:retrieval:epoch") or "1"
        digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return f"securerag:retrieval:{epoch}:{role}:{digest}"
    except Exception:
        return None


def get_cached(role: str, query: str):
    client = get_redis()
    if client is None:
        return None
    key = _cache_key(client, role, query)
    if not key:
        return None
    try:
        value = client.get(key)
        if not value:
            return None
        decoded = json.loads(value)
        if not isinstance(decoded, list):
            return None
        return [SimpleNamespace(id=item["id"], payload=item["payload"], score=item["score"],
                                semantic_score=item.get("semantic_score"))
                for item in decoded if isinstance(item, dict)
                and isinstance(item.get("id"), str) and isinstance(item.get("payload"), dict)]
    except Exception:
        return None


def set_cached(role: str, query: str, results: list, ttl: int = 45) -> None:
    client = get_redis()
    if client is None:
        return
    key = _cache_key(client, role, query)
    if not key:
        return
    try:
        value = [{"id": str(item.id), "payload": item.payload,
                  "score": float(getattr(item, "score", 0.0) or 0.0),
                  "semantic_score": getattr(item, "semantic_score", None)} for item in results]
        client.setex(key, ttl, json.dumps(value, separators=(",", ":")))
    except Exception:
        return


def invalidate_retrieval_cache() -> bool:
    """Advance the shared cache epoch after a successful index mutation."""
    client = get_redis()
    if client is None:
        return False
    try:
        client.incr("securerag:retrieval:epoch")
        return True
    except Exception:
        return False
