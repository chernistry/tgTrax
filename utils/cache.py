import os
import hashlib
import pickle
import zlib
from typing import Any, Optional

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover
    redis = None  # type: ignore


_CLIENT = None
_MEM: dict[str, tuple[float, bytes]] = {}
import time


def get_redis_client():
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    if redis is None:
        return None
    url = os.getenv("REDIS_URL")
    if url:
        try:
            _CLIENT = redis.Redis.from_url(url)
            _CLIENT.ping()
            return _CLIENT
        except Exception:
            return None
    host = os.getenv("REDIS_HOST", "127.0.0.1")
    port = int(os.getenv("REDIS_PORT", "6379"))
    db = int(os.getenv("REDIS_DB", "0"))
    try:
        _CLIENT = redis.Redis(host=host, port=port, db=db)
        _CLIENT.ping()
        return _CLIENT
    except Exception:
        return None


def default_ttl(kind: str = "heavy") -> int:
    if kind == "short":
        return int(os.getenv("REDIS_TTL_SHORT_SECONDS", "600"))
    if kind == "medium":
        return int(os.getenv("REDIS_TTL_MEDIUM_SECONDS", "14400"))
    # heavy default
    return int(os.getenv("REDIS_TTL_HEAVY_SECONDS", os.getenv("REDIS_CACHE_TTL_SECONDS", "86400")))


def khash(*parts: Any) -> str:
    h = hashlib.sha256()
    for p in parts:
        if isinstance(p, bytes):
            h.update(p)
        else:
            h.update(str(p).encode("utf-8", errors="ignore"))
        h.update(b"|")
    return h.hexdigest()


def cache_get(key: str) -> Optional[Any]:
    # In-memory first
    now = time.time()
    try:
        exp, raw = _MEM.get(key, (0.0, b""))
        if exp and exp > now and raw:
            return pickle.loads(zlib.decompress(raw))
        elif key in _MEM:
            _MEM.pop(key, None)
    except Exception:
        pass
    # Fallback to Redis (optional)
    client = get_redis_client()
    if client is None:
        return None
    try:
        raw = client.get(key)
        if not raw:
            return None
        val = pickle.loads(zlib.decompress(raw))
        try:
            ttl = int(client.ttl(key) or default_ttl("heavy"))
            _MEM[key] = (now + ttl, zlib.compress(pickle.dumps(val, protocol=pickle.HIGHEST_PROTOCOL)))
        except Exception:
            pass
        return val
    except Exception:
        return None


def cache_set(key: str, value: Any, ttl: Optional[int] = None) -> None:
    raw = zlib.compress(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    _ttl = int(ttl if ttl is not None else default_ttl("heavy"))
    try:
        _MEM[key] = (time.time() + _ttl, raw)
    except Exception:
        pass
    client = get_redis_client()
    if client is None:
        return
    try:
        client.set(key, raw, ex=_ttl)
    except Exception:
        return
