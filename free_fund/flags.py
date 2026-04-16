from __future__ import annotations

import redis

from .config import load_settings


def _redis_client() -> redis.Redis:
    settings = load_settings()
    redis_url = settings.model_dump().get("redis", {}).get("url", "redis://localhost:6379/0")
    return redis.Redis.from_url(redis_url, decode_responses=True)


def feature_enabled(flag: str) -> bool:
    client = _redis_client()
    value = client.get(f"ainhf:flags:{flag}")
    return str(value).lower() in {"1", "true", "on", "yes", "enabled"}

