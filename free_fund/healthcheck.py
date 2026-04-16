from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json

import redis

from .config import load_settings
from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class HealthMonitor:
    deadman_timeout_sec: int = 900
    key: str = "ainhf:heartbeat"

    def __post_init__(self) -> None:
        settings = load_settings()
        redis_url = settings.model_dump().get("redis", {}).get("url", "redis://localhost:6379/0")
        self._fallback: dict[str, str] | None = None
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)
        try:
            self.redis.ping()
        except Exception as exc:
            logger.warning("health.redis_unavailable", error=str(exc))
            self.redis = None  # type: ignore

    def beat(self, run_id: str, status: str) -> None:
        payload = {
            "run_id": run_id,
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        if self.redis is None:
            self._fallback = payload
            return
        self.redis.set(self.key, json.dumps(payload, sort_keys=True), ex=self.deadman_timeout_sec * 2)

    def last(self) -> dict[str, str] | None:
        if self.redis is None:
            return self._fallback
        raw = self.redis.get(self.key)
        if not raw:
            return None
        return json.loads(raw)

    def is_stale(self) -> bool:
        payload = self.last()
        if not payload:
            return True
        ts = datetime.fromisoformat(payload["timestamp_utc"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
        return age > self.deadman_timeout_sec
