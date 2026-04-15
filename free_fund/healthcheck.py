from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json

import redis

from .config import load_settings


@dataclass
class HealthMonitor:
    deadman_timeout_sec: int = 900
    key: str = "ainhf:heartbeat"

    def __post_init__(self) -> None:
        settings = load_settings()
        redis_url = settings.model_dump().get("redis", {}).get("url", "redis://localhost:6379/0")
        self.redis = redis.Redis.from_url(redis_url, decode_responses=True)

    def beat(self, run_id: str, status: str) -> None:
        payload = {
            "run_id": run_id,
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.redis.set(self.key, json.dumps(payload, sort_keys=True), ex=self.deadman_timeout_sec * 2)

    def last(self) -> dict[str, str] | None:
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

