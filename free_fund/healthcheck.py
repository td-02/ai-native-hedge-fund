from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path


@dataclass
class HealthMonitor:
    output_dir: Path
    deadman_timeout_sec: int = 900

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / "heartbeat.json"

    def beat(self, run_id: str, status: str) -> None:
        payload = {
            "run_id": run_id,
            "status": status,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        self.path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    def is_stale(self) -> bool:
        if not self.path.exists():
            return True
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        ts = datetime.fromisoformat(payload["timestamp_utc"])
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age = (datetime.now(timezone.utc) - ts.astimezone(timezone.utc)).total_seconds()
        return age > self.deadman_timeout_sec
