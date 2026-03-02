from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json

from .contracts import sha256_hex


@dataclass
class AuditLedger:
    base_dir: Path

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.event_log_path = self.base_dir / "events.jsonl"

    def _last_hash(self) -> str:
        if not self.event_log_path.exists():
            return "0" * 64
        lines = self.event_log_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return "0" * 64
        try:
            return json.loads(lines[-1]).get("event_hash", "0" * 64)
        except Exception:
            return "0" * 64

    def append(self, event_type: str, run_id: str, payload: dict[str, Any]) -> str:
        now_utc = datetime.now(timezone.utc).isoformat()
        prev_hash = self._last_hash()
        body = {
            "event_type": event_type,
            "run_id": run_id,
            "timestamp_utc": now_utc,
            "payload": payload,
            "prev_hash": prev_hash,
        }
        event_hash = sha256_hex(body)
        event = {**body, "event_hash": event_hash}

        with self.event_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, sort_keys=True))
            f.write("\n")
        return event_hash

    def verify_tail(self, last_n: int = 20) -> bool:
        if not self.event_log_path.exists():
            return True
        lines = self.event_log_path.read_text(encoding="utf-8").splitlines()
        if not lines:
            return True
        subset = lines[-last_n:]
        prev = None
        for raw in subset:
            obj = json.loads(raw)
            body = {
                "event_type": obj["event_type"],
                "run_id": obj["run_id"],
                "timestamp_utc": obj["timestamp_utc"],
                "payload": obj["payload"],
                "prev_hash": obj["prev_hash"],
            }
            expected = sha256_hex(body)
            if expected != obj.get("event_hash"):
                return False
            if prev is not None and obj.get("prev_hash") != prev:
                return False
            prev = obj.get("event_hash")
        return True
