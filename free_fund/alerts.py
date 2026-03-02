from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

import requests


@dataclass
class AlertManager:
    slack_webhook: str = ""
    enabled: bool = False
    timeout_sec: int = 8

    def notify(self, title: str, payload: dict[str, Any]) -> None:
        if not self.enabled or not self.slack_webhook:
            return
        message = {"text": f"{title}\n```{json.dumps(payload, sort_keys=True)}```"}
        try:
            requests.post(self.slack_webhook, json=message, timeout=self.timeout_sec)
        except Exception:
            # Alert failures must never break trading loop.
            return
