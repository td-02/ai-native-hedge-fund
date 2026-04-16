from __future__ import annotations

import json

from free_fund.config import load_config
from free_fund.healthcheck import HealthMonitor


def main() -> None:
    cfg = load_config("configs/default.yaml")
    timeout = int(cfg.get("health", {}).get("deadman_timeout_sec", 900))
    monitor = HealthMonitor(deadman_timeout_sec=timeout)
    stale = monitor.is_stale()
    payload = {"stale": stale, "heartbeat": monitor.last()}
    print(json.dumps(payload, indent=2))
    if stale:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
