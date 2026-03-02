from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from free_fund.config import load_config
from free_fund.healthcheck import HealthMonitor


def main() -> None:
    cfg = load_config("configs/default.yaml")
    out_dir = Path(cfg.get("system", {}).get("output_dir", "outputs"))
    timeout = int(cfg.get("health", {}).get("deadman_timeout_sec", 900))
    monitor = HealthMonitor(out_dir, deadman_timeout_sec=timeout)
    stale = monitor.is_stale()
    payload = {"stale": stale, "heartbeat_path": str(out_dir / "heartbeat.json")}
    print(json.dumps(payload, indent=2))
    if stale:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
