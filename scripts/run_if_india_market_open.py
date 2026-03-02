from __future__ import annotations

import argparse
from datetime import datetime, time
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo


def _load_holidays(path: Path) -> set[str]:
    if not path.exists():
        return set()
    out: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line)
    return out


def _is_market_open(now_ist: datetime, holidays: set[str]) -> bool:
    if now_ist.weekday() >= 5:
        return False
    if now_ist.date().isoformat() in holidays:
        return False
    start = time(9, 15)
    end = time(15, 30)
    t = now_ist.time()
    return start <= t <= end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--holidays-file", default="configs/market/nse_holidays.txt")
    parser.add_argument(
        "--command",
        default="python scripts/run_daily.py --config configs/live_stub.yaml",
        help="Command to run when market is open.",
    )
    args = parser.parse_args()

    now_ist = datetime.now(ZoneInfo("Asia/Kolkata"))
    holidays = _load_holidays(Path(args.holidays_file))
    if not _is_market_open(now_ist, holidays):
        print(f"Market closed (IST={now_ist.isoformat()}); skipping run.")
        return

    print(f"Market open (IST={now_ist.isoformat()}); running command.")
    completed = subprocess.run(args.command, shell=True)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
