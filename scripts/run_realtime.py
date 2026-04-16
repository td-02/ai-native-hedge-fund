from __future__ import annotations

import argparse

from free_fund.config import load_config
from free_fund.env_utils import load_dotenv_file
from free_fund.orchestrator import CentralizedHedgeFundSystem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 means run forever.")
    parser.add_argument("--execute", action="store_true", help="Submit orders to configured broker.")
    args = parser.parse_args()

    load_dotenv_file()
    cfg = load_config(args.config)
    system = CentralizedHedgeFundSystem(cfg)
    max_cycles = None if args.max_cycles <= 0 else args.max_cycles
    system.run_realtime(
        poll_seconds=args.poll_seconds,
        execute=args.execute,
        max_cycles=max_cycles,
    )


if __name__ == "__main__":
    main()
