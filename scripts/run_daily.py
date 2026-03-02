from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from free_fund.config import load_config
from free_fund.env_utils import load_dotenv_file
from free_fund.orchestrator import CentralizedHedgeFundSystem


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--dry-run', action='store_true', help='Skip broker order submission.')
    args = parser.parse_args()

    load_dotenv_file()
    cfg = load_config(args.config)
    decision = CentralizedHedgeFundSystem(cfg).run_cycle(execute=not args.dry_run)

    print(f'run_id: {decision.run_id}')
    print('weights:')
    for symbol, weight in decision.target_weights.items():
        print(f'  {symbol}: {weight:.6f}')
    if decision.risk_flags:
        print('risk_flags:', ', '.join(decision.risk_flags))


if __name__ == '__main__':
    main()
