from __future__ import annotations

import argparse

from free_fund.config import load_config
from free_fund.data import download_close_prices
from free_fund.backtest import run_backtest, save_results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()

    cfg = load_config(args.config)
    pcfg = cfg['portfolio']

    prices = download_close_prices(
        symbols=pcfg['symbols'],
        start_date=pcfg['start_date'],
        end_date=pcfg['end_date'],
    )
    result = run_backtest(prices, cfg)
    save_results(result, args.out)

    print('Backtest complete.')
    for k, v in result.metrics.items():
        print(f'{k}: {v:.4f}')


if __name__ == '__main__':
    main()
