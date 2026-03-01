from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from free_fund.config import load_config
from free_fund.data import download_close_prices
from free_fund.agents import SignalAgent, RiskAgent, AllocatorAgent
from free_fund.paper import AlpacaPaperBroker, PaperBrokerStub
from free_fund.env_utils import load_dotenv_file


def main() -> None:
    load_dotenv_file()
    cfg = load_config('configs/default.yaml')
    pcfg = cfg['portfolio']
    rcfg = cfg['risk']
    acfg = cfg['agent']
    ecfg = cfg.get('execution', {})

    prices = download_close_prices(
        symbols=pcfg['symbols'],
        start_date=pcfg['start_date'],
        end_date=pcfg['end_date'],
    )
    window = prices.tail(pcfg['lookback_days'])

    signal = SignalAgent(
        lookback_days=pcfg['lookback_days'],
        enable_ollama_overlay=acfg['enable_ollama_overlay'],
        ollama_url=acfg['ollama_url'],
        ollama_model=acfg['ollama_model'],
    )
    risk = RiskAgent(
        target_annual_vol=rcfg['target_annual_vol'],
        min_annual_vol_floor=rcfg['min_annual_vol_floor'],
    )
    alloc = AllocatorAgent(max_weight=pcfg['max_weight'], gross_limit=pcfg['gross_limit'])

    s = signal.run(window)
    r = risk.run(s, window)
    w = alloc.run(r)
    latest_prices = prices.iloc[-1]

    broker_name = ecfg.get('broker', 'stub')
    if broker_name == 'alpaca_paper':
        broker = AlpacaPaperBroker(
            base_url=ecfg.get('alpaca_base_url', 'https://paper-api.alpaca.markets'),
            min_order_notional=float(ecfg.get('min_order_notional', 10.0)),
        )
        broker.submit_target_weights(w, latest_prices)
    else:
        PaperBrokerStub().submit_target_weights(w)


if __name__ == '__main__':
    main()
