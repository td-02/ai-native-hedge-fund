import sys, copy
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
import pandas as pd
import numpy as np
from free_fund.config import load_config
from free_fund.data import download_close_prices
from free_fund.orchestrator import CentralizedHedgeFundSystem

cfg = load_config('configs/performance_v2.yaml')
cfg['backtest']['fast_mode'] = True
cfg['data_quality']['max_staleness_minutes'] = 99999999

symbols = ['SPY', 'QQQ', 'XLK', 'XLV', 'XLE', 'TLT', 'GLD']
cfg['portfolio']['symbols'] = symbols
prices = download_close_prices(symbols=symbols, start_date='2020-01-01', end_date='2024-01-01')

system = CentralizedHedgeFundSystem(cfg)
system.cfg['portfolio']['end_date'] = '2024-01-01'
decision = system.run_cycle(execute=False, prices_override=prices)

w = decision.target_weights
print('=== TARGET WEIGHTS ===')
for k,v in sorted(w.items(), key=lambda x: abs(x[1]), reverse=True):
    print(f'  {k}: {v:.4f}')
print(f'  GROSS: {sum(abs(v) for v in w.values()):.4f}')
print(f'  NET:   {sum(w.values()):.4f}')

print('\n=== RISK FLAGS ===')
print(decision.risk_flags if hasattr(decision, 'risk_flags') else 'No risk_flags attribute')

print('\n=== FULL DECISION ===')
import json
print(json.dumps({k: str(v)[:100] for k,v in vars(decision).items()}, indent=2))

