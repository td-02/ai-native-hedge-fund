import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
import pandas as pd
import numpy as np
from free_fund.config import load_config
from free_fund.data import download_close_prices
from free_fund.strategy_stack import StrategyEnsembleAgent

cfg = load_config('configs/performance_v2.yaml')
symbols = ['SPY', 'QQQ', 'XLK', 'XLV', 'XLE', 'TLT', 'GLD', 'IWM', 'MTUM']
prices = download_close_prices(symbols=symbols, start_date='2020-01-01', end_date='2024-01-01')

# Check what each strategy actually returns
weights_cfg = cfg['strategies']['weights']
agent = StrategyEnsembleAgent(
    strategy_weights=weights_cfg,
    dynamic_min_weight=0.03,
    dynamic_max_weight=0.45,
    dynamic_smoothing=0.20,
)

# Use last 252 days as window
window = prices.tail(252)
from free_fund.contracts import ResearchSignal
research = {}

scores = agent.run(window, research)

print('=== STRATEGY SCORES (positive = overweight, negative = underweight) ===')
for name, series in scores.items():
    print(f"\n{name}:")
    for sym, val in series.sort_values(ascending=False).items():
        print(f"  {sym}: {val:+.4f}")

# Check recent returns to verify direction
print('\n=== ACTUAL 1Y RETURNS (what signals SHOULD track) ===')
ret_1y = prices.pct_change(252).iloc[-1].sort_values(ascending=False)
for sym, val in ret_1y.items():
    print(f"  {sym}: {val:+.1%}")

print('\n=== CORRELATION: trend signal vs actual 1Y return ===')
trend = scores['trend_following']
print(pd.DataFrame({'trend_signal': trend, 'actual_1y_return': ret_1y}).to_string())

# Direction sanity check (top-3 scored assets vs trailing 252-day returns)
top3 = trend.sort_values(ascending=False).head(3).index.tolist()
neg_count = int((ret_1y.reindex(top3).fillna(0.0) < 0).sum())
if neg_count >= 2:
    print('\nWARNING: Signal direction sanity check failed - more than 50% of top-3 trend scores have negative 252-day returns.')
else:
    print('\nSignal direction sanity check passed.')

