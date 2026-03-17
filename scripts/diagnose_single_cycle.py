import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
import pandas as pd
import numpy as np
from free_fund.config import load_config
from free_fund.data import download_close_prices
from free_fund.orchestrator import CentralizedHedgeFundSystem
from free_fund.strategy_stack import StrategyEnsembleAgent, FundManagerAgent
from free_fund.contracts import ResearchSignal

cfg = load_config('configs/performance_v2.yaml')
cfg['backtest']['fast_mode'] = True
cfg['data_quality']['max_staleness_minutes'] = 99999999
cfg.setdefault('debug', {})
cfg['debug']['trace_weights'] = True

symbols = ['SPY', 'QQQ', 'XLK', 'XLV', 'XLE', 'TLT', 'GLD', 'IWM', 'MTUM']
cfg['portfolio']['symbols'] = symbols
cfg['portfolio']['lookback_days'] = int(cfg['portfolio'].get('lookback_days', 252))

prices = download_close_prices(symbols=symbols, start_date='2018-01-01', end_date='2021-01-01')

# Use a 2021 window where QQQ/XLK were clear winners
window = prices.loc[:'2021-01-01'].tail(252)

# Step 1: Raw strategy scores
agent = StrategyEnsembleAgent(
    strategy_weights=cfg['strategies']['weights'],
    dynamic_min_weight=0.03,
    dynamic_max_weight=0.45,
    dynamic_smoothing=0.20,
)
scores = agent.run(window, {})
combined = agent.weighted_score(scores)

print('=== STEP 1: COMBINED SCORE (before FundManager) ===')
for sym, val in combined.sort_values(ascending=False).items():
    print(f'  {sym}: {val:+.4f}')

# Step 2: FundManager output
fm_cfg = cfg.get('fund_manager', {})
fm = FundManagerAgent(
    max_weight=cfg['risk_hard_limits']['max_weight'],
    gross_limit=cfg['portfolio']['gross_limit'],
)
fm_weights = fm.run(
    combined_score=combined,
    prev_weights=None,
    risk_penalty=None,
    turnover_penalty=float(fm_cfg.get('turnover_penalty', 0.5)),
    gross_limit_override=1.0,
    top_k=int(fm_cfg.get('top_k', 10)),
)
print('\n=== STEP 2: FUND MANAGER OUTPUT (before RiskManager) ===')
for sym, val in fm_weights.sort_values(ascending=False).items():
    print(f'  {sym}: {val:+.4f}')
print(f'  GROSS: {fm_weights.abs().sum():.4f}  NET: {fm_weights.sum():.4f}')

# Step 3: Recreate orchestrator risk_penalty construction for same window
risk_penalty = window.pct_change().dropna().tail(60).std(ddof=0).reindex(symbols).fillna(0.0)
std = max(float(risk_penalty.std(ddof=0)), 1e-12)
z = ((risk_penalty - float(risk_penalty.mean())) / std)
risk_penalty_scaled = z * float(fm_cfg.get('risk_penalty_scale', 0.5))
print('\n=== STEP 3: ORCHESTRATOR RISK_PENALTY (scaled zscore) ===')
for sym, val in risk_penalty_scaled.sort_values(ascending=False).items():
    print(f'  {sym}: {val:+.4f}')
print('  (higher positive means stronger penalty subtraction in FundManager)')

# Step 4: Full orchestrator run for same date
system = CentralizedHedgeFundSystem(cfg)
system.cfg['portfolio']['end_date'] = '2021-01-01'
decision = system.run_cycle(execute=False, prices_override=prices.loc[:'2021-01-01'])
final_w = pd.Series(decision.target_weights).reindex(symbols).fillna(0.0)

print('\n=== STEP 4: FINAL ORCHESTRATOR WEIGHTS ===')
for sym, val in final_w.sort_values(ascending=False).items():
    print(f'  {sym}: {val:+.4f}')
print(f'  GROSS: {final_w.abs().sum():.4f}  NET: {final_w.sum():.4f}')
print(f'  Risk flags: {decision.risk_flags}')

print('\n=== DELTA: FundManager -> Final ===')
delta = final_w - fm_weights.reindex(final_w.index).fillna(0.0)
for sym, val in delta.sort_values(ascending=False).items():
    if abs(val) > 0.01:
        print(f'  {sym}: {val:+.4f}  (FM={fm_weights.get(sym,0):+.4f} -> Final={final_w[sym]:+.4f})')
