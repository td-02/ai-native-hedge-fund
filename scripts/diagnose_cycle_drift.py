import sys, copy
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))
import pandas as pd
import numpy as np
from free_fund.config import load_config
from free_fund.data import download_close_prices
from free_fund.orchestrator import CentralizedHedgeFundSystem

cfg = load_config("configs/performance_v2.yaml")
cfg["backtest"]["fast_mode"] = True
cfg["data_quality"]["max_staleness_minutes"] = 99999999

symbols = ["SPY", "QQQ", "XLK", "XLV", "XLE", "TLT", "GLD", "IWM", "MTUM"]
cfg["portfolio"]["symbols"] = symbols
cfg["portfolio"]["lookback_days"] = 126

prices = download_close_prices(symbols=symbols, start_date="2018-01-01", end_date="2023-01-01")
rets = prices.pct_change().fillna(0.0)

system = CentralizedHedgeFundSystem(cfg)
system.cfg["portfolio"]["end_date"] = None
system.cfg["portfolio"]["start_date"] = "2018-01-01"

lookback = 126
step = 21
decision_rows = list(range(lookback, len(prices)-1, step))[:24]

prev_w = pd.Series(0.0, index=symbols)
cumulative = 1.0
print(f"{'Cycle':<5} {'Date':<12} {'Gross':>6} {'Net':>7} {'Ret+1':>8} {'CumRet':>8}  Weights  Flags")
print("-" * 100)

for idx, i in enumerate(decision_rows):
    dt = prices.index[i]
    system.cfg["portfolio"]["end_date"] = dt.strftime("%Y-%m-%d")
    try:
        decision = system.run_cycle(execute=False, prices_override=prices.iloc[:i+1])
        w = pd.Series(decision.target_weights).reindex(symbols).fillna(0.0)
        flags = decision.risk_flags
    except Exception as e:
        print(f"{idx:<5} {str(dt.date()):<12}  ERROR: {e}")
        continue

    gross = w.abs().sum()
    net = w.sum()
    next_i = decision_rows[idx+1] if idx+1 < len(decision_rows) else min(i+step, len(prices)-1)
    hold_rets = rets.iloc[i+1:next_i+1]
    daily_p = (hold_rets * w).sum(axis=1)
    next_ret = float((1+daily_p).prod()-1) if not daily_p.empty else 0.0
    cumulative *= (1 + next_ret)
    top = w.abs().nlargest(4)
    w_str = " ".join(f"{s}:{w[s]:+.2f}" for s in top.index)
    flags_str = ",".join(flags) if flags else "none"
    print(f"{idx:<5} {str(dt.date()):<12} {gross:>6.3f} {net:>7.3f} {next_ret:>8.4f} {cumulative:>8.4f}  {w_str}  [{flags_str}]")

# Print strategy weight evolution if accessible
print("\n=== STRATEGY WEIGHTS AFTER 24 CYCLES ===")
try:
    ens = system.strategy_ensemble
    for k,v in sorted(ens.strategy_weights.items()):
        print(f"  {k}: {v:.4f}")
except:
    print("  (cannot access strategy_ensemble)")

print("\n=== INITIAL WEIGHTS FROM CONFIG ===")
for k,v in sorted(cfg["strategies"]["weights"].items()):
    print(f"  {k}: {v:.4f}")

