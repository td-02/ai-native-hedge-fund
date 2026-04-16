import sys
from pathlib import Path
sys.path.insert(0, str(Path(".").resolve()))

import inspect
import subprocess
import pandas as pd

import free_fund.orchestrator as orch

print("orchestrator.py location:", inspect.getfile(orch))

system_src = inspect.getsource(orch.CentralizedHedgeFundSystem)
for i, line in enumerate(system_src.split("\n")):
    if "StrategyEnsemble" in line or "weighted_score" in line or "strategy_stack" in line:
        print(f"  line {i}: {line}")

result = subprocess.run(["find", ".", "-name", "strategy_stack.py"], capture_output=True, text=True)
if result.returncode != 0 or not result.stdout.strip():
    result = subprocess.run(
        [sys.executable, "-c", "import pathlib; print('\\n'.join(str(p) for p in pathlib.Path('.').rglob('strategy_stack.py')))"],
        capture_output=True,
        text=True,
    )
print("\nAll strategy_stack.py files found:")
print(result.stdout)

from free_fund.strategy_stack import StrategyEnsembleAgent
print("\nStrategyEnsembleAgent imported from:", inspect.getfile(StrategyEnsembleAgent))

src = inspect.getsource(StrategyEnsembleAgent.weighted_score)
print("\nActual weighted_score source:")
print(src)

scores = {
    "trend_following": pd.Series({"IWM": 0.2376, "SPY": 0.9667, "QQQ": 0.9879, "XLK": 0.8862,
                                  "MTUM": 0.8099, "XLV": -0.1788, "GLD": -0.4743, "XLE": -1.3922, "TLT": -1.8430}),
    "relative_strength_rotation": pd.Series({"QQQ": 1.7678, "XLK": 1.7678, "GLD": -0.3536,
                                             "MTUM": -0.3536, "IWM": -0.3536, "SPY": -0.3536,
                                             "TLT": -0.3536, "XLV": -0.3536, "XLE": -1.4142}),
    "dual_momentum_gate": pd.Series({"QQQ": 1.2151, "IWM": 1.1739, "XLK": 1.0375, "MTUM": 0.4582,
                                     "SPY": 0.0693, "TLT": -0.2912, "GLD": -0.7955, "XLV": -1.3192, "XLE": -1.5480}),
    "volatility_carry": pd.Series({"SPY": 1.2797, "XLV": 0.8292, "QQQ": 0.6538, "XLK": 0.6164,
                                   "MTUM": 0.2705, "GLD": -0.1791, "TLT": -0.4474, "IWM": -0.8071, "XLE": -2.2159}),
    "regime_switching": pd.Series({"QQQ": 0.9879, "SPY": 0.9667, "XLK": 0.8862, "MTUM": 0.8099,
                                   "IWM": 0.2376, "XLV": -0.1788, "GLD": -0.4743, "XLE": -1.3922, "TLT": -1.8430}),
    "mean_reversion": pd.Series({"XLE": 0.2978, "GLD": 0.0720, "XLK": 0.0190, "SPY": -0.0094,
                                 "TLT": -0.0334, "MTUM": -0.0457, "QQQ": -0.0588, "XLV": -0.0910, "IWM": -0.1504}),
    "event_driven": pd.Series({"GLD": 0.8585, "XLE": 0.7057, "XLV": 0.3681, "XLK": 0.1655,
                               "SPY": 0.1285, "MTUM": 0.0645, "QQQ": -0.0898, "TLT": -0.8161, "IWM": -1.3848}),
}

weights = {"trend_following": 0.465, "relative_strength_rotation": 0.265, "dual_momentum_gate": 0.085,
           "volatility_carry": 0.086, "regime_switching": 0.085, "mean_reversion": 0.014, "event_driven": 0.000}

combined = pd.Series(0.0, index=list(scores["trend_following"].index))
for name, w in weights.items():
    combined += w * scores[name]

print("\nRAW combined (before any zscore):")
for sym, val in combined.sort_values(ascending=False).items():
    print(f"  {sym}: {val:+.4f}")

print(f"\nQQQ should be strongly positive. If it is, the bug is in the ORCHESTRATOR's")
print(f"call to agent.run() â€” it may be passing a DIFFERENT window that produces")
print(f"different per-strategy scores than the standalone diagnostic.")


