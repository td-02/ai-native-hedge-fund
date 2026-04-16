from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st
import numpy as np
import pandas as pd

from free_fund.strategy_stack import RiskManagerAgent


def _window() -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=220, freq="D")
    x = np.linspace(100, 140, len(idx))
    return pd.DataFrame(
        {
            "SPY": x,
            "QQQ": x * 1.03,
            "IWM": x * 0.97,
            "TLT": x * 0.70,
            "GLD": x * 0.65,
        },
        index=idx,
    )


@settings(max_examples=30)
@given(
    w1=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    w2=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    w3=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    w4=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
    w5=st.floats(min_value=-1.0, max_value=1.0, allow_nan=False, allow_infinity=False),
)
def test_risk_manager_enforces_caps(w1: float, w2: float, w3: float, w4: float, w5: float):
    agent = RiskManagerAgent(
        max_weight=0.3,
        gross_limit=1.0,
        net_limit=0.6,
        max_annual_vol=0.4,
        drawdown_brake=0.2,
        brake_scale=0.5,
    )
    weights = pd.Series({"SPY": w1, "QQQ": w2, "IWM": w3, "TLT": w4, "GLD": w5}, dtype=float)
    out, _ = agent.run(weights, _window(), regime="trend", benchmark_symbol="SPY")
    assert float(out.abs().max()) <= 0.300001
    assert float(out.abs().sum()) <= 1.000001
    assert abs(float(out.sum())) <= 0.600001

