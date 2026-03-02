from __future__ import annotations

import numpy as np
import pandas as pd

from free_fund.strategy_stack import RiskManagerAgent


def _window() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=180, freq="D")
    x = np.linspace(100, 120, len(idx))
    return pd.DataFrame(
        {
            "SPY": x,
            "QQQ": x * 1.2,
            "IWM": x * 0.8,
            "TLT": x * 0.7,
            "GLD": x * 0.6,
        },
        index=idx,
    )


def test_concentration_and_beta_scaling():
    agent = RiskManagerAgent(
        max_weight=0.8,
        gross_limit=2.0,
        net_limit=1.0,
        max_annual_vol=1.0,
        drawdown_brake=0.5,
        brake_scale=0.5,
        concentration_top1_limit=0.3,
        concentration_top5_limit=0.8,
        beta_neutral_band=0.2,
        max_leverage_by_regime={"trend": 1.0},
    )
    w = pd.Series({"SPY": 0.9, "QQQ": 0.1, "IWM": 0.0, "TLT": 0.0, "GLD": 0.0})
    out, flags = agent.run(w, _window(), regime="trend", benchmark_symbol="SPY")
    assert float(out.abs().sum()) <= 1.0 + 1e-6
    assert any(f in flags for f in ["concentration_top1_scaled", "regime_leverage_scaled", "beta_neutrality_scaled"])
