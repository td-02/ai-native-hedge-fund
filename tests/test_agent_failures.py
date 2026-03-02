from __future__ import annotations

import numpy as np
import pandas as pd

from free_fund.orchestrator import CentralizedHedgeFundSystem


def _prices() -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=220, freq="D", tz="UTC")
    base = np.linspace(100, 120, len(idx))
    return pd.DataFrame(
        {
            "SPY": base,
            "QQQ": base * 1.05,
            "IWM": base * 0.95,
            "TLT": base * 0.80,
            "GLD": base * 0.70,
        },
        index=idx,
    )


def _cfg() -> dict:
    return {
        "system": {"output_dir": "outputs"},
        "portfolio": {
            "symbols": ["SPY", "QQQ", "IWM", "TLT", "GLD"],
            "start_date": "2020-01-01",
            "end_date": None,
            "lookback_days": 126,
            "max_weight": 0.3,
            "gross_limit": 1.0,
        },
        "agent": {"enable_llm_research": False, "max_headlines": 5},
        "strategies": {
            "weights": {
                "trend_following": 0.4,
                "mean_reversion": 0.2,
                "volatility_carry": 0.1,
                "regime_switching": 0.2,
                "event_driven": 0.1,
            }
        },
        "risk_hard_limits": {
            "max_weight": 0.3,
            "gross_limit": 1.0,
            "net_limit": 0.3,
            "max_annual_vol": 0.2,
            "drawdown_brake": 0.15,
            "brake_scale": 0.5,
        },
        "data_quality": {
            "max_nan_ratio": 1.0,
            "max_abs_daily_return": 2.0,
            "max_zero_price_ratio": 1.0,
            "max_staleness_minutes": 10_000_000,
        },
        "resilience": {"degraded_mode_enabled": True},
        "health": {"deadman_timeout_sec": 9999},
        "alerts": {"enabled": False, "thresholds": {"daily_pnl_drift": 1.0, "strategy_disagreement": 99}},
        "execution": {"broker": "stub", "market_mode": "us"},
    }


def test_research_failure_degrades(monkeypatch):
    import free_fund.orchestrator as orch

    monkeypatch.setattr(orch, "download_close_prices", lambda **_: _prices())
    system = CentralizedHedgeFundSystem(_cfg())
    monkeypatch.setattr(system.research, "run", lambda symbols: (_ for _ in ()).throw(RuntimeError("boom")))
    decision = system.run_cycle(execute=False)
    assert isinstance(decision.target_weights, dict)
    assert set(decision.target_weights.keys()) == {"SPY", "QQQ", "IWM", "TLT", "GLD"}
