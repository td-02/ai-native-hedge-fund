from __future__ import annotations

import numpy as np
import pandas as pd

from free_fund.data_quality import DataQualityAgent


def test_market_halt_data_gap_detection():
    idx = pd.date_range("2026-01-01", periods=150, freq="D", tz="UTC")
    prices = pd.DataFrame({"SPY": np.linspace(100, 110, len(idx))}, index=idx)
    # Inject gap and a huge jump to emulate bad market data.
    prices.iloc[40:45] = np.nan
    prices.iloc[60] = prices.iloc[59] * 2.0

    dq = DataQualityAgent(
        max_nan_ratio=0.01,
        max_abs_daily_return=0.30,
        max_zero_price_ratio=0.0,
        max_staleness_minutes=10_000_000,
    )
    result = dq.run(prices)
    assert not result.ok
    assert "nan_ratio_exceeded" in result.reasons or "daily_return_outlier" in result.reasons


def test_stale_data_detection():
    idx = pd.date_range("2024-01-01", periods=120, freq="D", tz="UTC")
    prices = pd.DataFrame({"SPY": np.linspace(100, 120, len(idx))}, index=idx)
    dq = DataQualityAgent(max_staleness_minutes=15)
    result = dq.run(prices)
    assert not result.ok
    assert "stale_market_data" in result.reasons
