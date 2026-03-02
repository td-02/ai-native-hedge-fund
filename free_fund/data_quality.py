from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd


@dataclass
class DataQualityResult:
    ok: bool
    reasons: list[str]


@dataclass
class DataQualityAgent:
    max_nan_ratio: float = 0.01
    max_abs_daily_return: float = 0.30
    max_zero_price_ratio: float = 0.0
    max_staleness_minutes: int = 15

    def run(self, prices: pd.DataFrame) -> DataQualityResult:
        reasons: list[str] = []
        if prices.empty:
            return DataQualityResult(ok=False, reasons=["empty_price_frame"])

        nan_ratio = float(prices.isna().mean().mean())
        if nan_ratio > self.max_nan_ratio:
            reasons.append("nan_ratio_exceeded")

        zero_ratio = float((prices <= 0).mean().mean())
        if zero_ratio > self.max_zero_price_ratio:
            reasons.append("zero_or_negative_prices_detected")

        returns = prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna(how="all")
        if not returns.empty:
            max_abs_ret = float(returns.abs().max().max())
            if max_abs_ret > self.max_abs_daily_return:
                reasons.append("daily_return_outlier")

        idx = prices.index
        if len(idx) >= 2:
            last_ts = idx[-1]
            if hasattr(last_ts, "to_pydatetime"):
                last_dt = last_ts.to_pydatetime()
            else:
                last_dt = pd.Timestamp(last_ts).to_pydatetime()
            if last_dt.tzinfo is None:
                last_dt = last_dt.replace(tzinfo=timezone.utc)
            now = datetime.now(timezone.utc)
            delta_min = (now - last_dt.astimezone(timezone.utc)).total_seconds() / 60.0
            diffs = idx.to_series().diff().dropna()
            cadence_min = 0.0
            if not diffs.empty:
                cadence_min = float(diffs.median().total_seconds() / 60.0)
            effective_limit = max(float(self.max_staleness_minutes), cadence_min * 2.0)
            if delta_min > effective_limit:
                reasons.append("stale_market_data")

        return DataQualityResult(ok=len(reasons) == 0, reasons=reasons)
