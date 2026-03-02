from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .contracts import ResearchSignal


def _zscore(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=series.index)
    return ((series - float(series.mean())) / std).clip(-3, 3)


@dataclass
class StrategyEnsembleAgent:
    strategy_weights: dict[str, float]

    def _trend(self, window: pd.DataFrame) -> pd.Series:
        r63 = window.pct_change(63).iloc[-1]
        ma20 = window.rolling(20).mean().iloc[-1]
        ma100 = window.rolling(100).mean().iloc[-1]
        trend = (ma20 > ma100).astype(float) * 2 - 1
        return _zscore(0.7 * r63 + 0.3 * trend)

    def _mean_reversion(self, window: pd.DataFrame) -> pd.Series:
        r5 = window.pct_change(5).iloc[-1]
        return _zscore(-r5)

    def _vol_carry(self, window: pd.DataFrame) -> pd.Series:
        vol = window.pct_change().dropna().std(ddof=0)
        inv_vol = 1.0 / vol.clip(lower=1e-4)
        return _zscore(inv_vol)

    def _regime_switch(self, window: pd.DataFrame, trend: pd.Series, mean_rev: pd.Series) -> pd.Series:
        market_proxy = window.mean(axis=1)
        regime_score = float(market_proxy.pct_change(63).iloc[-1])
        risk_on = regime_score > 0
        return trend if risk_on else mean_rev

    def _event_driven(self, symbols: list[str], research: dict[str, ResearchSignal]) -> pd.Series:
        data: dict[str, float] = {}
        for symbol in symbols:
            rs = research.get(symbol)
            if rs is None:
                data[symbol] = 0.0
            else:
                data[symbol] = rs.sentiment * rs.confidence
        return _zscore(pd.Series(data, dtype=float))

    def run(self, window: pd.DataFrame, research: dict[str, ResearchSignal]) -> dict[str, pd.Series]:
        symbols = list(window.columns)
        trend = self._trend(window)
        mean_rev = self._mean_reversion(window)
        vol_carry = self._vol_carry(window)
        regime = self._regime_switch(window, trend, mean_rev)
        event = self._event_driven(symbols, research)

        outputs = {
            "trend_following": trend,
            "mean_reversion": mean_rev,
            "volatility_carry": vol_carry,
            "regime_switching": regime,
            "event_driven": event,
        }
        return {name: series.reindex(symbols).fillna(0.0) for name, series in outputs.items()}

    def weighted_score(self, strategy_scores: dict[str, pd.Series]) -> pd.Series:
        symbols = next(iter(strategy_scores.values())).index
        combined = pd.Series(0.0, index=symbols)
        for name, score in strategy_scores.items():
            combined = combined + float(self.strategy_weights.get(name, 0.0)) * score
        return _zscore(combined)


@dataclass
class FundManagerAgent:
    max_weight: float
    gross_limit: float

    def run(self, combined_score: pd.Series) -> pd.Series:
        score = combined_score.fillna(0.0)
        gross = float(score.abs().sum())
        if gross <= 1e-12:
            return pd.Series(0.0, index=score.index)
        weights = (score / gross) * self.gross_limit
        weights = weights.clip(lower=-self.max_weight, upper=self.max_weight)

        gross2 = float(weights.abs().sum())
        if gross2 > self.gross_limit and gross2 > 1e-12:
            weights = weights / gross2 * self.gross_limit
        return weights.fillna(0.0)


@dataclass
class RiskManagerAgent:
    max_weight: float
    gross_limit: float
    net_limit: float
    max_annual_vol: float
    drawdown_brake: float
    brake_scale: float

    def run(self, candidate_weights: pd.Series, window: pd.DataFrame) -> tuple[pd.Series, list[str]]:
        flags: list[str] = []
        weights = candidate_weights.clip(-self.max_weight, self.max_weight).copy()

        net = float(weights.sum())
        if abs(net) > self.net_limit and abs(net) > 1e-12:
            adjust = (net - np.sign(net) * self.net_limit) / len(weights)
            weights = weights - adjust
            flags.append("net_exposure_clamped")

        gross = float(weights.abs().sum())
        if gross > self.gross_limit and gross > 1e-12:
            weights = weights / gross * self.gross_limit
            flags.append("gross_exposure_scaled")

        asset_returns = window.pct_change().dropna()
        if len(asset_returns) > 30:
            portfolio_returns = (asset_returns * weights).sum(axis=1)
            annual_vol = float(portfolio_returns.std(ddof=0) * np.sqrt(252))
            if annual_vol > self.max_annual_vol and annual_vol > 1e-12:
                weights = weights * (self.max_annual_vol / annual_vol)
                flags.append("vol_target_scaled")

            equity = (1 + portfolio_returns).cumprod()
            drawdown = float((equity / equity.cummax() - 1).min())
            if drawdown < -abs(self.drawdown_brake):
                weights = weights * self.brake_scale
                flags.append("drawdown_brake_active")

        return weights.fillna(0.0), flags
