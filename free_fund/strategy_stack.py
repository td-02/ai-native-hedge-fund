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
    dynamic_min_weight: float = 0.05
    dynamic_max_weight: float = 0.50
    dynamic_smoothing: float = 0.30

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

    def estimate_strategy_quality(
        self,
        window: pd.DataFrame,
        strategy_scores: dict[str, pd.Series],
        eval_days: int = 60,
    ) -> dict[str, float]:
        """Estimate recent edge quality via simple next-day strategy returns."""
        returns = window.pct_change().dropna()
        if len(returns) < 20:
            return {k: 0.0 for k in strategy_scores}
        tail = returns.tail(max(20, int(eval_days)))
        quality: dict[str, float] = {}
        for name, score in strategy_scores.items():
            s = score.reindex(tail.columns).fillna(0.0)
            gross = float(s.abs().sum())
            if gross <= 1e-12:
                quality[name] = 0.0
                continue
            w = s / gross
            strat_ret = (tail * w).sum(axis=1)
            mu = float(strat_ret.mean())
            sig = float(strat_ret.std(ddof=0))
            quality[name] = mu / (sig + 1e-12)
        return quality

    def apply_dynamic_weights(self, quality_scores: dict[str, float]) -> dict[str, float]:
        """Blend base weights toward quality-ranked weights with hard min/max caps."""
        names = list(self.strategy_weights.keys())
        if not names:
            return {}
        q = pd.Series({k: float(quality_scores.get(k, 0.0)) for k in names})
        q = q.clip(lower=-3.0, upper=3.0)
        # Softmax-like transform from quality to positive weights.
        qexp = np.exp(q - float(q.max()))
        target = pd.Series(qexp, index=names, dtype=float)
        target = target / float(target.sum() + 1e-12)
        # Blend with existing weights for stability.
        cur = pd.Series({k: float(v) for k, v in self.strategy_weights.items()}, dtype=float)
        blended = (1.0 - self.dynamic_smoothing) * cur + self.dynamic_smoothing * target
        blended = blended.clip(lower=self.dynamic_min_weight, upper=self.dynamic_max_weight)
        blended = blended / float(blended.sum() + 1e-12)
        self.strategy_weights = {k: float(blended[k]) for k in names}
        return self.strategy_weights


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
    var_limit_95: float = 0.03
    es_limit_95: float = 0.04
    concentration_top1_limit: float = 0.30
    concentration_top5_limit: float = 0.80
    beta_neutral_band: float = 0.20
    jump_threshold: float = 0.06
    max_leverage_by_regime: dict[str, float] | None = None

    def _calc_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        if len(aligned) < 20:
            return 0.0
        cov = float(np.cov(aligned.iloc[:, 0], aligned.iloc[:, 1], ddof=0)[0, 1])
        var_b = float(np.var(aligned.iloc[:, 1], ddof=0))
        if var_b <= 1e-12:
            return 0.0
        return cov / var_b

    def run(
        self,
        candidate_weights: pd.Series,
        window: pd.DataFrame,
        regime: str = "trend",
        benchmark_symbol: str | None = None,
    ) -> tuple[pd.Series, list[str]]:
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

        # Regime-aware leverage cap.
        if self.max_leverage_by_regime:
            regime_cap = float(self.max_leverage_by_regime.get(regime, self.gross_limit))
            gross_now = float(weights.abs().sum())
            if gross_now > regime_cap and gross_now > 1e-12:
                weights = weights / gross_now * regime_cap
                flags.append("regime_leverage_scaled")

        # Concentration checks.
        sorted_abs = weights.abs().sort_values(ascending=False)
        if len(sorted_abs) > 0 and float(sorted_abs.iloc[0]) > self.concentration_top1_limit:
            weights = weights * (self.concentration_top1_limit / float(sorted_abs.iloc[0]))
            flags.append("concentration_top1_scaled")
        if len(sorted_abs) >= 5 and float(sorted_abs.iloc[:5].sum()) > self.concentration_top5_limit:
            weights = weights * (self.concentration_top5_limit / float(sorted_abs.iloc[:5].sum()))
            flags.append("concentration_top5_scaled")

        asset_returns = window.pct_change().dropna()
        if len(asset_returns) > 30:
            portfolio_returns = (asset_returns * weights).sum(axis=1)
            annual_vol = float(portfolio_returns.std(ddof=0) * np.sqrt(252))
            if annual_vol > self.max_annual_vol and annual_vol > 1e-12:
                weights = weights * (self.max_annual_vol / annual_vol)
                flags.append("vol_target_scaled")

            # Historical VaR/ES 95%.
            var95 = float(np.quantile(portfolio_returns, 0.05))
            es95 = float(portfolio_returns[portfolio_returns <= var95].mean()) if (portfolio_returns <= var95).any() else var95
            if abs(var95) > self.var_limit_95 and abs(var95) > 1e-12:
                scale = self.var_limit_95 / abs(var95)
                weights = weights * scale
                flags.append("var95_scaled")
            if abs(es95) > self.es_limit_95 and abs(es95) > 1e-12:
                scale = self.es_limit_95 / abs(es95)
                weights = weights * scale
                flags.append("es95_scaled")

            equity = (1 + portfolio_returns).cumprod()
            drawdown = float((equity / equity.cummax() - 1).min())
            if drawdown < -abs(self.drawdown_brake):
                weights = weights * self.brake_scale
                flags.append("drawdown_brake_active")

            # Tail risk jump detection.
            if float(portfolio_returns.abs().max()) > self.jump_threshold:
                weights = weights * self.brake_scale
                flags.append("tail_jump_brake_active")

            # Beta neutrality check.
            if benchmark_symbol and benchmark_symbol in asset_returns.columns:
                beta = self._calc_beta(portfolio_returns, asset_returns[benchmark_symbol])
                if abs(beta) > self.beta_neutral_band and abs(beta) > 1e-12:
                    beta_scale = self.beta_neutral_band / abs(beta)
                    weights = weights * beta_scale
                    flags.append("beta_neutrality_scaled")

        return weights.fillna(0.0), flags
