from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np
import pandas as pd

from .contracts import ResearchSignal
try:
    from probabilistic_core import KalmanSignalTracker, BayesianSignalAggregator, BayesianPortfolioOptimizer, var_cvar
except Exception:  # pragma: no cover - compatibility fallback
    class KalmanSignalTracker:  # type: ignore
        def __init__(self, process_noise: float = 0.01, measurement_noise: float = 0.1):
            self.value = 0.0

        def update(self, measurement: float) -> dict:
            self.value = float(measurement)
            return {"filtered_signal": self.value, "signal_velocity": 0.0, "uncertainty": 1.0}

        def reset(self):
            self.value = 0.0

    class BayesianSignalAggregator:  # type: ignore
        def __init__(self, signal_names: list):
            self.signal_names = signal_names
            self.values = {}

        def update_signal(self, name: str, value: float, std: float):
            self.values[name] = (value, std)

        def get_combined_signal(self) -> dict:
            if not self.values:
                return {"combined": 0.0, "uncertainty": 1.0, "weights": {}, "n_signals": 0}
            vals = [float(v[0]) for v in self.values.values()]
            w = 1.0 / len(vals)
            return {
                "combined": float(sum(vals) * w),
                "uncertainty": 1.0,
                "weights": {k: w for k in self.values},
                "n_signals": len(vals),
            }

    class BayesianPortfolioOptimizer:  # type: ignore
        def __init__(self, risk_aversion: float = 2.5, tau: float = 0.05):
            self.risk_aversion = risk_aversion
            self.tau = tau

        def set_market_prior(self, returns_df: pd.DataFrame, market_caps: dict = None):
            return None

        def add_llm_view(self, view: dict):
            return None

        def optimize(self) -> dict:
            return {"weights": {}, "posterior_returns": {}, "posterior_cov": np.zeros((0, 0))}

        def optimize_with_uncertainty(self, n_samples: int = 500) -> dict:
            return {"mean_weights": {}, "std_weights": {}, "p5_weights": {}, "p95_weights": {}}

    def var_cvar(returns: np.ndarray, confidence: float = 0.95) -> dict:  # type: ignore
        arr = np.asarray(returns, dtype=float)
        if arr.size == 0:
            return {"var": 0.0, "cvar": 0.0, "confidence": confidence}
        q = float(np.quantile(arr, 1.0 - confidence))
        tail = arr[arr <= q]
        return {"var": q, "cvar": float(tail.mean()) if tail.size else q, "confidence": confidence}

from llm_router import llm_chat


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
        # Dual momentum: combine medium/long momentum with absolute trend filter.
        r21 = window.pct_change(21).iloc[-1]
        r126 = window.pct_change(126).iloc[-1]
        ma50 = window.rolling(50).mean().iloc[-1]
        ma200 = window.rolling(200).mean().iloc[-1]
        abs_trend = (ma50 > ma200).astype(float) * 2 - 1
        score = 0.45 * r21 + 0.40 * r126 + 0.15 * abs_trend
        return _zscore(score)

    def _mean_reversion(self, window: pd.DataFrame) -> pd.Series:
        # Mean-reversion from short-term overshoot against 20D moving average.
        px = window.iloc[-1]
        ma20 = window.rolling(20).mean().iloc[-1].replace(0, np.nan)
        dist = (px / ma20 - 1.0).fillna(0.0)
        r3 = window.pct_change(3).iloc[-1]
        return _zscore(-0.7 * dist - 0.3 * r3)

    def _vol_carry(self, window: pd.DataFrame) -> pd.Series:
        # Prefer low vol assets only if they are not in strong downtrends.
        rets = window.pct_change().dropna()
        vol20 = rets.tail(20).std(ddof=0).clip(lower=1e-4)
        inv_vol = 1.0 / vol20
        trend63 = window.pct_change(63).iloc[-1]
        score = inv_vol * (0.5 + 0.5 * np.sign(trend63))
        return _zscore(score)

    def _regime_switch(self, window: pd.DataFrame, trend: pd.Series, mean_rev: pd.Series) -> pd.Series:
        market_proxy = window.mean(axis=1).dropna()
        if len(market_proxy) < 64:
            return trend
        regime_score = float(market_proxy.pct_change(63).iloc[-1])
        drawdown = float((market_proxy / market_proxy.cummax() - 1.0).iloc[-1])
        breadth = float((window.pct_change(63).iloc[-1] > 0).mean())
        risk_on = regime_score > 0 and drawdown > -0.08 and breadth >= 0.5
        return trend if risk_on else mean_rev

    def _event_driven(self, symbols: list[str], research: dict[str, ResearchSignal]) -> pd.Series:
        data: dict[str, float] = {}
        for symbol in symbols:
            rs = research.get(symbol)
            if rs is None or rs.confidence <= 0.15:
                data[symbol] = 0.0
            else:
                data[symbol] = rs.sentiment * rs.confidence
        return _zscore(pd.Series(data, dtype=float))

    def _relative_strength_rotation(self, window: pd.DataFrame, top_n: int = 2, bottom_n: int = 1) -> pd.Series:
        # Cross-sectional rotation: overweight top momentum names, underweight weakest.
        mom = 0.6 * window.pct_change(63).iloc[-1] + 0.4 * window.pct_change(126).iloc[-1]
        mom = mom.reindex(window.columns).fillna(0.0)
        out = pd.Series(0.0, index=window.columns, dtype=float)
        if len(out) == 0:
            return out
        tn = max(1, min(int(top_n), len(out)))
        bn = max(0, min(int(bottom_n), len(out) - tn))
        top_idx = mom.nlargest(tn).index
        out.loc[top_idx] = 1.0
        if bn > 0:
            bot_idx = mom.nsmallest(bn).index
            out.loc[bot_idx] = -0.5
        return _zscore(out)

    def _dual_momentum_gate(self, window: pd.DataFrame) -> pd.Series:
        # Absolute + relative momentum with defensive fallback.
        symbols = list(window.columns)
        r126 = window.pct_change(126).iloc[-1].reindex(symbols).fillna(0.0)
        r21 = window.pct_change(21).iloc[-1].reindex(symbols).fillna(0.0)
        rel = _zscore(0.7 * r126 + 0.3 * r21)
        gate = (r126 > 0).astype(float)
        score = rel * gate
        if float(score.abs().sum()) <= 1e-12:
            # Risk-off fallback: prefer defensive assets if present.
            defensive = [s for s in symbols if s in ("TLT", "GLD", "IEF", "SHY", "BIL")]
            score = pd.Series(0.0, index=symbols, dtype=float)
            if defensive:
                w = 1.0 / len(defensive)
                for s in defensive:
                    score.loc[s] = w
        return _zscore(score)

    def run(self, window: pd.DataFrame, research: dict[str, ResearchSignal]) -> dict[str, pd.Series]:
        symbols = list(window.columns)
        trend = self._trend(window)
        mean_rev = self._mean_reversion(window)
        vol_carry = self._vol_carry(window)
        regime = self._regime_switch(window, trend, mean_rev)
        event = self._event_driven(symbols, research)
        rotation = self._relative_strength_rotation(window)
        dual_momentum = self._dual_momentum_gate(window)
        if float(event.abs().sum()) <= 1e-12:
            # Deterministic fallback when no high-confidence research is present.
            accel = (window.pct_change(5).iloc[-1] - window.pct_change(20).iloc[-1]).reindex(symbols).fillna(0.0)
            event = _zscore(accel)

        outputs = {
            "trend_following": trend,
            "mean_reversion": mean_rev,
            "volatility_carry": vol_carry,
            "regime_switching": regime,
            "event_driven": event,
            "relative_strength_rotation": rotation,
            "dual_momentum_gate": dual_momentum,
        }
        return {name: series.reindex(symbols).fillna(0.0) for name, series in outputs.items()}

    def weighted_score(
        self,
        strategy_scores: dict[str, pd.Series],
        weight_overrides: dict[str, float] | None = None,
    ) -> pd.Series:
        symbols = next(iter(strategy_scores.values())).index
        active_weights = self.strategy_weights if weight_overrides is None else weight_overrides
        combined = pd.Series(0.0, index=symbols)
        for name, score in strategy_scores.items():
            combined = combined + float(active_weights.get(name, 0.0)) * score
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

    def run(
        self,
        combined_score: pd.Series,
        prev_weights: pd.Series | None = None,
        risk_penalty: pd.Series | None = None,
        turnover_penalty: float = 0.0,
        gross_limit_override: float | None = None,
        top_k: int | None = None,
    ) -> pd.Series:
        score = combined_score.fillna(0.0).copy()
        if risk_penalty is not None:
            rp = risk_penalty.reindex(score.index).fillna(float(risk_penalty.mean()) if len(risk_penalty) else 0.0)
            score = score - rp

        if top_k is not None and top_k > 0 and top_k < len(score):
            keep = score.abs().nlargest(top_k).index
            score = score.where(score.index.isin(keep), 0.0)

        gross = float(score.abs().sum())
        if gross <= 1e-12:
            return pd.Series(0.0, index=score.index)
        gross_target = float(self.gross_limit if gross_limit_override is None else gross_limit_override)
        weights = (score / gross) * gross_target
        weights = weights.clip(lower=-self.max_weight, upper=self.max_weight)

        if prev_weights is not None:
            prev = prev_weights.reindex(weights.index).fillna(0.0)
            # Higher penalty keeps portfolio closer to previous allocations.
            alpha = 1.0 / (1.0 + max(0.0, turnover_penalty))
            weights = prev + alpha * (weights - prev)

        gross2 = float(weights.abs().sum())
        if gross2 > gross_target and gross2 > 1e-12:
            weights = weights / gross2 * gross_target
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


class LLMTrendStrategy:
    def __init__(self, tickers, config):
        self.tickers = list(tickers or [])
        self.config = dict(config or {})
        p_noise = float(self.config.get("kalman_process_noise", 0.01))
        m_noise = float(self.config.get("kalman_measurement_noise", 0.1))
        self.trackers = {t: KalmanSignalTracker(process_noise=p_noise, measurement_noise=m_noise) for t in self.tickers}

    def generate_signals(self, prices_df, headlines_by_ticker: dict = None) -> dict:
        prices = prices_df.copy()
        out: dict = {}
        for t in self.tickers:
            if t not in prices.columns or len(prices[t].dropna()) < 35:
                out[t] = {"signal": 0.0, "kalman_uncertainty": 1.0, "llm_used": False}
                continue
            s = prices[t].astype(float).dropna()
            ema10 = float(s.ewm(span=10, adjust=False).mean().iloc[-1])
            ema30 = float(s.ewm(span=30, adjust=False).mean().iloc[-1])
            raw = (ema10 / max(1e-12, ema30)) - 1.0
            kal = self.trackers[t].update(raw)
            filtered = float(kal["filtered_signal"])
            uncertainty = float(kal["uncertainty"])
            llm_used = False

            heads = []
            if headlines_by_ticker and t in headlines_by_ticker:
                heads = list(headlines_by_ticker.get(t, []))
            if heads:
                try:
                    prompt = (
                        f"Price trend signal {filtered:.3f} for {t}, headlines: {heads}. "
                        "Return JSON: {'trend_strength': float -1 to 1, "
                        "'trend_continuation_prob': float 0-1, 'reversal_risk': 'low|medium|high', "
                        "'conviction': float 0-1}"
                    )
                    data = json.loads(llm_chat(prompt=prompt, system="", json_mode=True, timeout=15))
                    trend_strength = float(max(-1.0, min(1.0, data.get("trend_strength", filtered))))
                    conviction = float(max(0.0, min(1.0, data.get("conviction", 0.0))))
                    filtered = filtered * (1.0 - 0.3 * conviction) + trend_strength * 0.3 * conviction
                    llm_used = True
                except Exception:
                    llm_used = False

            out[t] = {"signal": float(filtered), "kalman_uncertainty": uncertainty, "llm_used": llm_used}
        return out


class LLMPairsTradingStrategy:
    def __init__(self, pairs: list[tuple], config):
        self.pairs = list(pairs or [])
        self.config = dict(config or {})

    def generate_signals(self, prices_df) -> dict:
        out: dict = {}
        prices = prices_df.copy()
        for pair in self.pairs:
            if len(pair) != 2:
                continue
            a, b = pair
            key = f"{a}_{b}"
            if a not in prices.columns or b not in prices.columns:
                out[key] = {"spread_zscore": 0.0, "signal": 0.0, "llm_validated": False}
                continue
            pa = prices[a].astype(float).dropna()
            pb = prices[b].astype(float).dropna()
            idx = pa.index.intersection(pb.index)
            if len(idx) < 25:
                out[key] = {"spread_zscore": 0.0, "signal": 0.0, "llm_validated": False}
                continue
            spread = np.log(pa.loc[idx]) - np.log(pb.loc[idx])
            z = float((spread.iloc[-1] - spread.tail(20).mean()) / max(1e-12, spread.tail(20).std(ddof=0)))
            base_signal = float(-np.sign(z) * min(1.0, abs(z) / 3.0))
            llm_validated = False
            signal = base_signal
            try:
                prompt = (
                    f"Assets {a} and {b}, spread z-score={z:.2f}. "
                    "Return JSON: {'causal_link_strength': float 0-1, "
                    "'link_type': 'fundamental|statistical|broken', "
                    "'trade_signal': float -1 to 1, 'stop_if_zscore_exceeds': float}"
                )
                data = json.loads(llm_chat(prompt=prompt, system="", json_mode=True, timeout=15))
                strength = float(max(0.0, min(1.0, data.get("causal_link_strength", 0.0))))
                link_type = str(data.get("link_type", "statistical"))
                trade_signal = float(max(-1.0, min(1.0, data.get("trade_signal", base_signal))))
                if link_type != "broken" and strength > 0.5:
                    signal = trade_signal
                    llm_validated = True
            except Exception:
                signal = base_signal
            out[key] = {"spread_zscore": z, "signal": float(signal), "llm_validated": llm_validated}
        return out


class LLMRegimeSwitchingStrategy:
    def __init__(self, tickers, config):
        self.tickers = list(tickers or [])
        self.config = dict(config or {})
        from probabilistic_core import RegimeHMM

        self.hmm = RegimeHMM(n_regimes=4)

    def generate_signals(self, prices_df, regime_posterior: dict = None) -> dict:
        prices = prices_df.copy()
        if len(prices) < 20:
            return {t: {"signal": 0.0, "regime_scale": 1.0, "llm_used": False} for t in self.tickers}
        returns = prices.pct_change().dropna().mean(axis=1)
        if len(returns) > 60:
            try:
                self.hmm.fit(returns.tail(250).to_numpy(dtype=float))
            except Exception:
                pass
        post = regime_posterior or self.hmm.predict_proba(returns.tail(120).to_numpy(dtype=float))
        vix = float(self.config.get("vix", 20.0))
        ret = float(returns.tail(20).mean()) if len(returns) >= 20 else 0.0
        vol = float(returns.tail(20).std(ddof=0)) if len(returns) >= 20 else 0.0
        gross = 1.0
        llm_used = False
        try:
            prompt = (
                f"VIX={vix}, recent_return={ret:.2%}, vol={vol:.2%}, model regime={post}. "
                "Return JSON: {'confirmed_regime': 'bull|bear|crisis|sideways', "
                "'regime_confidence': float 0-1, 'recommended_gross_exposure': float 0.5-1.5, "
                "'favored_sectors': [list], 'risk_off_signal': bool}"
            )
            data = json.loads(llm_chat(prompt=prompt, system="", json_mode=True, timeout=15))
            gross = float(max(0.5, min(1.5, data.get("recommended_gross_exposure", 1.0))))
            llm_used = True
        except Exception:
            llm_used = False

        base = prices.pct_change(21).iloc[-1].reindex(self.tickers).fillna(0.0)
        z = (base - float(base.mean())) / max(1e-12, float(base.std(ddof=0)))
        z = z.fillna(0.0) * gross
        return {t: {"signal": float(z.get(t, 0.0)), "regime_scale": gross, "llm_used": llm_used} for t in self.tickers}


class AIAlphaComposite:
    def __init__(self, tickers, pairs, config):
        self.tickers = list(tickers or [])
        self.config = dict(config or {})
        self.trend = LLMTrendStrategy(self.tickers, self.config)
        self.pairs = LLMPairsTradingStrategy(list(pairs or []), self.config)
        self.regime = LLMRegimeSwitchingStrategy(self.tickers, self.config)
        self.bl_optimizer = BayesianPortfolioOptimizer(
            risk_aversion=float(self.config.get("risk_aversion", 2.5)),
            tau=float(self.config.get("tau", 0.05)),
        )

    def generate_signals(self, prices_df, headlines_by_ticker, regime_posterior) -> dict:
        trend = self.trend.generate_signals(prices_df, headlines_by_ticker=headlines_by_ticker or {})
        pairs = self.pairs.generate_signals(prices_df)
        regime = self.regime.generate_signals(prices_df, regime_posterior=regime_posterior or {})

        out: dict = {}
        for t in self.tickers:
            agg = BayesianSignalAggregator(signal_names=["trend", "pairs", "regime"])
            trend_signal = float(trend.get(t, {}).get("signal", 0.0))
            trend_unc = float(max(1e-6, trend.get(t, {}).get("kalman_uncertainty", 1.0)))
            agg.update_signal("trend", trend_signal, trend_unc)

            pair_vals = []
            for k, v in pairs.items():
                if k.startswith(f"{t}_") or k.endswith(f"_{t}"):
                    pair_vals.append(float(v.get("signal", 0.0)))
            pair_signal = float(np.mean(pair_vals)) if pair_vals else 0.0
            agg.update_signal("pairs", pair_signal, 0.2 if pair_vals else 1.0)

            regime_signal = float(regime.get(t, {}).get("signal", 0.0))
            regime_scale = float(max(1e-6, regime.get(t, {}).get("regime_scale", 1.0)))
            agg.update_signal("regime", regime_signal, 0.3 / regime_scale)

            combined = agg.get_combined_signal()
            out[t] = {
                "composite_signal": float(combined["combined"]),
                "uncertainty": float(combined["uncertainty"]),
                "contributing_strategies": {
                    "trend": trend.get(t, {}),
                    "pairs": {"signal": pair_signal},
                    "regime": regime.get(t, {}),
                    "weights": combined["weights"],
                    "tail_risk": var_cvar(prices_df[t].pct_change().dropna().tail(60).to_numpy(dtype=float), confidence=0.95)
                    if t in prices_df.columns
                    else {"var": 0.0, "cvar": 0.0, "confidence": 0.95},
                },
            }
        return out
