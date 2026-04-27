from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, TypeVar
import numpy as np
import pandas as pd

from .alerts import AlertManager
from .advanced_alpha import (
    AdaptiveLearningLayer,
    AlphaPipelineAgents,
    CrossAssetArbitrageAgents,
    MacroIntelligenceAgents,
    MicrostructureAgents,
    PrivateDataAlphaAgents,
)
from .audit import AuditLedger
from .brokers import build_broker_router
from .contracts import DecisionCycle, sha256_hex
from .data import download_close_prices
from .data_quality import DataQualityAgent
from .healthcheck import HealthMonitor
from .regime import MacroRegimeAgent, RegimeSnapshot
from .research_council import LLMResearchCouncil
from .research import ResearchAgent
from .resilience import BreakerRegistry
from .strategy_stack import FundManagerAgent, RiskManagerAgent, StrategyEnsembleAgent
from .tracing import TraceLMLogger
from .contracts import ResearchSignal
from .logging import get_logger
from .services import run_execution_stage as svc_run_execution_stage
from .services import run_research_stage as svc_run_research_stage
from .services import run_strategy_stage as svc_run_strategy_stage
try:
    from advanced_alpha import combine_advanced_alpha  # type: ignore
except Exception:  # pragma: no cover
    from .advanced_alpha import combine_advanced_alpha  # type: ignore
try:
    from research import enrich_signals_with_llm  # type: ignore
except Exception:  # pragma: no cover
    from .research import enrich_signals_with_llm  # type: ignore
from probabilistic_core import BayesianPortfolioOptimizer

T = TypeVar("T")
logger = get_logger(__name__)


@dataclass
class CentralizedHedgeFundSystem:
    cfg: dict

    def __post_init__(self) -> None:
        system_cfg = self.cfg.get("system", {})
        self.out_dir = Path(system_cfg.get("output_dir", "outputs"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLedger()
        self.health = HealthMonitor(
            deadman_timeout_sec=int(self.cfg.get("health", {}).get("deadman_timeout_sec", 900)),
        )
        res_cfg = self.cfg.get("resilience", {})
        self.breakers = BreakerRegistry(
            default_failure_threshold=int(res_cfg.get("failure_threshold", 3)),
            default_recovery_timeout_sec=int(res_cfg.get("recovery_timeout_sec", 120)),
        )
        alert_cfg = self.cfg.get("alerts", {})
        self.alerts = AlertManager(
            slack_webhook=str(alert_cfg.get("slack_webhook", "")),
            enabled=bool(alert_cfg.get("enabled", False)),
        )
        self.tracer = TraceLMLogger(
            enabled=bool(self.cfg.get("tracing", {}).get("enabled", True)),
            output_dir=self.out_dir / "traces",
        )

        acfg = self.cfg["agent"]
        scfg = self.cfg["strategies"]
        rcfg = self.cfg["risk_hard_limits"]
        pcfg = self.cfg["portfolio"]
        dqcfg = self.cfg.get("data_quality", {})
        self.degraded_mode = bool(self.cfg.get("resilience", {}).get("degraded_mode_enabled", True))

        self.research = ResearchAgent(
            enable_llm=bool(acfg.get("enable_llm_research", False)),
            ollama_model=acfg.get("ollama_model", "llama3.1:8b"),
            ollama_url=acfg.get("ollama_url", "http://localhost:11434/api/generate"),
            max_headlines=int(acfg.get("max_headlines", 8)),
            max_news_age_minutes=int(acfg.get("max_news_age_minutes", 15)),
            max_retries=int(acfg.get("max_retries", 2)),
            retry_base_delay_sec=float(acfg.get("retry_base_delay_sec", 0.4)),
        )
        self.strategy = StrategyEnsembleAgent(strategy_weights=scfg["weights"])
        dyn_cfg = self.cfg.get("learning", {})
        self.strategy.dynamic_min_weight = float(dyn_cfg.get("dynamic_min_weight", 0.05))
        self.strategy.dynamic_max_weight = float(dyn_cfg.get("dynamic_max_weight", 0.50))
        self.strategy.dynamic_smoothing = float(dyn_cfg.get("dynamic_smoothing", 0.30))
        self.alpha_pipeline = AlphaPipelineAgents(
            enabled_signals=list(self.cfg.get("alpha_pipeline", {}).get("enabled_signals", []) or []),
        )
        self.arbitrage = CrossAssetArbitrageAgents()
        self.macro_intel = MacroIntelligenceAgents()
        self.micro = MicrostructureAgents()
        self.private_alpha = PrivateDataAlphaAgents()
        self.learning = AdaptiveLearningLayer(
            decay=float(self.cfg.get("learning", {}).get("decay", 0.95)),
            min_weight=float(self.cfg.get("learning", {}).get("min_weight", 0.05)),
        )
        self.research_council = LLMResearchCouncil(
            enable=bool(self.cfg.get("research_council", {}).get("enabled", False)),
            ollama_model=acfg.get("ollama_model", "llama3.1:8b"),
            ollama_url=acfg.get("ollama_url", "http://localhost:11434/api/generate"),
            max_rounds=int(self.cfg.get("research_council", {}).get("max_rounds", 2)),
            request_timeout_sec=int(self.cfg.get("research_council", {}).get("request_timeout_sec", 8)),
        )
        self.regime_agent = MacroRegimeAgent()
        self.data_quality = DataQualityAgent(
            max_nan_ratio=float(dqcfg.get("max_nan_ratio", 0.01)),
            max_abs_daily_return=float(dqcfg.get("max_abs_daily_return", 0.30)),
            max_zero_price_ratio=float(dqcfg.get("max_zero_price_ratio", 0.0)),
            max_staleness_minutes=int(dqcfg.get("max_staleness_minutes", 15)),
        )
        self.fund_manager = FundManagerAgent(
            max_weight=float(pcfg["max_weight"]),
            gross_limit=float(pcfg["gross_limit"]),
        )
        self.risk = RiskManagerAgent(
            max_weight=float(rcfg["max_weight"]),
            gross_limit=float(rcfg["gross_limit"]),
            net_limit=float(rcfg["net_limit"]),
            max_annual_vol=float(rcfg["max_annual_vol"]),
            drawdown_brake=float(rcfg["drawdown_brake"]),
            brake_scale=float(rcfg["brake_scale"]),
            var_limit_95=float(rcfg.get("var_limit_95", 0.03)),
            es_limit_95=float(rcfg.get("es_limit_95", 0.04)),
            concentration_top1_limit=float(rcfg.get("concentration_top1_limit", 0.30)),
            concentration_top5_limit=float(rcfg.get("concentration_top5_limit", 0.80)),
            beta_neutral_band=float(rcfg.get("beta_neutral_band", 0.20)),
            jump_threshold=float(rcfg.get("jump_threshold", 0.06)),
            max_leverage_by_regime=rcfg.get("max_leverage_by_regime", {}),
            enable_var_scaling=bool(rcfg.get("enable_var_scaling", True)),
            min_net_exposure=float(rcfg.get("min_net_exposure", 0.0)),
        )
        self._last_final_weights = pd.Series(dtype=float)
        self._cycle_count = 0
        self._last_rebalance_cycle = -10**9

    def _load_window(self) -> tuple[pd.DataFrame, list[str], str]:
        pcfg = self.cfg["portfolio"]
        symbols = list(pcfg["symbols"])
        lookback_days = int(pcfg["lookback_days"])
        prices = download_close_prices(
            symbols=symbols,
            start_date=pcfg["start_date"],
            end_date=pcfg["end_date"],
        )
        window = prices.tail(lookback_days)
        if len(window) < 105:
            raise ValueError("Not enough history for multi-strategy decision.")
        run_id = self._make_run_id(window)
        return window, symbols, run_id

    def run_research_stage(self) -> dict:
        return svc_run_research_stage(self.cfg)

    def run_strategy_stage(self, research_payload: dict) -> dict:
        return svc_run_strategy_stage(self.cfg, research_payload)

    def run_execution_stage(self, weights_payload: dict) -> dict:
        return svc_run_execution_stage(self.cfg, weights_payload)

    @staticmethod
    def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
        s = float(sum(max(0.0, float(v)) for v in weights.values()))
        if s <= 1e-12:
            return weights
        return {k: max(0.0, float(v)) / s for k, v in weights.items()}

    @staticmethod
    def _zscore_series(series: pd.Series) -> pd.Series:
        std = float(series.std(ddof=0))
        if std <= 1e-12:
            return pd.Series(0.0, index=series.index)
        return (series - float(series.mean())) / std

    def _stage_call(
        self,
        stage: str,
        run_id: str,
        fn: Callable[[], T],
        degraded: Callable[[], T],
        trace_id: str = "",
        parent_span_id: str = "",
    ) -> T:
        stage_span_id = self.tracer.start_span(
            trace_id,
            parent_span_id,
            name=f"stage:{stage}",
            metadata={"run_id": run_id},
        )
        breaker = self.breakers.get(stage)
        if not breaker.allow_call():
            payload = {"stage": stage, "state": breaker.state, "reason": "breaker_open"}
            self.audit.append("degraded_mode", run_id, payload)
            out = degraded()
            self.tracer.finish_span(
                trace_id,
                stage_span_id,
                metadata_update={"mode": "degraded", "reason": "breaker_open"},
            )
            return out
        try:
            out = fn()
            breaker.on_success()
            self.tracer.finish_span(trace_id, stage_span_id, metadata_update={"mode": "normal"})
            return out
        except Exception as exc:
            breaker.on_failure()
            self.audit.append(
                "stage_error",
                run_id,
                {"stage": stage, "error": str(exc), "breaker_state": breaker.state},
            )
            self.alerts.notify("agent_stage_error", {"stage": stage, "run_id": run_id, "error": str(exc)})
            if self.degraded_mode:
                self.audit.append("degraded_mode", run_id, {"stage": stage, "reason": "exception"})
                out = degraded()
                self.tracer.finish_span(
                    trace_id,
                    stage_span_id,
                    metadata_update={"mode": "degraded", "reason": "exception"},
                    error=str(exc),
                )
                return out
            self.tracer.finish_span(trace_id, stage_span_id, error=str(exc))
            raise

    @staticmethod
    def _disagreement(strategy_scores: dict[str, pd.Series]) -> float:
        if not strategy_scores:
            return 0.0
        frame = pd.DataFrame(strategy_scores)
        return float(frame.std(axis=1, ddof=0).mean())

    def _check_pnl_drift(self, prices: pd.DataFrame, weights: pd.Series) -> float:
        # Approximate expected daily return as rolling mean of weighted returns; compare last day.
        returns = prices.pct_change().dropna()
        if len(returns) < 20:
            return 0.0
        port = (returns * weights).sum(axis=1)
        expected = float(port.iloc[-20:].mean())
        realized = float(port.iloc[-1])
        return abs(realized - expected)

    def _make_run_id(self, prices: pd.DataFrame) -> str:
        snapshot = {
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            "symbols": list(prices.columns),
            "tail_rows_json": prices.tail(3).round(6).to_json(date_format="iso", orient="split"),
        }
        return sha256_hex(snapshot)[:16]

    def _execution_broker(self):
        return build_broker_router(self.cfg)

    def _regime_from_window(self, window: pd.DataFrame) -> RegimeSnapshot:
        returns = window.pct_change().dropna()
        if returns.empty:
            return RegimeSnapshot(regime="trend", confidence=0.5, leverage_cap=1.0, risk_multiplier=1.0)
        market = returns.mean(axis=1)
        ann_vol = float(market.std(ddof=0) * np.sqrt(252))
        if ann_vol >= 0.30:
            return RegimeSnapshot(regime="stress", confidence=0.8, leverage_cap=1.0, risk_multiplier=0.5)
        if ann_vol >= 0.20:
            return RegimeSnapshot(regime="meanrev", confidence=0.6, leverage_cap=2.5, risk_multiplier=0.8)
        return RegimeSnapshot(regime="trend", confidence=0.7, leverage_cap=3.0, risk_multiplier=1.0)

    def run_cycle(self, execute: bool = False, prices_override: pd.DataFrame | None = None) -> DecisionCycle:
        pcfg = self.cfg["portfolio"]
        symbols = list(pcfg["symbols"])
        lookback_days = int(pcfg["lookback_days"])
        fast_backtest = bool(self.cfg.get("backtest", {}).get("fast_mode", False))

        if prices_override is not None:
            prices = prices_override.copy()
        else:
            prices = download_close_prices(
                symbols=symbols,
                start_date=pcfg["start_date"],
                end_date=pcfg["end_date"],
            )
        window = prices.tail(lookback_days)
        if len(window) < 105:
            raise ValueError("Not enough history for multi-strategy decision.")

        run_id = self._make_run_id(window)
        trace_id, root_span_id = self.tracer.start_trace(
            name="run_cycle",
            metadata={
                "run_id": run_id,
                "execute": execute,
                "symbols": symbols,
            },
        )
        self.health.beat(run_id, "cycle_start")
        self.audit.append("market_snapshot", run_id, {"rows": len(window), "symbols": symbols})
        debug_trace = bool(self.cfg.get("debug", {}).get("trace_weights", False))

        def _dbg(label: str, series: pd.Series) -> None:
            if not debug_trace:
                return
            s = series.reindex(symbols).fillna(0.0)
            focus = ["QQQ", "XLK", "SPY", "IWM", "TLT", "GLD", "XLE", "XLV", "MTUM"]
            vals = " ".join(f"{k}:{float(s.get(k, 0.0)):+.4f}" for k in focus if k in s.index)
            logger.info(
                "orchestrator.trace",
                label=label,
                gross=float(s.abs().sum()),
                net=float(s.sum()),
                values=vals,
            )

        def _check_score_integrity(score: pd.Series, label: str, reference_score: pd.Series | None) -> None:
            if reference_score is None:
                return
            import warnings

            corr = float(score.corr(reference_score)) if len(score) and len(reference_score) else float("nan")
            if not np.isfinite(corr):
                corr = 1.0
            if debug_trace:
                logger.info("orchestrator.score_integrity", label=label, correlation=corr)
            if corr < 0.5:
                warnings.warn(
                    f"[SCORE INTEGRITY] {label}: correlation with pre-blend score = {corr:.3f} < 0.5. "
                    "Possible blend corruption.",
                    RuntimeWarning,
                )

        def _apply_blend_guard(
            base: pd.Series,
            label: str,
            blend_series: pd.Series,
            blend_weight: float,
        ) -> pd.Series:
            if float(blend_weight) <= 0.0:
                return base
            candidate = base + float(blend_weight) * blend_series.reindex(symbols).fillna(0.0)
            _check_score_integrity(candidate, label, base)
            base_abs = float(base.abs().sum())
            cand_abs = float(candidate.abs().sum())
            if base_abs > 1e-12 and cand_abs < 0.5 * base_abs:
                import warnings

                warnings.warn(
                    f"[SCORE INTEGRITY] {label}: blend reduced |combined_score| by >50% "
                    f"(from {base_abs:.4f} to {cand_abs:.4f}); skipping blend.",
                    RuntimeWarning,
                )
                return base
            return candidate

        dq = self.data_quality.run(window)
        if not dq.ok:
            self.audit.append("data_quality_failed", run_id, {"reasons": dq.reasons})
            self.health.beat(run_id, "data_quality_failed")
            # Degraded safe mode: keep zero weights and skip execution.
            decision = DecisionCycle(
                run_id=run_id,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                symbols=symbols,
                target_weights={s: 0.0 for s in symbols},
                risk_flags=["data_quality_failed"] + dq.reasons,
                model_versions={"runtime": "degraded_v1"},
            )
            self.audit.append("decision", run_id, decision.to_dict())
            (self.out_dir / "last_decision.json").write_text(
                json.dumps(decision.to_dict(), indent=2, sort_keys=True),
                encoding="utf-8",
            )
            self.alerts.notify("data_quality_failed", {"run_id": run_id, "reasons": dq.reasons})
            self.tracer.finalize_trace(
                trace_id,
                root_span_id,
                metadata_update={"status": "data_quality_failed", "reasons": dq.reasons},
            )
            return decision

        if fast_backtest:
            research = {
                s: ResearchSignal(
                    symbol=s,
                    sentiment=0.0,
                    confidence=0.0,
                    summary="fast_backtest_research_bypassed",
                    source_urls=[],
                )
                for s in symbols
            }
        else:
            research = self._stage_call(
                "research",
                run_id,
                lambda: self.research.run(symbols),
                lambda: {
                    s: ResearchSignal(
                        symbol=s,
                        sentiment=0.0,
                        confidence=0.0,
                        summary="research_degraded_fallback",
                        source_urls=[],
                    )
                    for s in symbols
                },
                trace_id=trace_id,
                parent_span_id=root_span_id,
            )
        self.audit.append("research_output", run_id, {k: v.to_dict() for k, v in research.items()})

        if debug_trace and len(window) > 0:
            start_dt = window.index[0]
            end_dt = window.index[-1]
            logger.info("orchestrator.window_debug", shape=window.shape, start=str(start_dt), end=str(end_dt))
            if "QQQ" in window.columns and "IWM" in window.columns:
                logger.info(
                    "orchestrator.window_last_row",
                    qqq=float(window["QQQ"].iloc[-1]),
                    iwm=float(window["IWM"].iloc[-1]),
                )
                qqq_252 = window["QQQ"].pct_change(252).iloc[-1] if len(window) > 252 else float("nan")
                iwm_252 = window["IWM"].pct_change(252).iloc[-1] if len(window) > 252 else float("nan")
                logger.info("orchestrator.window_252d", qqq_252=float(qqq_252), iwm_252=float(iwm_252))

        strategy_scores = self._stage_call(
            "strategy",
            run_id,
            lambda: self.strategy.run(window, research),
            lambda: {"trend_following": self.strategy._trend(window).reindex(symbols).fillna(0.0)},
            trace_id=trace_id,
            parent_span_id=root_span_id,
        )
        self.audit.append(
            "strategy_scores",
            run_id,
            {name: score.round(6).to_dict() for name, score in strategy_scores.items()},
        )

        alpha_weight = float(self.cfg.get("alpha_pipeline", {}).get("blend_weight", 0.15))
        if alpha_weight > 0:
            alpha_scores = self._stage_call(
                "alpha_pipeline",
                run_id,
                lambda: self.alpha_pipeline.run(window),
                lambda: {"alpha_pipeline_degraded": pd.Series(0.0, index=symbols)},
                trace_id=trace_id,
                parent_span_id=root_span_id,
            )
        else:
            alpha_scores = {}
        self.audit.append(
            "alpha_pipeline_scores",
            run_id,
            {name: score.round(6).to_dict() for name, score in alpha_scores.items()},
        )

        arb_weight = float(self.cfg.get("arbitrage", {}).get("blend_weight", 0.10))
        if arb_weight > 0:
            arb_scores = self._stage_call(
                "arbitrage",
                run_id,
                lambda: self.arbitrage.run(window),
                lambda: {"arbitrage_degraded": pd.Series(0.0, index=symbols)},
                trace_id=trace_id,
                parent_span_id=root_span_id,
            )
        else:
            arb_scores = {}
        self.audit.append(
            "arbitrage_scores",
            run_id,
            {name: score.round(6).to_dict() for name, score in arb_scores.items()},
        )

        private_weight = float(self.cfg.get("private_alpha", {}).get("blend_weight", 0.05))
        private_scores = self.private_alpha.run(symbols) if private_weight > 0 else {}
        council_weight = float(self.cfg.get("research_council", {}).get("blend_weight", 0.10))
        if fast_backtest or council_weight <= 0:
            council_scores = pd.Series(0.0, index=symbols)
            council_details = {s: {"mode": "disabled_or_fast_backtest"} for s in symbols}
        else:
            council_scores, council_details = self.research_council.run(symbols)
        self.audit.append(
            "research_council",
            run_id,
            {"scores": council_scores.round(6).to_dict(), "details": council_details},
        )
        disagreement = self._disagreement(strategy_scores)
        self.audit.append("strategy_disagreement", run_id, {"value": disagreement})
        dis_threshold = float(self.cfg.get("alerts", {}).get("thresholds", {}).get("strategy_disagreement", 0.7))
        if disagreement > dis_threshold:
            self.alerts.notify("strategy_disagreement_high", {"run_id": run_id, "value": disagreement})

        if fast_backtest:
            regime_snapshot = self._regime_from_window(window)
        else:
            regime_snapshot = self._stage_call(
                "regime",
                run_id,
                lambda: self.regime_agent.run(prefer_india=str(self.cfg.get("execution", {}).get("market_mode", "us")) == "india"),
                lambda: RegimeSnapshot(regime="trend", confidence=0.5, leverage_cap=1.0, risk_multiplier=1.0),
                trace_id=trace_id,
                parent_span_id=root_span_id,
            )
        self.audit.append(
            "regime_snapshot",
            run_id,
            {
                "regime": regime_snapshot.regime,
                "confidence": regime_snapshot.confidence,
                "leverage_cap": regime_snapshot.leverage_cap,
                "risk_multiplier": regime_snapshot.risk_multiplier,
            },
        )

        regime_cfg = self.cfg.get("regime_controls", {})
        regime_w_map = regime_cfg.get("strategy_weights_by_regime", {})
        base_weights = dict(self.cfg.get("strategies", {}).get("weights", {}) or {})
        regime_weights = regime_w_map.get(regime_snapshot.regime)
        w_override = None
        if isinstance(regime_weights, dict):
            blend_factor = float(regime_cfg.get("regime_blend_factor", 0.30))
            blend_factor = min(1.0, max(0.0, blend_factor))
            keys = set(base_weights.keys()) | set(regime_weights.keys())
            blended = {
                k: (1.0 - blend_factor) * float(base_weights.get(k, regime_weights.get(k, 0.0)))
                + blend_factor * float(regime_weights.get(k, base_weights.get(k, 0.0)))
                for k in keys
            }
            w_override = self._normalize_weights(blended)
        if debug_trace:
            logger.info("orchestrator.weight_overrides", value=w_override)
        combined = self.strategy.weighted_score(strategy_scores, weight_overrides=w_override) * regime_snapshot.risk_multiplier
        _dbg("combined_score", combined)

        if alpha_weight > 0 and alpha_scores:
            alpha_combined = pd.concat(alpha_scores.values(), axis=1).mean(axis=1).reindex(symbols).fillna(0.0)
            combined = _apply_blend_guard(combined, "alpha_pipeline", alpha_combined, alpha_weight)
        if arb_weight > 0 and arb_scores:
            arb_combined = pd.concat(arb_scores.values(), axis=1).mean(axis=1).reindex(symbols).fillna(0.0)
            combined = _apply_blend_guard(combined, "arbitrage", arb_combined, arb_weight)
        if private_weight > 0 and private_scores:
            p_combined = pd.concat(private_scores.values(), axis=1).mean(axis=1).reindex(symbols).fillna(0.0)
            combined = _apply_blend_guard(combined, "private_alpha", p_combined, private_weight)
        if council_weight > 0:
            combined = _apply_blend_guard(
                combined,
                "research_council",
                council_scores.reindex(symbols).fillna(0.0),
                council_weight,
            )

        # Explicit benchmark-relative objective: expected excess return vs benchmark.
        bench_cfg = self.cfg.get("benchmark", {})
        bench_mode = str(bench_cfg.get("mode", "symbol"))
        bench_symbol = str(bench_cfg.get("symbol", symbols[0]))
        bench_w = float(bench_cfg.get("alpha_weight", 0.20))
        r20 = window.pct_change(20).iloc[-1].reindex(symbols).fillna(0.0)
        if bench_mode == "equal_weight":
            bench_ret = float(r20.mean())
        else:
            bench_ret = float(r20.get(bench_symbol, 0.0))
        if bench_w > 0:
            bench_series = self._zscore_series(r20 - bench_ret).reindex(symbols).fillna(0.0)
            combined = _apply_blend_guard(combined, "benchmark_relative", bench_series, bench_w)

        ai_alpha_cfg = self.cfg.get("ai_alpha", {})
        if bool(ai_alpha_cfg.get("enabled", False)):
            ai_alpha_out = run_ai_alpha_layer(
                window,
                {k: float(v) for k, v in combined.reindex(symbols).fillna(0.0).to_dict().items()},
                self.cfg,
                audit_logger=self.audit,
            )
            combined = pd.Series(
                {k: float(v) for k, v in ai_alpha_out.items()},
                dtype=float,
            ).reindex(symbols).fillna(0.0)
            self.audit.append(
                "ai_alpha_layer",
                run_id,
                {
                    "enabled": True,
                    "blend_weight": float(ai_alpha_cfg.get("blend_weight", 0.3)),
                    "tickers": list(symbols),
                },
            )

        # Optional adaptive update for strategy blend.
        if bool(self.cfg.get("learning", {}).get("enabled", False)):
            recent_perf = self.strategy.estimate_strategy_quality(
                window=window,
                strategy_scores=strategy_scores,
                eval_days=int(self.cfg.get("learning", {}).get("quality_eval_days", 60)),
            )
            self.strategy.apply_dynamic_weights(recent_perf)
            self.strategy.strategy_weights = self.learning.update_weights(
                self.strategy.strategy_weights,
                recent_perf,
            )
            self.audit.append("adaptive_learning_update", run_id, {"weights": self.strategy.strategy_weights})

        micro_weight = float(self.cfg.get("microstructure", {}).get("blend_weight", 0.05))
        if micro_weight > 0:
            micro = self.micro.order_book_agent(symbols) + self.micro.tape_reading(symbols) + self.micro.latency_arb(symbols)
            if self.micro.flash_crash_detector(window):
                combined = combined * 0.0
                self.audit.append("flash_crash_detector", run_id, {"triggered": True})
            else:
                combined = _apply_blend_guard(combined, "microstructure", micro, micro_weight)

        macro_snapshot = {"mode": "fast_backtest_disabled"} if fast_backtest else self.macro_intel.snapshot()
        self.audit.append("macro_intelligence", run_id, macro_snapshot)
        risk_penalty = window.pct_change().dropna().tail(60).std(ddof=0).reindex(symbols).fillna(0.0)
        fm_cfg = self.cfg.get("fund_manager", {})
        turnover_penalty = float(fm_cfg.get("turnover_penalty", 1.0))
        risk_penalty_scale = float(fm_cfg.get("risk_penalty_scale", 0.5))
        gross_by_regime = regime_cfg.get("gross_limit_by_regime", {})
        gross_override = gross_by_regime.get(regime_snapshot.regime)

        pre_risk = self.fund_manager.run(
            combined_score=combined,
            prev_weights=self._last_final_weights.reindex(symbols).fillna(0.0) if not self._last_final_weights.empty else None,
            risk_penalty=self._zscore_series(risk_penalty) * risk_penalty_scale,
            turnover_penalty=turnover_penalty,
            gross_limit_override=(float(gross_override) if gross_override is not None else None),
            top_k=int(fm_cfg.get("top_k", 0)) if int(fm_cfg.get("top_k", 0)) > 0 else None,
        )
        _dbg("fund_manager_pre_risk", pre_risk)
        if regime_snapshot.regime == "stress":
            defensive_assets = regime_cfg.get("defensive_assets", ["TLT", "GLD"])
            defensive_tilt = float(regime_cfg.get("defensive_tilt", 0.10))
            for ds in defensive_assets:
                if ds in pre_risk.index:
                    pre_risk.loc[ds] = pre_risk.loc[ds] + defensive_tilt / max(1, len(defensive_assets))
            # Re-normalize after defensive tilt.
            gross_now = float(pre_risk.abs().sum())
            gross_lim = float(gross_override) if gross_override is not None else float(self.cfg["portfolio"]["gross_limit"])
            if gross_now > gross_lim and gross_now > 1e-12:
                pre_risk = pre_risk / gross_now * gross_lim
        regime = regime_snapshot.regime
        final_weights, risk_flags = self._stage_call(
            "risk",
            run_id,
            lambda: self.risk.run(pre_risk, window, regime=regime, benchmark_symbol=bench_symbol),
            lambda: (pd.Series(0.0, index=symbols), ["risk_stage_failed_safe_zero"]),
            trace_id=trace_id,
            parent_span_id=root_span_id,
        )

        if bool(self.cfg.get("portfolio", {}).get("long_only", False)):
            final_weights = final_weights.clip(lower=0.0)
            gross_lo = float(final_weights.sum())
            gross_cap = float(min(1.0, self.cfg.get("portfolio", {}).get("gross_limit", 1.0)))
            if gross_lo > 1e-12:
                final_weights = final_weights / gross_lo * gross_cap
                risk_flags = risk_flags + ["long_only_enforced"]
        _dbg("risk_manager_output", final_weights)
        # --- Bayesian weight blend (additive, runs only if enabled) ---
        bayes_cfg = self.cfg.get("bayesian_optimizer", {})
        if bool(bayes_cfg.get("enabled", False)):
            try:
                signals = {k: float(v) for k, v in combined.reindex(symbols).fillna(0.0).to_dict().items()}
                prices_df = window.copy()
                bayes_result = run_bayesian_optimization(signals, prices_df, self.cfg)
                if bayes_result and "mean_weights" in bayes_result:
                    bayes_blend = float(bayes_cfg.get("blend_weight", 0.25))
                    for ticker in list(final_weights.index):
                        if ticker in bayes_result["mean_weights"]:
                            diff = abs(float(bayes_result["mean_weights"][ticker]) - float(final_weights[ticker]))
                            if diff < 0.20:  # Safety guard: only blend if difference is small
                                final_weights[ticker] = (
                                    (1 - bayes_blend) * float(final_weights[ticker]) +
                                    bayes_blend * float(bayes_result["mean_weights"][ticker])
                                )
                    self.audit.append(
                        "bayesian_optimizer_layer",
                        run_id,
                        {
                            "enabled": True,
                            "blend_weight": bayes_blend,
                            "n_assets": len(bayes_result.get("mean_weights", {})),
                        },
                    )
            except Exception:
                pass  # Never let bayesian layer break existing flow
        # --- End Bayesian blend ---
        controls = self.cfg.get("execution_controls", {})
        min_delta = float(controls.get("min_weight_change_to_trade", 0.02))
        max_turnover = float(controls.get("max_turnover_per_cycle", 0.40))
        cooldown = int(controls.get("rebalance_cooldown_cycles", 1))

        prev = self._last_final_weights.reindex(symbols).fillna(0.0) if not self._last_final_weights.empty else pd.Series(0.0, index=symbols)
        raw_delta = final_weights - prev
        raw_turnover = float(raw_delta.abs().sum())
        if (self._cycle_count - self._last_rebalance_cycle) <= cooldown and raw_turnover > 0:
            final_weights = prev.copy()
            risk_flags = risk_flags + ["rebalance_cooldown_hold"]
        else:
            # Ignore tiny changes to reduce noise churn.
            delta = raw_delta.where(raw_delta.abs() >= min_delta, 0.0)
            final_weights = prev + delta
            turnover = float(delta.abs().sum())
            prev_gross = float(prev.abs().sum())
            if prev_gross >= 0.05:
                if turnover > max_turnover and turnover > 1e-12:
                    scale = max_turnover / turnover
                    final_weights = prev + delta * scale
                    risk_flags = risk_flags + ["turnover_capped"]
            if float(raw_delta.abs().sum()) > 0 and np.allclose(final_weights.values, prev.values):
                risk_flags = risk_flags + ["trade_threshold_hold"]
            self._last_rebalance_cycle = self._cycle_count
        _dbg("post_execution_controls", final_weights)

        if self.cfg.get("portfolio", {}).get("long_only", False):
            final_weights = final_weights.clip(lower=0.0)
            gross = float(final_weights.sum())
            target_gross = float(self.cfg.get("portfolio", {}).get("gross_limit", 1.0))
            if gross > 1e-12:
                final_weights = final_weights / gross * min(target_gross, gross)
                risk_flags = risk_flags + ["long_only_enforced_post_controls"]

        self._last_final_weights = final_weights.copy()
        self._cycle_count += 1
        pnl_drift = self._check_pnl_drift(window, final_weights)
        drift_threshold = float(self.cfg.get("alerts", {}).get("thresholds", {}).get("daily_pnl_drift", 0.03))
        if pnl_drift > drift_threshold:
            risk_flags = risk_flags + ["pnl_drift_alert"]
            self.alerts.notify("pnl_drift_alert", {"run_id": run_id, "value": pnl_drift})

        decision = DecisionCycle(
            run_id=run_id,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            symbols=symbols,
            target_weights={k: float(v) for k, v in final_weights.round(6).to_dict().items()},
            risk_flags=risk_flags,
            model_versions={
                "research_agent": "v1",
                "strategy_ensemble": "v1",
                "fund_manager": "v1",
                "risk_manager": "v1",
            },
        )
        self.audit.append("decision", run_id, decision.to_dict())
        if not self.audit.verify_tail(last_n=30):
            self.audit.append("audit_anomaly", run_id, {"issue": "hash_chain_verification_failed"})
            self.alerts.notify("audit_anomaly", {"run_id": run_id})

        if execute:
            exec_span_id = self.tracer.start_span(
                trace_id,
                root_span_id,
                name="execution",
                metadata={"run_id": run_id},
            )
            broker = self._execution_broker()
            latest_prices = prices.iloc[-1]
            broker_used = broker.submit_target_weights(final_weights, latest_prices, run_id=run_id)
            self.audit.append("execution_submitted", run_id, {"broker": broker_used})
            self.health.beat(run_id, "execution_submitted")
            self.tracer.finish_span(
                trace_id,
                exec_span_id,
                metadata_update={"broker": broker_used, "orders": "submitted"},
            )
        else:
            self.health.beat(run_id, "decision_only")

        (self.out_dir / "last_decision.json").write_text(
            json.dumps(decision.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.tracer.finalize_trace(
            trace_id,
            root_span_id,
            metadata_update={
                "risk_flags": decision.risk_flags,
                "target_weights": decision.target_weights,
            },
        )
        return decision

    def run_realtime(self, poll_seconds: int, execute: bool, max_cycles: int | None = None) -> None:
        import time

        counter = 0
        while True:
            if self.health.is_stale():
                stale_id = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
                self.audit.append("deadman_switch", stale_id, {"action": "detected_stale_heartbeat"})
                self.alerts.notify("deadman_switch_triggered", {"run_id": stale_id})
            self.run_cycle(execute=execute)
            counter += 1
            if max_cycles is not None and counter >= max_cycles:
                break
            time.sleep(poll_seconds)


def run_ai_alpha_layer(prices_df, signals, config, audit_logger=None) -> dict:
    import logging

    if not bool(config.get("ai_alpha", {}).get("enabled", False)):
        return signals
    fast_mode = bool(config.get("backtest", {}).get("fast_mode", False))
    logger = logging.getLogger(__name__)
    macro_snapshot = config.get(
        "macro_snapshot",
        {"vix": 20, "yield_10y": 0.04, "yield_2y": 0.05, "dxy": 103},
    )
    blend = float(config.get("ai_alpha", {}).get("blend_weight", 0.3))
    out = dict(signals)
    alpha_dump: dict = {}
    for ticker in list(out.keys()):
        try:
            existing = out[ticker] if isinstance(out[ticker], float) else out[ticker].get("signal", 0.0)
            if fast_mode:
                series = prices_df[ticker].dropna().astype(float) if ticker in prices_df.columns else pd.Series(dtype=float)
                ret_5d = float(series.pct_change(5).iloc[-1]) if len(series) > 6 else 0.0
                ret_20d = float(series.pct_change(20).iloc[-1]) if len(series) > 21 else ret_5d
                ret_63d = float(series.pct_change(63).iloc[-1]) if len(series) > 64 else ret_20d
                combined_alpha = 0.50 * ret_5d + 0.30 * ret_20d + 0.20 * ret_63d
                alpha = {
                    "combined_alpha": float(combined_alpha),
                    "uncertainty": 0.25,
                    "breakdown": {
                        "momentum_5d": ret_5d,
                        "momentum_20d": ret_20d,
                        "momentum_63d": ret_63d,
                    },
                    "macro_regime": "fast_mode_proxy",
                    "days_to_earnings": None,
                }
            else:
                alpha = combine_advanced_alpha(ticker, prices_df, macro_snapshot)
            out[ticker] = (1.0 - blend) * float(existing) + blend * float(alpha.get("combined_alpha", 0.0))
            alpha_dump[ticker] = alpha
        except Exception as e:
            logger.warning("AI alpha layer failed for %s: %s", ticker, e)
            continue

    if not fast_mode:
        try:
            # Safe default: no headlines provided in this integration path.
            enriched = enrich_signals_with_llm(out, [])
            for ticker, v in enriched.items():
                if ticker in out and isinstance(v, dict):
                    out[ticker] = float(v.get("signal", out[ticker] if isinstance(out[ticker], float) else 0.0))
        except Exception:
            pass

    if audit_logger is not None and hasattr(audit_logger, "append"):
        try:
            payload = {
                "tickers": list(out.keys()),
                "blend_weight": blend,
                "macro_snapshot": macro_snapshot,
                "n_alpha": len(alpha_dump),
            }
            audit_logger.append(event_type="AI_ALPHA_LAYER", run_id=sha256_hex(payload)[:16], payload=payload)
        except Exception:
            pass

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        (out_dir / "ai_alpha_latest.json").write_text(json.dumps(alpha_dump, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        pass
    return out


def run_bayesian_optimization(signals: dict, prices_df: pd.DataFrame, config: dict) -> dict:
    if not bool(config.get("bayesian_optimizer", {}).get("enabled", False)):
        return {}
    try:
        bcfg = config.get("bayesian_optimizer", {})
        optimizer = BayesianPortfolioOptimizer(
            risk_aversion=float(bcfg.get("risk_aversion", 2.5)),
            tau=float(bcfg.get("tau", 0.05)),
        )
        optimizer.set_market_prior(prices_df)
        sorted_signals = sorted(signals.items(), key=lambda kv: abs(float(kv[1])), reverse=True)[:3]
        for ticker, signal_value in sorted_signals:
            optimizer.add_llm_view(
                {"assets": [ticker], "outperformance": float(signal_value), "confidence": 0.6}
            )
        if bool(config.get("backtest", {}).get("fast_mode", False)):
            result = optimizer.optimize()
            result = {
                "mean_weights": result.get("weights", {}),
                "std_weights": {k: 0.0 for k in result.get("weights", {})},
                "p5_weights": {k: float(v) for k, v in result.get("weights", {}).items()},
                "p95_weights": {k: float(v) for k, v in result.get("weights", {}).items()},
                "posterior_cov": result.get("posterior_cov", np.zeros((0, 0))),
            }
        else:
            result = optimizer.optimize_with_uncertainty(
                n_samples=int(bcfg.get("n_uncertainty_samples", 500))
            )
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            serializable = dict(result)
            if isinstance(serializable.get("posterior_cov"), np.ndarray):
                serializable["posterior_cov"] = serializable["posterior_cov"].tolist()
            (out_dir / "bayesian_weights_latest.json").write_text(
                json.dumps(serializable, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except Exception:
            pass
        return result
    except Exception:
        return {}
