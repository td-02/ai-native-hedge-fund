from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Callable, TypeVar
import pandas as pd

from .alerts import AlertManager
from .audit import AuditLedger
from .contracts import DecisionCycle, sha256_hex
from .data import download_close_prices
from .data_quality import DataQualityAgent
from .healthcheck import HealthMonitor
from .paper import AlpacaPaperBroker, PaperBrokerStub
from .regime import MacroRegimeAgent, RegimeSnapshot
from .research import ResearchAgent
from .resilience import BreakerRegistry
from .strategy_stack import FundManagerAgent, RiskManagerAgent, StrategyEnsembleAgent
from .contracts import ResearchSignal

T = TypeVar("T")


@dataclass
class CentralizedHedgeFundSystem:
    cfg: dict

    def __post_init__(self) -> None:
        system_cfg = self.cfg.get("system", {})
        self.out_dir = Path(system_cfg.get("output_dir", "outputs"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLedger(self.out_dir / "audit")
        self.health = HealthMonitor(
            self.out_dir,
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
        )

    def _stage_call(self, stage: str, run_id: str, fn: Callable[[], T], degraded: Callable[[], T]) -> T:
        breaker = self.breakers.get(stage)
        if not breaker.allow_call():
            payload = {"stage": stage, "state": breaker.state, "reason": "breaker_open"}
            self.audit.append("degraded_mode", run_id, payload)
            return degraded()
        try:
            out = fn()
            breaker.on_success()
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
                return degraded()
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
        ecfg = self.cfg.get("execution", {})
        broker_name = ecfg.get("broker", "stub")
        if broker_name == "alpaca_paper":
            return AlpacaPaperBroker(
                base_url=ecfg.get("alpaca_base_url", "https://paper-api.alpaca.markets"),
                min_order_notional=float(ecfg.get("min_order_notional", 10.0)),
                order_style=str(ecfg.get("order_style", "twap")),
                twap_slices=int(ecfg.get("twap_slices", 3)),
                max_participation_adv=float(ecfg.get("max_participation_adv", 0.10)),
                adv_notional_default=float(ecfg.get("adv_notional_default", 2_000_000.0)),
                market_mode=str(ecfg.get("market_mode", "us")),
            )
        return PaperBrokerStub()

    def run_cycle(self, execute: bool = False) -> DecisionCycle:
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
        self.health.beat(run_id, "cycle_start")
        self.audit.append("market_snapshot", run_id, {"rows": len(window), "symbols": symbols})

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
            return decision

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
        )
        self.audit.append("research_output", run_id, {k: v.to_dict() for k, v in research.items()})

        strategy_scores = self._stage_call(
            "strategy",
            run_id,
            lambda: self.strategy.run(window, research),
            lambda: {"trend_following": self.strategy._trend(window).reindex(symbols).fillna(0.0)},
        )
        self.audit.append(
            "strategy_scores",
            run_id,
            {name: score.round(6).to_dict() for name, score in strategy_scores.items()},
        )
        disagreement = self._disagreement(strategy_scores)
        self.audit.append("strategy_disagreement", run_id, {"value": disagreement})
        dis_threshold = float(self.cfg.get("alerts", {}).get("thresholds", {}).get("strategy_disagreement", 0.7))
        if disagreement > dis_threshold:
            self.alerts.notify("strategy_disagreement_high", {"run_id": run_id, "value": disagreement})

        regime_snapshot = self._stage_call(
            "regime",
            run_id,
            lambda: self.regime_agent.run(prefer_india=str(self.cfg.get("execution", {}).get("market_mode", "us")) == "india"),
            lambda: RegimeSnapshot(regime="trend", confidence=0.5, leverage_cap=1.0, risk_multiplier=1.0),
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

        combined = self.strategy.weighted_score(strategy_scores) * regime_snapshot.risk_multiplier
        pre_risk = self.fund_manager.run(combined)
        regime = regime_snapshot.regime
        final_weights, risk_flags = self._stage_call(
            "risk",
            run_id,
            lambda: self.risk.run(pre_risk, window, regime=regime, benchmark_symbol=symbols[0]),
            lambda: (pd.Series(0.0, index=symbols), ["risk_stage_failed_safe_zero"]),
        )
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
            broker = self._execution_broker()
            latest_prices = prices.iloc[-1]
            if isinstance(broker, AlpacaPaperBroker):
                broker.submit_target_weights(final_weights, latest_prices, run_id=run_id)
            else:
                broker.submit_target_weights(final_weights, run_id=run_id)
            self.audit.append("execution_submitted", run_id, {"broker": type(broker).__name__})
            self.health.beat(run_id, "execution_submitted")
        else:
            self.health.beat(run_id, "decision_only")

        (self.out_dir / "last_decision.json").write_text(
            json.dumps(decision.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
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
