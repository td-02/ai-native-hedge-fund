from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import pandas as pd

from .audit import AuditLedger
from .contracts import DecisionCycle, sha256_hex
from .data import download_close_prices
from .paper import AlpacaPaperBroker, PaperBrokerStub
from .research import ResearchAgent
from .strategy_stack import FundManagerAgent, RiskManagerAgent, StrategyEnsembleAgent


@dataclass
class CentralizedHedgeFundSystem:
    cfg: dict

    def __post_init__(self) -> None:
        system_cfg = self.cfg.get("system", {})
        self.out_dir = Path(system_cfg.get("output_dir", "outputs"))
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.audit = AuditLedger(self.out_dir / "audit")

        acfg = self.cfg["agent"]
        scfg = self.cfg["strategies"]
        rcfg = self.cfg["risk_hard_limits"]
        pcfg = self.cfg["portfolio"]

        self.research = ResearchAgent(
            enable_llm=bool(acfg.get("enable_llm_research", False)),
            ollama_model=acfg.get("ollama_model", "llama3.1:8b"),
            ollama_url=acfg.get("ollama_url", "http://localhost:11434/api/generate"),
            max_headlines=int(acfg.get("max_headlines", 8)),
        )
        self.strategy = StrategyEnsembleAgent(strategy_weights=scfg["weights"])
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
        )

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
        self.audit.append("market_snapshot", run_id, {"rows": len(window), "symbols": symbols})

        research = self.research.run(symbols)
        self.audit.append("research_output", run_id, {k: v.to_dict() for k, v in research.items()})

        strategy_scores = self.strategy.run(window, research)
        self.audit.append(
            "strategy_scores",
            run_id,
            {name: score.round(6).to_dict() for name, score in strategy_scores.items()},
        )

        combined = self.strategy.weighted_score(strategy_scores)
        pre_risk = self.fund_manager.run(combined)
        final_weights, risk_flags = self.risk.run(pre_risk, window)

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

        if execute:
            broker = self._execution_broker()
            latest_prices = prices.iloc[-1]
            if isinstance(broker, AlpacaPaperBroker):
                broker.submit_target_weights(final_weights, latest_prices)
            else:
                broker.submit_target_weights(final_weights)
            self.audit.append("execution_submitted", run_id, {"broker": type(broker).__name__})

        (self.out_dir / "last_decision.json").write_text(
            json.dumps(decision.to_dict(), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        return decision

    def run_realtime(self, poll_seconds: int, execute: bool, max_cycles: int | None = None) -> None:
        import time

        counter = 0
        while True:
            self.run_cycle(execute=execute)
            counter += 1
            if max_cycles is not None and counter >= max_cycles:
                break
            time.sleep(poll_seconds)
