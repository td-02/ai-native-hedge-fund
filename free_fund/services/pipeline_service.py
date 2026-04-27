from __future__ import annotations

from datetime import datetime, timezone

from free_fund.contracts import DecisionCycle
from free_fund.services.execution_service import run_execution_stage
from free_fund.services.research_service import run_research_stage
from free_fund.services.strategy_service import run_strategy_stage


def run_decision_pipeline(cfg: dict, execute: bool = False) -> DecisionCycle:
    research_payload = run_research_stage(cfg)
    strategy_payload = run_strategy_stage(cfg, research_payload)
    run_id = str(strategy_payload.get("run_id", research_payload.get("run_id", "")))
    symbols = list(research_payload.get("symbols", cfg.get("portfolio", {}).get("symbols", [])))
    weights = dict(strategy_payload.get("weights", {}))
    risk_flags = list(strategy_payload.get("risk_flags", []))

    if execute:
        exec_payload = run_execution_stage(cfg, strategy_payload)
        if exec_payload.get("status") != "submitted":
            risk_flags.append("execution_not_submitted")

    return DecisionCycle(
        run_id=run_id,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        symbols=symbols,
        target_weights={k: float(v) for k, v in weights.items()},
        risk_flags=risk_flags,
        model_versions={
            "runtime": "service_pipeline_v1",
            "research": "service_v1",
            "strategy": "service_v1",
            "execution": "service_v1",
        },
    )

