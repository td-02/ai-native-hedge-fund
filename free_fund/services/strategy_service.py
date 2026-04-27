from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from free_fund.audit import AuditLedger
from free_fund.contracts import ResearchSignal, sha256_hex
from free_fund.data import download_close_prices
from free_fund.strategy_stack import FundManagerAgent, RiskManagerAgent, StrategyEnsembleAgent


def _make_run_id(window: pd.DataFrame) -> str:
    snapshot = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "symbols": list(window.columns),
        "tail_rows_json": window.tail(3).round(6).to_json(date_format="iso", orient="split"),
    }
    return sha256_hex(snapshot)[:16]


def run_strategy_stage(cfg: dict, research_payload: dict) -> dict:
    pcfg = cfg["portfolio"]
    symbols = list(pcfg["symbols"])
    prices = download_close_prices(
        symbols=symbols,
        start_date=pcfg["start_date"],
        end_date=pcfg["end_date"],
    )
    window = prices.tail(int(pcfg["lookback_days"]))
    run_id = _make_run_id(window)

    research = {
        symbol: ResearchSignal(**value)
        for symbol, value in (research_payload.get("research", {}) or {}).items()
    }

    strategy_cfg = cfg["strategies"]
    ensemble = StrategyEnsembleAgent(strategy_weights=strategy_cfg["weights"])
    strategy_scores = ensemble.run(window, research)
    combined = ensemble.weighted_score(strategy_scores)

    fund = FundManagerAgent(
        max_weight=float(pcfg["max_weight"]),
        gross_limit=float(pcfg["gross_limit"]),
    )
    pre_risk = fund.run(combined_score=combined)

    rcfg = cfg["risk_hard_limits"]
    risk = RiskManagerAgent(
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
    final_weights, risk_flags = risk.run(pre_risk, window, regime="trend", benchmark_symbol=str(symbols[0]))

    payload = {
        "run_id": run_id,
        "weights": {k: float(v) for k, v in final_weights.to_dict().items()},
        "risk_flags": risk_flags,
    }
    AuditLedger().append("strategy_stage", run_id, payload)
    return payload

