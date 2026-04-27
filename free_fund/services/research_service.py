from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from free_fund.audit import AuditLedger
from free_fund.contracts import ResearchSignal, sha256_hex
from free_fund.data import download_close_prices
from free_fund.research import ResearchAgent


def _make_run_id(window: pd.DataFrame) -> str:
    snapshot = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "symbols": list(window.columns),
        "tail_rows_json": window.tail(3).round(6).to_json(date_format="iso", orient="split"),
    }
    return sha256_hex(snapshot)[:16]


def run_research_stage(cfg: dict) -> dict:
    pcfg = cfg["portfolio"]
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

    run_id = _make_run_id(window)
    acfg = cfg["agent"]
    agent = ResearchAgent(
        enable_llm=bool(acfg.get("enable_llm_research", False)),
        ollama_model=acfg.get("ollama_model", "llama3.1:8b"),
        ollama_url=acfg.get("ollama_url", "http://localhost:11434/api/generate"),
        max_headlines=int(acfg.get("max_headlines", 8)),
        max_news_age_minutes=int(acfg.get("max_news_age_minutes", 15)),
        max_retries=int(acfg.get("max_retries", 2)),
        retry_base_delay_sec=float(acfg.get("retry_base_delay_sec", 0.4)),
    )
    research = agent.run(symbols)
    payload = {
        "run_id": run_id,
        "symbols": symbols,
        "research": {k: v.to_dict() for k, v in research.items()},
        "window_tail": window.tail(5).to_json(date_format="iso", orient="split"),
    }
    AuditLedger().append("research_stage", run_id, payload)
    return payload

