from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from free_fund.audit import AuditLedger
from free_fund.brokers import build_broker_router
from free_fund.contracts import sha256_hex
from free_fund.data import download_close_prices
from free_fund.healthcheck import HealthMonitor


def _make_run_id(window: pd.DataFrame) -> str:
    snapshot = {
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "symbols": list(window.columns),
        "tail_rows_json": window.tail(3).round(6).to_json(date_format="iso", orient="split"),
    }
    return sha256_hex(snapshot)[:16]


def run_execution_stage(cfg: dict, weights_payload: dict) -> dict:
    pcfg = cfg["portfolio"]
    symbols = list(pcfg["symbols"])
    prices = download_close_prices(
        symbols=symbols,
        start_date=pcfg["start_date"],
        end_date=pcfg["end_date"],
    )
    window = prices.tail(int(pcfg["lookback_days"]))
    run_id = _make_run_id(window)
    weights = pd.Series(weights_payload.get("weights", {}), dtype=float).reindex(symbols).fillna(0.0)
    latest_prices = window.iloc[-1]
    broker = build_broker_router(cfg)
    broker_used = broker.submit_target_weights(weights, latest_prices, run_id=run_id)

    payload = {"run_id": run_id, "broker": broker_used, "status": "submitted"}
    AuditLedger().append("execution_stage", run_id, payload)
    HealthMonitor(deadman_timeout_sec=int(cfg.get("health", {}).get("deadman_timeout_sec", 900))).beat(
        run_id, "execution_submitted"
    )
    return payload

