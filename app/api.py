from __future__ import annotations

from fastapi import FastAPI, HTTPException, Response

from free_fund.config import load_config
from free_fund.healthcheck import HealthMonitor
from free_fund.metrics import render_metrics
from free_fund.orchestrator import CentralizedHedgeFundSystem

app = FastAPI(title="AINHF API", version="0.2.0")


@app.get("/healthz")
def healthz() -> dict:
    monitor = HealthMonitor()
    last = monitor.last()
    return {
        "ok": not monitor.is_stale(),
        "stale": monitor.is_stale(),
        "heartbeat": last,
    }


@app.get("/metrics")
def metrics() -> Response:
    payload, content_type = render_metrics()
    return Response(content=payload, media_type=content_type)


@app.post("/decision")
def decision(config_path: str = "configs/default.yaml", execute: bool = False) -> dict:
    try:
        cfg = load_config(config_path)
        out = CentralizedHedgeFundSystem(cfg).run_cycle(execute=execute)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return out.to_dict()

