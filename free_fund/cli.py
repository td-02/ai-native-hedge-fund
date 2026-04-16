from __future__ import annotations

import typer
import uvicorn
from celery.bin.worker import worker as celery_worker

from .config import load_config
from .healthcheck import HealthMonitor
from .orchestrator import CentralizedHedgeFundSystem

app = typer.Typer(help="AINHF enterprise CLI")


@app.command()
def run(config: str = "configs/default.yaml", execute: bool = False) -> None:
    cfg = load_config(config)
    decision = CentralizedHedgeFundSystem(cfg).run_cycle(execute=execute)
    typer.echo(f"{decision.run_id}")


@app.command()
def backtest(config: str = "configs/default.yaml") -> None:
    cfg = load_config(config)
    decision = CentralizedHedgeFundSystem(cfg).run_cycle(execute=False)
    typer.echo(f"{decision.run_id}")


@app.command()
def healthcheck(deadman_timeout_sec: int = 900) -> None:
    monitor = HealthMonitor(deadman_timeout_sec=deadman_timeout_sec)
    typer.echo("ok" if not monitor.is_stale() else "stale")


@app.command()
def worker(
    queues: str = "research,strategy,execution_high_priority",
    concurrency: int = 1,
) -> None:
    from .celery_app import celery_app

    worker = celery_worker(app=celery_app)
    worker.run(loglevel="INFO", concurrency=concurrency, queues=queues)


@app.command()
def api(host: str = "0.0.0.0", port: int = 8000) -> None:
    uvicorn.run("app.api:app", host=host, port=port, reload=False)

