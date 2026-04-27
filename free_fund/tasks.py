from __future__ import annotations

from celery import Task

from .celery_app import celery_app
from .config import load_config
from .services import run_execution_stage, run_research_stage, run_strategy_stage


class RetryableTask(Task):
    autoretry_for = (Exception,)
    max_retries = 3
    retry_backoff = True
    retry_jitter = True
    acks_late = True


@celery_app.task(bind=True, base=RetryableTask, name="free_fund.tasks.research_worker", queue="research")
def research_worker(self, config_path: str = "configs/default.yaml") -> dict:
    cfg = load_config(config_path)
    return run_research_stage(cfg)


@celery_app.task(bind=True, base=RetryableTask, name="free_fund.tasks.strategy_worker", queue="strategy")
def strategy_worker(self, research_payload: dict, config_path: str = "configs/default.yaml") -> dict:
    cfg = load_config(config_path)
    return run_strategy_stage(cfg, research_payload)


@celery_app.task(
    bind=True,
    base=RetryableTask,
    name="free_fund.tasks.execution_worker",
    queue="execution_high_priority",
)
def execution_worker(self, weights_payload: dict, config_path: str = "configs/default.yaml") -> dict:
    cfg = load_config(config_path)
    return run_execution_stage(cfg, weights_payload)
