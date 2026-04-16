from __future__ import annotations

from celery import Celery

from .config import load_settings

settings = load_settings()
cfg = settings.model_dump()
redis_url = cfg.get("redis", {}).get("url", "redis://localhost:6379/0")

celery_app = Celery("ainhf", broker=redis_url, backend=redis_url)
celery_app.conf.update(
    task_acks_late=True,
    task_default_retry_delay=3,
    task_routes={
        "free_fund.tasks.execution_worker": {"queue": "execution_high_priority"},
        "free_fund.tasks.research_worker": {"queue": "research"},
        "free_fund.tasks.strategy_worker": {"queue": "strategy"},
    },
)
celery_app.autodiscover_tasks(["free_fund"])

