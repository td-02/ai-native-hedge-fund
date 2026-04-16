from __future__ import annotations


def test_celery_tasks_registered():
    from free_fund.celery_app import celery_app
    from free_fund.tasks import execution_worker, research_worker, strategy_worker

    task_names = set(celery_app.tasks.keys())
    assert research_worker.name in task_names
    assert strategy_worker.name in task_names
    assert execution_worker.name in task_names

