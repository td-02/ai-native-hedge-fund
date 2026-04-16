# AI-Native Hedge Fund (Enterprise Runtime)

Production-oriented Python 3.11 trading runtime with:
- FastAPI service endpoints
- Celery/Redis worker pipeline
- SQLAlchemy + Alembic persistence
- Pydantic Settings v2 config layering
- TraceLM tracing + Prometheus metrics
- Typer CLI entrypoint

This repository supports research, strategy evaluation, backtest flows, and paper execution. It is for paper trading/research only, not financial advice.

## Stack

- Runtime: Python 3.11
- Package manager: `uv`
- App/API: FastAPI + Uvicorn
- Queue: Celery 5 + Redis
- DB: PostgreSQL + TimescaleDB extension (for OHLCV)
- ORM/migrations: SQLAlchemy 2 + Alembic
- Config: Pydantic Settings v2 (`__` nested env overrides)
- Observability: structlog, Prometheus client, TraceLM

## Project Entry Points

- CLI: `ainhf` (defined in `pyproject.toml`)
- API module: `app/api.py`
- Celery app: `free_fund/celery_app.py`
- Celery tasks: `free_fund/tasks.py`

## Quick Start

```bash
git clone https://github.com/td-02/ai-native-hedge-fund.git
cd ai-native-hedge-fund

# Install with uv
uv sync --all-extras
cp .env.example .env
```

Run one decision cycle:

```bash
uv run ainhf run --config configs/default.yaml
```

Run API:

```bash
uv run ainhf api --host 0.0.0.0 --port 8000
```

Run workers:

```bash
uv run ainhf worker --queues research,strategy,execution_high_priority --concurrency 1
```

## API

- `GET /healthz`: heartbeat and stale-state check
- `GET /metrics`: Prometheus metrics
- `POST /decision`: run one decision cycle

Example:

```bash
curl -X POST "http://127.0.0.1:8000/decision?config_path=configs/default.yaml&execute=false"
```

## Worker Pipeline

Celery tasks are split into:
- `research_worker`
- `strategy_worker`
- `execution_worker`

Execution runs on isolated queue:
- `execution_high_priority`

Task behavior:
- `acks_late=True`
- `max_retries=3`

## Configuration

Primary loader: `free_fund/config.py` via `AppSettings`.

Nested env overrides use `__`, e.g.:

```bash
export DATABASE__URL="postgresql+psycopg://postgres:postgres@localhost:5432/ainhf"
export REDIS__URL="redis://localhost:6379/0"
export PORTFOLIO__LOOKBACK_DAYS=252
```

YAML remains supported for baseline config files in `configs/`.

## Database & Migrations

Alembic config:
- `alembic.ini`
- `alembic/env.py`
- `alembic/versions/20260416_0001_enterprise_schema.py`

Core tables:
- `audit_events`
- `ohlcv` (hypertable target in Postgres/Timescale)

Apply migration:

```bash
uv run alembic upgrade head
```

## Observability

- Structured logs: `free_fund/logging.py` (structlog)
- Metrics: `free_fund/metrics.py`
  - `orders_total`
  - `order_latency_seconds`
- Tracing: `free_fund/tracing.py` (hard TraceLM dependency)

## Brokers

Adapter abstraction in `free_fund/brokers.py`:
- `BrokerAdapter` (ABC)
- `AlpacaPaperBroker`
- `ZerodhaAdapter` (`kiteconnect`)
- `AngelOneAdapter` (`smartapi-python`)
- `PaperBrokerStub` (safe default)

## Feature Flags

Redis-backed flags in `free_fund/flags.py`:
- key pattern: `ainhf:flags:<flag>`
- helper: `feature_enabled(flag: str) -> bool`

## Testing & CI

Tests include:
- Celery registration checks
- VCR cassette replay sample
- Hypothesis property tests for risk layer
- Audit chain DB tamper detection

Run tests:

```bash
uv run pytest
```

Coverage gate:
- `--cov-fail-under=80` via `pyproject.toml`

GitHub workflows:
- `.github/workflows/ci.yml`
- `.github/workflows/india-market-paper.yml`

## Legacy Scripts

Legacy scripts in `scripts/` are still available for backtest and diagnostics but now run without `sys.path` injection hacks.

