# AI-Native Hedge Fund Prototype (Free + Portable)

Centralized multi-agent trading system with deterministic risk/execution logic, LangChain-enabled research, and audit-ready event logs.

## Architecture
Data ingest -> Research Agent -> Strategy Ensemble -> Fund Manager -> Risk Manager -> Execution -> Audit Ledger

## Agents
- Market/Data: pulls OHLCV from yfinance.
- Research Agent: RSS headline analysis (deterministic), optional LangChain + local Ollama overlay.
- Strategy Agents: trend, mean reversion, volatility carry, regime switching, event-driven.
- Fund Manager: combines strategy scores into target weights.
- Risk Manager: hard clamps, volatility scaling, drawdown brake.
- Execution Agent: Alpaca paper or local stub.
- Audit Agent: hash-linked event log for reproducibility.

## Determinism and Auditability
- Strict structured outputs and fixed strategy blend weights.
- Hash-chained immutable JSONL audit log: `outputs/audit/events.jsonl`.
- Latest decision snapshot: `outputs/last_decision.json`.

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Environment
Create `.env` from template and set keys for Alpaca paper execution:
```bash
copy .env.example .env
```

Required for live paper execution:
- `APCA_API_KEY_ID`
- `APCA_API_SECRET_KEY`

Optional:
- `APCA_PAPER_BASE_URL` (defaults to `https://paper-api.alpaca.markets`)

## Run Single Decision Cycle
Dry run (no orders):
```bash
python scripts/run_daily.py --dry-run
```

Paper execution:
```bash
python scripts/run_daily.py
```

## Run Realtime Loop
Dry loop:
```bash
python scripts/run_realtime.py --poll-seconds 300 --max-cycles 0
```

Paper execution loop:
```bash
python scripts/run_realtime.py --execute --poll-seconds 300 --max-cycles 0
```

## Backtest
```bash
python scripts/run_backtest.py --config configs/default.yaml
```

## Dashboard
```bash
streamlit run app/streamlit_app.py
```

Shows:
- Backtest metrics/equity.
- Latest live decision.
- Audit event tail.

## Portable Deployment (Docker)
```bash
docker compose up --build
```

## Notes
- Free/open-source only. No paid API required.
- Keep `execution.broker: stub` during testing.
- For LLM research, set `agent.enable_llm_research: true` and run local Ollama.
