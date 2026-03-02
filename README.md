# AI-Native Hedge Fund Prototype (Free + Portable)

Centralized multi-agent trading system with deterministic risk/execution logic, LangChain-enabled research, and audit-ready event logs.

## Architecture
Data ingest -> Research Agent -> Strategy Ensemble -> Fund Manager -> Risk Manager -> Execution -> Audit Ledger

## Agents
- Market/Data: pulls OHLCV from yfinance.
- Research Agent: RSS headline analysis (deterministic), optional LangChain + local Ollama overlay.
- Strategy Agents: trend, mean reversion, volatility carry, regime switching, event-driven.
- Alpha Pipeline Agents: earnings momentum, FII/DII proxy flow, options-flow proxy, short-interest proxy, block-deal proxy.
- Cross-Asset Arbitrage Agents: NSE/BSE arb hook, cash-futures basis, ETF NAV arb, ADR arb hook.
- Macro Intelligence Agents: RBI policy hook, global carry, crude-gold correlation, rupee regime.
- Research Council (LLM-native): researcher/news/peer/synthesis multi-agent ranking.
- Adaptive Learning Layer: Bayesian-style weight drift and decay updates.
- Private Data Alpha Hooks: bulk deals, margin pressure, F&O lot changes, corp-action arb.
- Fund Manager: combines strategy scores into target weights.
- Risk Manager: hard clamps, volatility scaling, drawdown brake.
- Execution Agent: broker failover router (Alpaca -> Zerodha/Upstox/Angel hooks -> stub), TWAP/VWAP-style slicing, impact and session guards.
- Audit Agent: hash-linked event log for reproducibility.
- Resilience Layer: circuit breakers, retries/backoff, degraded mode, dead-man heartbeat.

## Determinism and Auditability
- Strict structured outputs and fixed strategy blend weights.
- Hash-chained immutable JSONL audit log: `outputs/audit/events.jsonl`.
- Latest decision snapshot: `outputs/last_decision.json`.
- Heartbeat file for dead-man switch: `outputs/heartbeat.json`.

## Reliability Controls
- Circuit breakers per stage (`research`, `strategy`, `regime`, `risk`).
- Data quality gate (staleness, NaN ratio, return outliers, invalid prices).
- Risk extensions: VaR/ES, concentration caps, beta-neutrality band, jump-risk brake, regime leverage.
- Alerts: Slack webhook option for stage failures, disagreement spikes, PnL drift, dead-man triggers.
- Execution guards: TWAP slicing, ADV impact cap, holiday/session checks, market-mode compliance guard.

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

Paper-trading server entrypoint:
```bash
python scripts/live.py --config configs/live_stub.yaml --poll-seconds 300 --max-cycles 0
```

## Backtest
```bash
python scripts/run_backtest.py --config configs/default.yaml
```

## Health Check
```bash
python scripts/healthcheck.py
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

## Oracle Always Free (24/7)
Use the Oracle deployment bundle:
- `deploy/oracle/README.md`
- `deploy/oracle/install.sh`

Quick start on VM:
```bash
chmod +x deploy/oracle/install.sh
./deploy/oracle/install.sh
```

## Notes
- Free/open-source only. No paid API required.
- Keep `execution.broker: stub` during testing.
- For LLM research, set `agent.enable_llm_research: true` and run local Ollama.

## Tests
```bash
pytest -q
```
