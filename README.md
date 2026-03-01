# AI-Native Hedge Fund Prototype (100% Free)

A zero-paid, local-first side project scaffold for multi-agent portfolio simulation.

## What this includes
- Free market data via `yfinance`
- Local optional LLM routing through Ollama (`http://localhost:11434`)
- 3-agent flow: signal -> risk -> allocator
- Daily backtest with costs/slippage
- Streamlit dashboard for equity + weights + metrics
- Paper-trading stub file for Alpaca paper extension

## Project layout
- `free_fund/` core logic
- `scripts/run_backtest.py` run and save backtest results
- `app/streamlit_app.py` dashboard
- `configs/default.yaml` settings
- `outputs/` generated CSV results

## Setup
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Optional local LLM (free)
```bash
ollama pull llama3.1:8b
ollama serve
```

## Run backtest
```bash
python scripts/run_backtest.py --config configs/default.yaml
```

## Launch dashboard
```bash
streamlit run app/streamlit_app.py
```

## Run daily paper rebalance
Default is local stub (prints target weights):
```bash
python scripts/run_daily.py
```

Use Alpaca free paper account:
1. Create `.env` from example and fill keys:
```bash
copy .env.example .env
```
2. Or set credentials in your shell:
```bash
set APCA_API_KEY_ID=your_key
set APCA_API_SECRET_KEY=your_secret
```
3. In `configs/default.yaml`, set:
```yaml
execution:
  broker: alpaca_paper
```
4. Run:
```bash
python scripts/run_daily.py
```

## Notes
- Default strategy is fully free and works without any paid API.
- If Ollama is unavailable, the system auto-falls back to deterministic signals.
- This is for research/paper trading only.
