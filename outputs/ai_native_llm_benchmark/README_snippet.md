### Setup

- Model: `llama3.1:8b` served locally by Ollama 0.34.4 on Apple M5 (24.0 GB).
- Window: price data 2020-01-01 to 2026-03-01; one decision every 5 trading days, 60 decision cycles from 2021-01-05 to 2022-03-08; universe SPY, QQQ, IWM, TLT, GLD.
- Config: `configs/ai_native_llm.yaml` (LLM forecasts on, research council on, full orchestrator).

### Backtest comparison

| Variant | Sharpe | CAGR | Vol | Max DD | Total return | Avg turnover |
|---|---|---|---|---|---|---|
| Baseline orchestrator (no v2 overlay) | 2.60 | +16.7% | 6.0% | -2.1% | +3.7% | 0.09 |
| AI-native v2, LLM forecasts (Ollama) | 2.40 | +15.6% | 6.1% | -2.3% | +3.5% | 0.11 |
| AI-native v2, deterministic fallback | 2.49 | +16.1% | 6.1% | -2.1% | +3.6% | 0.11 |
| SPY | 0.94 | +12.8% | 13.9% | -6.8% | +2.9% | n/a |
| Equal weight | 0.62 | +7.0% | 12.1% | -7.0% | +1.6% | n/a |

### LLM usage (proof the AI path ran)

| Variant | LLM calls | Answered by LLM | Forecast source | Provider | Latency p50 / p95 | Wall clock |
|---|---|---|---|---|---|---|
| AI-native v2, LLM forecasts (Ollama) | 300 | 300 | llm: 300 | ollama: 300 | 3.09 s / 3.41 s | 15.3 min |
| AI-native v2, deterministic fallback | 0 | 0 | deterministic_fallback: 300 | none | 0.00 s / 0.00 s | 0.2 min |

### Live decision cycle with the LLM research council

15 council calls (3 roles x 5 symbols), 15 answered by the LLM, latency p50 3.31 s, cycle wall clock 55 s with council vs 2 s without.

| Symbol | Researcher | News analyst | Peer reviewer | Council score | Weight, council on | Weight, council off |
|---|---|---|---|---|---|---|
| SPY | +0.20 | +0.60 | +0.70 | +0.500 | 0.233 | 0.235 |
| QQQ | +0.30 | +0.40 | +0.60 | +0.433 | 0.129 | 0.128 |
| IWM | -0.50 | +0.20 | +0.20 | -0.033 | 0.035 | 0.036 |
| TLT | -0.80 | -0.80 | -0.80 | -0.800 | -0.209 | -0.206 |
| GLD | -0.70 | -0.50 | -0.60 | -0.600 | -0.195 | -0.194 |
