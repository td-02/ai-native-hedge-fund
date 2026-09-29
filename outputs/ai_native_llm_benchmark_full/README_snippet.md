### Setup

- Model: `llama3.1:8b` served locally by Ollama 0.34.4 on Apple M5 (24.0 GB).
- Window: price data 2020-01-01 to 2026-03-01; one decision every 5 trading days, 259 decision cycles from 2021-01-05 to 2026-02-25; universe SPY, QQQ, IWM, TLT, GLD.
- Config: `configs/ai_native_llm.yaml` (LLM forecasts on, research council on, full orchestrator).

### Backtest comparison

| Variant | Sharpe | CAGR | Vol | Max DD | Total return | Avg turnover |
|---|---|---|---|---|---|---|
| Baseline orchestrator (no v2 overlay) | 0.01 | -0.3% | 8.1% | -10.3% | -0.3% | 0.09 |
| AI-native v2, LLM forecasts (Ollama) | -0.07 | -0.9% | 8.1% | -10.6% | -0.9% | 0.11 |
| AI-native v2, deterministic fallback | -0.12 | -1.3% | 8.1% | -10.8% | -1.3% | 0.11 |
| SPY | 1.71 | +37.3% | 19.7% | -14.3% | +38.5% | n/a |
| Equal weight | 2.38 | +41.5% | 15.1% | -9.3% | +42.9% | n/a |

### LLM usage (proof the AI path ran)

| Variant | LLM calls | Answered by LLM | Forecast source | Provider | Latency p50 / p95 | Wall clock |
|---|---|---|---|---|---|---|
| AI-native v2, LLM forecasts (Ollama) | 1295 | 1295 | llm: 1295 | ollama: 1295 | 3.08 s / 3.68 s | 67.3 min |
| AI-native v2, deterministic fallback | 0 | 0 | deterministic_fallback: 1295 | none | 0.00 s / 0.00 s | 0.8 min |
