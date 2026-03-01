from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

from .agents import SignalAgent, RiskAgent, AllocatorAgent


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    metrics: dict


def _compute_metrics(returns: pd.Series) -> dict:
    returns = returns.dropna()
    if returns.empty:
        return {}

    equity = (1 + returns).cumprod()
    cagr = equity.iloc[-1] ** (252 / len(returns)) - 1
    vol = returns.std(ddof=0) * np.sqrt(252)
    sharpe = (returns.mean() * 252) / (vol + 1e-12)
    drawdown = equity / equity.cummax() - 1
    max_dd = drawdown.min()

    return {
        'total_return': float(equity.iloc[-1] - 1),
        'cagr': float(cagr),
        'annual_vol': float(vol),
        'sharpe': float(sharpe),
        'max_drawdown': float(max_dd),
    }


def run_backtest(prices: pd.DataFrame, cfg: dict) -> BacktestResult:
    pcfg = cfg['portfolio']
    ccfg = cfg['costs']
    rcfg = cfg['risk']
    acfg = cfg['agent']

    signal = SignalAgent(
        lookback_days=pcfg['lookback_days'],
        enable_ollama_overlay=acfg['enable_ollama_overlay'],
        ollama_url=acfg['ollama_url'],
        ollama_model=acfg['ollama_model'],
    )
    risk = RiskAgent(
        target_annual_vol=rcfg['target_annual_vol'],
        min_annual_vol_floor=rcfg['min_annual_vol_floor'],
    )
    alloc = AllocatorAgent(
        max_weight=pcfg['max_weight'],
        gross_limit=pcfg['gross_limit'],
    )

    lookback = pcfg['lookback_days']
    rebalance_n = pcfg['rebalance_every_n_days']

    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    current_w = pd.Series(0.0, index=prices.columns)

    for i in range(lookback, len(prices) - 1):
        if (i - lookback) % rebalance_n == 0:
            window = prices.iloc[i - lookback:i]
            s = signal.run(window)
            r = risk.run(s, window)
            current_w = alloc.run(r)
        weights.iloc[i + 1] = current_w

    asset_rets = prices.pct_change().fillna(0.0)
    gross_turnover = (weights - weights.shift(1).fillna(0.0)).abs().sum(axis=1)

    fee = ccfg['transaction_cost_bps'] / 10000.0
    slip = ccfg['slippage_bps'] / 10000.0
    trading_cost = gross_turnover * (fee + slip)

    strat_returns = (weights.shift(1).fillna(0.0) * asset_rets).sum(axis=1) - trading_cost
    equity_curve = (1 + strat_returns).cumprod()

    metrics = _compute_metrics(strat_returns)
    metrics['avg_daily_turnover'] = float(gross_turnover.mean())

    return BacktestResult(equity_curve=equity_curve, returns=strat_returns, weights=weights, metrics=metrics)


def save_results(result: BacktestResult, out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    result.equity_curve.to_csv(out / 'equity_curve.csv', header=['equity'])
    result.returns.to_csv(out / 'returns.csv', header=['returns'])
    result.weights.to_csv(out / 'weights.csv')

    metrics_df = pd.DataFrame([result.metrics])
    metrics_df.to_csv(out / 'metrics.csv', index=False)
