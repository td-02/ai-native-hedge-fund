from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from free_fund.config import load_config
from free_fund.data import download_close_prices
from free_fund.orchestrator import CentralizedHedgeFundSystem


def _metrics(returns: pd.Series) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {}
    equity = (1.0 + returns).cumprod()
    vol = float(returns.std(ddof=0) * math.sqrt(252))
    sharpe = float((returns.mean() * 252) / (vol + 1e-12))
    drawdown = equity / equity.cummax() - 1.0
    cagr = float(equity.iloc[-1] ** (252 / len(returns)) - 1.0)
    return {
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def run_orchestrator_backtest(
    cfg: dict,
    start_date: str,
    end_date: str | None,
    step_days: int,
    max_cycles: int,
    use_prices_override: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, dict[str, float]]:
    pcfg = cfg["portfolio"]
    prices = download_close_prices(
        symbols=list(pcfg["symbols"]),
        start_date=start_date,
        end_date=end_date,
    )
    rets = prices.pct_change().fillna(0.0)

    lookback = int(pcfg.get("lookback_days", 126))
    step_days = max(1, int(step_days))
    decision_rows = list(range(lookback, len(prices) - 1, step_days))
    if max_cycles > 0:
        decision_rows = decision_rows[: max_cycles]

    if not decision_rows:
        raise ValueError("No decision rows available. Increase date range or reduce lookback.")

    weights_hist: list[pd.Series] = []
    gross_turnover: list[float] = []
    returns_hist: list[float] = []
    index_hist: list[pd.Timestamp] = []
    prev_w = pd.Series(0.0, index=prices.columns)

    cfg_rt = copy.deepcopy(cfg)
    cfg_rt.setdefault("data_quality", {})
    cfg_rt["data_quality"]["max_staleness_minutes"] = int(60 * 24 * 365 * 20)
    cfg_rt["portfolio"]["start_date"] = start_date
    cfg_rt["portfolio"]["end_date"] = None
    system = CentralizedHedgeFundSystem(cfg_rt)

    for i in decision_rows:
        dt = prices.index[i]
        system.cfg["portfolio"]["end_date"] = dt.strftime("%Y-%m-%d")
        decision = system.run_cycle(
            execute=False,
            prices_override=prices.iloc[: i + 1] if use_prices_override else None,
        )
        w = pd.Series(decision.target_weights).reindex(prices.columns).fillna(0.0)
        turnover = float((w - prev_w).abs().sum())
        next_ret = float((w * rets.iloc[i + 1]).sum())
        prev_w = w

        weights_hist.append(w)
        gross_turnover.append(turnover)
        returns_hist.append(next_ret)
        index_hist.append(prices.index[i + 1])

    wdf = pd.DataFrame(weights_hist, index=index_hist)
    turnover_s = pd.Series(gross_turnover, index=index_hist, name="gross_turnover")
    strat_returns = pd.Series(returns_hist, index=index_hist, name="returns")

    fee = float(cfg["costs"]["transaction_cost_bps"]) / 10000.0
    slip = float(cfg["costs"]["slippage_bps"]) / 10000.0
    costs = turnover_s * (fee + slip)
    net_returns = strat_returns - costs
    equity = (1.0 + net_returns).cumprod()
    metrics = _metrics(net_returns)
    metrics["avg_turnover"] = float(turnover_s.mean())
    metrics["cycles"] = float(len(index_hist))
    return wdf, strat_returns, net_returns, equity, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--from-date", default=None, help="Override portfolio.start_date")
    parser.add_argument("--to-date", default=None, help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--step-days", type=int, default=5, help="Decision frequency in trading days")
    parser.add_argument("--max-cycles", type=int, default=0, help="0 = all available cycles")
    parser.add_argument("--fast-mode", action="store_true", help="Disable network-heavy stages for fast replay.")
    parser.add_argument("--out", default="outputs/orchestrator_backtest")
    args = parser.parse_args()

    cfg = load_config(args.config)
    cfg.setdefault("backtest", {})
    cfg["backtest"]["fast_mode"] = bool(args.fast_mode)
    pcfg = cfg["portfolio"]
    start_date = args.from_date or pcfg["start_date"]
    end_date = args.to_date or pcfg.get("end_date")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    wdf, strat_returns, net_returns, equity, metrics = run_orchestrator_backtest(
        cfg=cfg,
        start_date=start_date,
        end_date=end_date,
        step_days=max(1, int(args.step_days)),
        max_cycles=int(args.max_cycles),
        use_prices_override=True,
    )

    wdf.to_csv(out_dir / "weights.csv")
    strat_returns.to_csv(out_dir / "gross_returns.csv", header=True)
    net_returns.to_csv(out_dir / "net_returns.csv", header=True)
    equity.to_csv(out_dir / "equity_curve.csv", header=["equity"])
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    print("Orchestrator-stack backtest complete.")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
