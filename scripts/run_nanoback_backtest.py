from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nanoback import BacktestConfig, MarketData, export_ledger_csv, run_compiled_policy_backtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from free_fund.config import load_config
from free_fund.data import download_close_prices


POLICIES = (
    "momentum",
    "mean_reversion",
    "moving_average_crossover",
    "volatility_filtered_momentum",
    "cross_sectional_momentum",
    "minimum_variance",
)


def _normalize_prices(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.sort_index().copy()
    frame = frame.ffill().bfill().dropna(how="any")
    if frame.empty:
        raise ValueError("No usable price data after cleaning.")
    return frame


def _to_market_data(prices: pd.DataFrame) -> MarketData:
    frame = _normalize_prices(prices)
    close = frame.to_numpy(dtype=np.float64, copy=True)
    timestamps = pd.DatetimeIndex(frame.index).asi8.astype(np.int64, copy=False)
    symbols = [str(col) for col in frame.columns]
    return MarketData(
        timestamps=timestamps,
        close=close,
        high=close.copy(),
        low=close.copy(),
        volume=np.full_like(close, 1_000_000.0, dtype=np.float64),
        symbols=symbols,
    )


def _build_backtest_config(args: argparse.Namespace, cfg: dict) -> BacktestConfig:
    costs = cfg.get("costs", {})
    nb_cfg = BacktestConfig()
    nb_cfg.starting_cash = float(args.starting_cash)
    nb_cfg.commission_bps = float(
        args.commission_bps if args.commission_bps is not None else costs.get("transaction_cost_bps", 0.0)
    )
    nb_cfg.slippage_bps = float(args.slippage_bps if args.slippage_bps is not None else costs.get("slippage_bps", 0.0))
    nb_cfg.max_position = int(args.gross_target)
    nb_cfg.max_gross_leverage = float(args.max_gross_leverage)
    nb_cfg.allow_short = not bool(args.no_short)
    nb_cfg.use_bid_ask_execution = bool(args.use_bid_ask_execution)
    nb_cfg.mark_to_market = bool(args.mark_to_market)
    nb_cfg.max_drawdown_pct = float(args.max_drawdown_pct)
    return nb_cfg


def _policy_kwargs(args: argparse.Namespace) -> dict[str, float | int]:
    if args.policy == "minimum_variance":
        return {
            "window": int(args.window),
            "ridge": float(args.ridge),
            "leverage": float(args.leverage),
            "gross_target": int(args.gross_target),
        }
    if args.policy == "moving_average_crossover":
        return {
            "fast_window": int(args.fast_window),
            "slow_window": int(args.slow_window),
            "max_position": int(args.gross_target),
        }
    if args.policy == "volatility_filtered_momentum":
        return {
            "lookback": int(args.lookback),
            "vol_window": int(args.vol_window),
            "volatility_ceiling": float(args.volatility_ceiling),
            "max_position": int(args.gross_target),
        }
    if args.policy == "cross_sectional_momentum":
        return {
            "lookback": int(args.lookback),
            "winners": int(args.winners),
            "losers": int(args.losers),
            "max_position": int(args.gross_target),
        }
    return {
        "lookback": int(args.lookback),
        "max_position": int(args.gross_target),
    }


def _equity_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().fillna(0.0)
    if returns.empty:
        return {}
    daily_vol = float(returns.std(ddof=0) * math.sqrt(252))
    sharpe = float((returns.mean() * 252) / (daily_vol + 1e-12))
    drawdown = equity / equity.cummax() - 1.0
    days = max(1.0, float((equity.index[-1] - equity.index[0]).days))
    years = days / 365.25
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / max(0.1, years)) - 1.0)
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": cagr,
        "annual_vol": daily_vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a nanoback-backed portfolio backtest.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--from-date", default=None, help="Override portfolio.start_date")
    parser.add_argument("--to-date", default=None, help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--policy", choices=POLICIES, default="minimum_variance")
    parser.add_argument("--lookback", type=int, default=63)
    parser.add_argument("--winners", type=int, default=2)
    parser.add_argument("--losers", type=int, default=2)
    parser.add_argument("--window", type=int, default=63)
    parser.add_argument("--ridge", type=float, default=1e-6)
    parser.add_argument("--leverage", type=float, default=1.0)
    parser.add_argument("--fast-window", type=int, default=21)
    parser.add_argument("--slow-window", type=int, default=63)
    parser.add_argument("--vol-window", type=int, default=21)
    parser.add_argument("--volatility-ceiling", type=float, default=0.25)
    parser.add_argument("--gross-target", type=int, default=100)
    parser.add_argument("--starting-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--max-gross-leverage", type=float, default=10.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=1.0)
    parser.add_argument("--no-short", action="store_true")
    parser.add_argument("--use-bid-ask-execution", action="store_true")
    parser.add_argument("--mark-to-market", action="store_true")
    parser.add_argument("--export-ledger", action="store_true")
    parser.add_argument("--out", default="outputs/nanoback_backtest")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pcfg = cfg["portfolio"]
    start_date = args.from_date or pcfg["start_date"]
    end_date = args.to_date or pcfg.get("end_date")

    prices = download_close_prices(
        symbols=list(pcfg["symbols"]),
        start_date=start_date,
        end_date=end_date,
    )
    cleaned_prices = _normalize_prices(prices)
    market = _to_market_data(cleaned_prices)
    nb_cfg = _build_backtest_config(args, cfg)
    result = run_compiled_policy_backtest(
        market,
        policy=args.policy,
        config=nb_cfg,
        **_policy_kwargs(args),
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    equity = pd.Series(result.equity_curve, index=cleaned_prices.index, name="equity")
    cash = pd.Series(result.cash_curve, index=cleaned_prices.index, name="cash")
    positions = pd.DataFrame(result.positions, index=cleaned_prices.index, columns=market.symbols)

    equity.to_csv(out_dir / "equity_curve.csv", header=True)
    cash.to_csv(out_dir / "cash_curve.csv", header=True)
    positions.to_csv(out_dir / "positions.csv")

    metrics = {
        "policy": args.policy,
        "starting_cash": float(nb_cfg.starting_cash),
        "ending_cash": float(result.ending_cash),
        "ending_equity": float(result.ending_equity),
        "pnl": float(result.pnl),
        "turnover": float(result.turnover),
        "total_fees": float(result.total_fees),
        "total_borrow_cost": float(result.total_borrow_cost),
        "total_cash_yield": float(result.total_cash_yield),
        "peak_equity": float(result.peak_equity),
        "max_drawdown_raw": float(result.max_drawdown),
        "halted_by_risk": bool(result.halted_by_risk),
        "submitted_orders": int(result.submitted_orders),
        "filled_orders": int(result.filled_orders),
        "rejected_orders": int(result.rejected_orders),
        "symbols": len(market.symbols),
        "rows": len(cleaned_prices),
        **_equity_metrics(equity),
    }
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)

    if args.export_ledger:
        export_ledger_csv(result.raw, out_dir / "ledger.csv")

    print("Nanoback backtest complete.")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
