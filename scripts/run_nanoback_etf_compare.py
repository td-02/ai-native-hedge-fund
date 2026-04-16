from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from nanoback import OrderIntent, Strategy, export_ledger_csv, run_strategy_backtest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from free_fund.config import load_config
from free_fund.data import download_close_prices
from scripts.run_nanoback_backtest import _build_backtest_config, _normalize_prices, _to_market_data


def _equity_metrics(equity: pd.Series) -> dict[str, float]:
    returns = equity.pct_change().fillna(0.0)
    if returns.empty:
        return {}
    vol = float(returns.std(ddof=0) * math.sqrt(252))
    sharpe = float((returns.mean() * 252) / (vol + 1e-12))
    drawdown = equity / equity.cummax() - 1.0
    days = max(1.0, float((equity.index[-1] - equity.index[0]).days))
    years = days / 365.25
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1.0 / max(0.1, years)) - 1.0)
    return {
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0),
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "win_rate": float((returns > 0).mean()),
    }


def _benchmark_returns(prices: pd.DataFrame, kind: str) -> pd.Series:
    rets = prices.pct_change().fillna(0.0)
    if kind == "benchmark_spy":
        if "SPY" not in rets.columns:
            return pd.Series(dtype=float, name=kind)
        return rets["SPY"].rename(kind)
    if kind == "benchmark_equal_weight":
        return rets.mean(axis=1).rename(kind)
    if kind == "benchmark_60_40":
        spy = rets["SPY"] if "SPY" in rets.columns else pd.Series(0.0, index=rets.index)
        tlt = rets["TLT"] if "TLT" in rets.columns else pd.Series(0.0, index=rets.index)
        return (0.6 * spy + 0.4 * tlt).rename(kind)
    raise ValueError(f"unknown benchmark kind: {kind}")


def _preset_definitions(lookback_days: int, rebalance_every: int) -> dict[str, dict[str, object]]:
    return {
        "nanoback_etf_core": {
            "strategy": "equal_weight",
            "lookback": max(21, rebalance_every),
            "target_gross_fraction": 0.95,
            "selection_count": 100,
        },
        "nanoback_defensive_core": {
            "strategy": "inverse_volatility",
            "lookback": max(63, lookback_days // 4, rebalance_every * 3),
            "target_gross_fraction": 0.95,
            "selection_count": 100,
        },
    }


def _window_scores(window: pd.DataFrame, mode: str) -> pd.Series:
    returns = window.pct_change().dropna(how="all")
    if returns.empty:
        return pd.Series(0.0, index=window.columns)
    if mode == "equal_weight":
        return pd.Series(1.0, index=window.columns)
    if mode == "inverse_volatility":
        vol = returns.std(ddof=0).replace(0.0, np.nan)
        scores = (1.0 / vol).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return scores.reindex(window.columns).fillna(0.0)
    if mode == "momentum_rotation":
        momentum = window.iloc[-1] / window.iloc[0] - 1.0
        vol = returns.std(ddof=0).replace(0.0, np.nan)
        scores = momentum / vol
        return scores.reindex(window.columns).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    raise ValueError(f"unknown strategy mode: {mode}")


class _ETFStrategy(Strategy):
    def __init__(
        self,
        *,
        mode: str,
        lookback: int,
        rebalance_every: int,
        target_gross_fraction: float,
        selection_count: int,
        starting_cash: float,
    ) -> None:
        self.mode = mode
        self.lookback = int(lookback)
        self.rebalance_every = max(1, int(rebalance_every))
        self.target_gross_fraction = max(0.0, float(target_gross_fraction))
        self.selection_count = max(1, int(selection_count))
        self.starting_cash = float(starting_cash)
        self.data = None

    def on_start(self, data) -> None:
        self.data = data

    def on_event(self, event):
        if self.data is None or event.index < self.lookback:
            return ()
        if (event.index - self.lookback) % self.rebalance_every != 0 and event.index != self.data.row_count - 1:
            return ()

        window = pd.DataFrame(
            self.data.close[event.index - self.lookback : event.index],
            columns=self.data.symbols,
        )
        scores = _window_scores(window, self.mode)
        if self.mode == "inverse_volatility":
            selected = list(scores.sort_values(ascending=False).head(self.selection_count).index)
            weights = scores.reindex(selected).clip(lower=0.0)
            if float(weights.sum()) <= 0.0:
                weights = pd.Series(1.0, index=selected)
            weights = weights / float(weights.sum())
        elif self.mode == "momentum_rotation":
            selected = list(scores.sort_values(ascending=False).head(self.selection_count).index)
            weights = pd.Series(1.0, index=selected)
            weights = weights / float(weights.sum())
        else:
            selected = list(window.columns)
            weights = pd.Series(1.0, index=selected)
            weights = weights / float(weights.sum())

        current_prices = pd.Series(event.close, index=self.data.symbols)
        target_value = self.starting_cash * self.target_gross_fraction
        target_quantities = {symbol: 0 for symbol in window.columns}
        for symbol in selected:
            price = float(current_prices.get(symbol, 0.0))
            if price <= 0.0:
                continue
            weight = float(weights.get(symbol, 0.0))
            quantity = int(round((target_value * weight) / price))
            target_quantities[symbol] = max(0, quantity)

        intents = []
        for symbol in window.columns:
            asset_idx = self.data.symbols.index(symbol)
            quantity = int(target_quantities.get(symbol, 0))
            intents.append(OrderIntent(asset=asset_idx, target_position=quantity))
        return intents


def _run_strategy(
    *,
    name: str,
    spec: dict[str, object],
    market,
    cleaned_prices: pd.DataFrame,
    rebalance_every: int,
    cfg: dict,
    args: argparse.Namespace,
    out_root: Path,
) -> tuple[pd.Series, dict[str, float]]:
    base_cfg = _build_backtest_config(args, cfg)
    price_floor = float(cleaned_prices.min().min())
    if not np.isfinite(price_floor) or price_floor <= 0:
        price_floor = 1.0
    # nanoback's max_position is a per-asset share cap, so set it high enough for
    # the ETF universe to express meaningful dollar allocations.
    base_cfg.max_position = max(
        int(base_cfg.max_position),
        int(math.ceil(base_cfg.starting_cash / price_floor)) * 2,
    )
    strategy = _ETFStrategy(
        mode=str(spec["strategy"]),
        lookback=int(spec["lookback"]),
        rebalance_every=int(rebalance_every),
        target_gross_fraction=float(spec["target_gross_fraction"]),
        selection_count=int(spec["selection_count"]),
        starting_cash=float(base_cfg.starting_cash),
    )
    result = run_strategy_backtest(market, strategy, config=base_cfg)

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    equity = pd.Series(result.equity_curve, index=cleaned_prices.index, name="equity")
    cash = pd.Series(result.cash_curve, index=cleaned_prices.index, name="cash")
    positions = pd.DataFrame(result.positions, index=cleaned_prices.index, columns=market.symbols)

    equity.to_csv(out_dir / "equity_curve.csv", header=True)
    cash.to_csv(out_dir / "cash_curve.csv", header=True)
    positions.to_csv(out_dir / "positions.csv")

    metrics = {
        "variant": name,
        "policy": str(spec["strategy"]),
        "lookback": int(spec["lookback"]),
        "target_gross_fraction": float(spec["target_gross_fraction"]),
        "selection_count": int(spec["selection_count"]),
        "effective_max_position": int(base_cfg.max_position),
        "starting_cash": float(base_cfg.starting_cash),
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

    return equity, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two nanoback presets against benchmarks.")
    parser.add_argument("--config", default="configs/performance_v2.yaml")
    parser.add_argument("--from-date", default=None, help="Override portfolio.start_date")
    parser.add_argument("--to-date", default=None, help="Backtest end date YYYY-MM-DD")
    parser.add_argument("--starting-cash", type=float, default=1_000_000.0)
    parser.add_argument("--commission-bps", type=float, default=None)
    parser.add_argument("--slippage-bps", type=float, default=None)
    parser.add_argument("--max-gross-leverage", type=float, default=10.0)
    parser.add_argument("--max-drawdown-pct", type=float, default=1.0)
    parser.add_argument("--gross-target", type=int, default=100)
    parser.add_argument("--no-short", action="store_true")
    parser.add_argument("--use-bid-ask-execution", action="store_true")
    parser.add_argument("--mark-to-market", action="store_true")
    parser.add_argument("--export-ledger", action="store_true")
    parser.add_argument("--out", default="outputs/nanoback_etf_compare")
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
    presets = _preset_definitions(int(pcfg.get("lookback_days", 126)), int(pcfg.get("rebalance_every_n_days", 21)))
    rebalance_every = int(pcfg.get("rebalance_every_n_days", 21))

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    strategy_equities: dict[str, pd.Series] = {}
    rows: list[dict[str, float | str]] = []
    for name, spec in presets.items():
        equity, metrics = _run_strategy(
            name=name,
            spec=spec,
            market=market,
            cleaned_prices=cleaned_prices,
            rebalance_every=rebalance_every,
            cfg=cfg,
            args=args,
            out_root=out_root,
        )
        strategy_equities[name] = equity
        rows.append(metrics)

    benchmark_specs = ("benchmark_spy", "benchmark_equal_weight", "benchmark_60_40")
    benchmark_equities: dict[str, pd.Series] = {}
    for kind in benchmark_specs:
        bench_returns = _benchmark_returns(cleaned_prices, kind)
        bench_equity = (1.0 + bench_returns).cumprod().rename(kind)
        benchmark_equities[kind] = bench_equity
        rows.append({"variant": kind, **_equity_metrics(bench_equity)})

    all_equities = {**strategy_equities, **benchmark_equities}
    eq_df = pd.concat(all_equities.values(), axis=1)
    eq_df.columns = list(all_equities.keys())
    eq_df.to_csv(out_root / "equity_curves.csv", index=True)

    comp = pd.DataFrame(rows)
    if "sharpe" in comp.columns:
        comp = comp.sort_values("sharpe", ascending=False)
    comp.to_csv(out_root / "comparison_metrics.csv", index=False)

    strategy_comp = comp[comp["variant"].isin(presets.keys())].set_index("variant")
    if "benchmark_spy" in comp["variant"].values:
        spy_sharpe = float(comp.loc[comp["variant"] == "benchmark_spy", "sharpe"].iloc[0])
        strategy_comp = strategy_comp.assign(sharpe_vs_spy=strategy_comp["sharpe"] - spy_sharpe)
    if "benchmark_equal_weight" in comp["variant"].values:
        ew_sharpe = float(comp.loc[comp["variant"] == "benchmark_equal_weight", "sharpe"].iloc[0])
        strategy_comp = strategy_comp.assign(sharpe_vs_equal_weight=strategy_comp["sharpe"] - ew_sharpe)

    print("Nanoback ETF comparison complete.")
    print(comp.to_string(index=False))
    print()
    print(strategy_comp.to_string())


if __name__ == "__main__":
    main()

