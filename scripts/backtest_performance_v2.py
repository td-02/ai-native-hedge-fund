from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest_orchestrator_stack import run_orchestrator_backtest
from free_fund.config import load_config


def _metrics(returns: pd.Series) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {}
    eq = (1.0 + r).cumprod()
    vol = float(r.std(ddof=0) * math.sqrt(252))
    sharpe = float((r.mean() * 252) / (vol + 1e-12))
    dd = eq / eq.cummax() - 1.0
    cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1.0)
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
    }


def _benchmark_returns(index: pd.DatetimeIndex, symbols: list[str], mode: str = "spy") -> pd.Series:
    start = index.min().date().isoformat()
    end = (index.max() + pd.Timedelta(days=1)).date().isoformat()
    if mode == "60_40":
        px = yf.download(["SPY", "TLT"], start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(px, pd.Series):
            px = px.to_frame()
        r = px.pct_change().fillna(0.0)
        out = 0.6 * r["SPY"] + 0.4 * r["TLT"]
        return out.reindex(index).fillna(0.0)

    s = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(s, pd.DataFrame):
        s = s.iloc[:, 0]
    return s.pct_change().fillna(0.0).reindex(index).fillna(0.0)


def _asset_attribution(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    rets = prices.pct_change().reindex(weights.index).fillna(0.0)
    contrib = (weights.reindex(columns=prices.columns).fillna(0.0) * rets.reindex(columns=weights.columns).fillna(0.0))
    out = pd.DataFrame(
        {
            "asset": contrib.columns,
            "mean_contribution": [float(contrib[c].mean()) for c in contrib.columns],
            "cum_contribution": [float((1.0 + contrib[c]).prod() - 1.0) for c in contrib.columns],
            "avg_abs_weight": [float(weights[c].abs().mean()) if c in weights.columns else 0.0 for c in contrib.columns],
        }
    ).sort_values("cum_contribution", ascending=False)
    return out


def _run_variant(cfg: dict, start_date: str, end_date: str | None, step_days: int, max_cycles: int):
    try:
        return run_orchestrator_backtest(
            cfg=cfg,
            start_date=start_date,
            end_date=end_date,
            step_days=step_days,
            max_cycles=max_cycles,
            use_prices_override=True,
        )
    except ValueError as e:
        if "No decision rows available" not in str(e):
            raise
        # Short-window fallback for sanity runs.
        cfg2 = copy.deepcopy(cfg)
        lb = int(cfg2.get("portfolio", {}).get("lookback_days", 252))
        cfg2["portfolio"]["lookback_days"] = max(63, lb // 2)
        return run_orchestrator_backtest(
            cfg=cfg2,
            start_date=start_date,
            end_date=end_date,
            step_days=step_days,
            max_cycles=max_cycles,
            use_prices_override=True,
        )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-config", default="configs/backtest_fast.yaml")
    p.add_argument("--v2-config", default="configs/performance_v2.yaml")
    p.add_argument("--from-date", default="2018-01-01")
    p.add_argument("--to-date", default=None)
    p.add_argument("--step-days", type=int, default=21)
    p.add_argument("--max-cycles", type=int, default=0)
    p.add_argument("--out", default="outputs/perf_v2_compare")
    args = p.parse_args()

    base = load_config(args.baseline_config)
    v2 = load_config(args.v2_config)
    base.setdefault("backtest", {})
    v2.setdefault("backtest", {})
    base["backtest"]["fast_mode"] = True
    v2["backtest"]["fast_mode"] = True

    bw, _, bnet, beq, bmet = _run_variant(copy.deepcopy(base), args.from_date, args.to_date, args.step_days, args.max_cycles)
    vw, _, vnet, veq, vmet = _run_variant(copy.deepcopy(v2), args.from_date, args.to_date, args.step_days, args.max_cycles)

    # Rebalance dates differ by universe/calendar. Compare on union index with flat return on non-rebalance dates.
    bnet.index = pd.to_datetime(bnet.index)
    vnet.index = pd.to_datetime(vnet.index)
    idx = bnet.index.union(vnet.index).sort_values()
    bnet = bnet.reindex(idx).fillna(0.0)
    vnet = vnet.reindex(idx).fillna(0.0)
    beq = (1.0 + bnet).cumprod()
    veq = (1.0 + vnet).cumprod()
    spy = _benchmark_returns(idx, symbols=[], mode="spy")
    s6040 = _benchmark_returns(idx, symbols=[], mode="60_40")

    smet = _metrics(spy)
    m6040 = _metrics(s6040)
    bmet = {**_metrics(bnet), "avg_turnover": float(bmet.get("avg_turnover", 0.0)), "cycles": float(len(idx))}
    vmet = {**_metrics(vnet), "avg_turnover": float(vmet.get("avg_turnover", 0.0)), "cycles": float(len(idx))}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    comp = pd.DataFrame(
        [
            {"variant": "baseline", **bmet},
            {"variant": "performance_v2", **vmet},
            {"variant": "benchmark_spy", **smet},
            {"variant": "benchmark_60_40", **m6040},
        ]
    )
    comp.to_csv(out / "comparison_metrics.csv", index=False)

    eq_df = pd.DataFrame(
        {
            "baseline": beq,
            "performance_v2": veq,
            "benchmark_spy": (1.0 + spy).cumprod(),
            "benchmark_60_40": (1.0 + s6040).cumprod(),
        }
    )
    eq_df.to_csv(out / "equity_curves.csv", index=True)

    # Attribution for baseline and v2.
    price_base = yf.download(list(base["portfolio"]["symbols"]), start=args.from_date, end=(args.to_date or pd.Timestamp.today().date().isoformat()), auto_adjust=True, progress=False)["Close"]
    if isinstance(price_base, pd.Series):
        price_base = price_base.to_frame()
    price_v2 = yf.download(list(v2["portfolio"]["symbols"]), start=args.from_date, end=(args.to_date or pd.Timestamp.today().date().isoformat()), auto_adjust=True, progress=False)["Close"]
    if isinstance(price_v2, pd.Series):
        price_v2 = price_v2.to_frame()

    b_attr = _asset_attribution(bw.fillna(0.0), price_base)
    v_attr = _asset_attribution(vw.fillna(0.0), price_v2)
    b_attr.to_csv(out / "signal_attribution_baseline.csv", index=False)
    v_attr.to_csv(out / "signal_attribution_performance_v2.csv", index=False)

    sharpe_lift = float(vmet.get("sharpe", 0.0) - bmet.get("sharpe", 0.0))
    winner = "performance_v2" if vmet.get("sharpe", -1e9) > bmet.get("sharpe", -1e9) else "baseline"
    loser = "baseline" if winner == "performance_v2" else "performance_v2"

    print("Performance comparison complete.")
    print(comp.to_string(index=False))
    print(f"Winner by Sharpe: {winner}")
    print(f"Loser by Sharpe: {loser}")
    print(f"Sharpe lift (v2 - baseline): {sharpe_lift:.4f}")


if __name__ == "__main__":
    main()
