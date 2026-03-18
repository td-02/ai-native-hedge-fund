from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest_orchestrator_stack import run_orchestrator_backtest
from free_fund.config import load_config


def _metrics(returns: pd.Series, label: str = "") -> dict:
    r = returns.dropna()
    if r.empty:
        return {}
    equity = (1.0 + r).cumprod()
    # Infer actual trading days per return observation from index spacing.
    if len(r) >= 2:
        avg_spacing = (r.index[-1] - r.index[0]).days / max(1, len(r) - 1)
        periods_per_year = 365.25 / max(1, avg_spacing)
    else:
        periods_per_year = 252  # fallback
    vol = float(r.std(ddof=0) * np.sqrt(periods_per_year))
    mu = float(r.mean() * periods_per_year)
    sharpe = mu / (vol + 1e-12)
    dd = equity / equity.cummax() - 1.0
    n_years = (r.index[-1] - r.index[0]).days / 365.25
    cagr = float(equity.iloc[-1] ** (1.0 / max(0.1, n_years)) - 1.0)
    calmar = cagr / (abs(float(dd.min())) + 1e-12)
    win_rate = float((r > 0).mean())
    return {
        "label": label,
        "total_return": round(float(equity.iloc[-1] - 1.0), 4),
        "cagr": round(cagr, 4),
        "annual_vol": round(vol, 4),
        "sharpe": round(sharpe, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(float(dd.min()), 4),
        "win_rate": round(win_rate, 4),
        "cycles": len(r),
    }


def _spy_benchmark(prices: pd.DataFrame, decision_rows: list, step: int) -> pd.Series:
    """SPY buy-and-hold with holding-period returns matching fund rebalance dates."""
    if "SPY" not in prices.columns:
        return pd.Series(dtype=float, name="SPY")
    rets = prices["SPY"].pct_change().fillna(0.0)
    period_rets = []
    idx_list = []
    for k, i in enumerate(decision_rows):
        next_i = decision_rows[k + 1] if k + 1 < len(decision_rows) else min(i + step, len(prices) - 1)
        r = rets.iloc[i + 1 : next_i + 1]
        period_rets.append(float((1 + r).prod() - 1) if not r.empty else 0.0)
        idx_list.append(prices.index[next_i])
    return pd.Series(period_rets, index=idx_list, name="SPY")


def _6040_benchmark(prices: pd.DataFrame, decision_rows: list, step: int) -> pd.Series:
    """60/40 SPY+TLT with same holding periods."""
    if "SPY" not in prices.columns:
        return pd.Series(dtype=float, name="60_40")
    spy = prices["SPY"].pct_change().fillna(0.0)
    tlt = prices["TLT"].pct_change().fillna(0.0) if "TLT" in prices.columns else spy * 0
    period_rets = []
    idx_list = []
    for k, i in enumerate(decision_rows):
        next_i = decision_rows[k + 1] if k + 1 < len(decision_rows) else min(i + step, len(prices) - 1)
        rs = spy.iloc[i + 1 : next_i + 1]
        rt = tlt.iloc[i + 1 : next_i + 1]
        r = 0.6 * (float((1 + rs).prod() - 1) if not rs.empty else 0.0) + \
            0.4 * (float((1 + rt).prod() - 1) if not rt.empty else 0.0)
        period_rets.append(r)
        idx_list.append(prices.index[next_i])
    return pd.Series(period_rets, index=idx_list, name="60_40")


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

    bw, _, bnet, beq, bmet, prices_base, decision_rows_base = _run_variant(
        copy.deepcopy(base), args.from_date, args.to_date, args.step_days, args.max_cycles
    )
    vw, _, vnet, veq, vmet, prices_v2, decision_rows_v2 = _run_variant(
        copy.deepcopy(v2), args.from_date, args.to_date, args.step_days, args.max_cycles
    )

    # Compare on union index with flat return on non-overlapping decision points.
    bnet.index = pd.to_datetime(bnet.index)
    vnet.index = pd.to_datetime(vnet.index)
    idx = bnet.index.union(vnet.index).sort_values()
    bnet = bnet.reindex(idx).fillna(0.0)
    vnet = vnet.reindex(idx).fillna(0.0)
    beq = (1.0 + bnet).cumprod()
    veq = (1.0 + vnet).cumprod()
    spy = _spy_benchmark(prices_v2, decision_rows_v2, args.step_days).reindex(idx).fillna(0.0)
    s6040 = _6040_benchmark(prices_v2, decision_rows_v2, args.step_days).reindex(idx).fillna(0.0)

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
    b_attr = _asset_attribution(bw.fillna(0.0), prices_base)
    v_attr = _asset_attribution(vw.fillna(0.0), prices_v2)
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
