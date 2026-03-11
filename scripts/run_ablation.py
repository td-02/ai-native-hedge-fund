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
    if isinstance(returns, pd.DataFrame):
        returns = returns.iloc[:, 0]
    returns = returns.dropna()
    if returns.empty:
        return {}
    eq = (1.0 + returns).cumprod()
    vol = float(returns.std(ddof=0) * math.sqrt(252))
    sharpe = float((returns.mean() * 252) / (vol + 1e-12))
    dd = eq / eq.cummax() - 1.0
    cagr = float(eq.iloc[-1] ** (252 / len(returns)) - 1.0)
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(dd.min()),
    }


def _benchmarks(index: pd.DatetimeIndex, symbols: list[str]) -> dict[str, dict[str, float]]:
    start = index.min().date().isoformat()
    end = (index.max() + pd.Timedelta(days=1)).date().isoformat()
    spy = yf.download("SPY", start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(spy, pd.DataFrame):
        spy = spy.iloc[:, 0]
    spy = spy.reindex(index).ffill()
    spy_ret = spy.pct_change().fillna(0.0)

    px = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    px = px.reindex(index).ffill()
    ew_ret = px.pct_change().fillna(0.0).mean(axis=1)
    return {"SPY": _metrics(spy_ret), "EW5": _metrics(ew_ret)}


def _renorm_strategy_weights(cfg: dict) -> None:
    w = dict(cfg["strategies"]["weights"])
    s = sum(float(v) for v in w.values())
    if s <= 1e-12:
        return
    cfg["strategies"]["weights"] = {k: float(v) / s for k, v in w.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--from-date", default="2024-01-01")
    parser.add_argument("--to-date", default="2026-03-01")
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--max-cycles", type=int, default=20)
    parser.add_argument("--fast-mode", action="store_true", help="Use fast backtest mode (no RSS/LLM/macro API calls).")
    parser.add_argument("--out", default="outputs/ablation")
    args = parser.parse_args()

    base = load_config(args.config)
    base.setdefault("backtest", {})
    base["backtest"]["fast_mode"] = bool(args.fast_mode)
    variants: list[tuple[str, dict]] = []
    alpha_signals = list(
        base.get("alpha_pipeline", {}).get(
            "enabled_signals",
            [
                "earnings_momentum",
                "analyst_revisions",
                "options_iv_term_structure",
                "volume_liquidity_shock",
                "short_interest",
                "block_deals",
                "quality_profitability",
                "pead_signal",
            ],
        )
        or []
    )

    variants.append(("full", copy.deepcopy(base)))

    v = copy.deepcopy(base)
    v["alpha_pipeline"]["blend_weight"] = 0.0
    variants.append(("no_alpha_pipeline", v))

    v = copy.deepcopy(base)
    v["research_council"]["blend_weight"] = 0.0
    v["research_council"]["enabled"] = False
    variants.append(("no_research_council", v))

    v = copy.deepcopy(base)
    v["microstructure"]["blend_weight"] = 0.0
    variants.append(("no_microstructure", v))

    v = copy.deepcopy(base)
    v["private_alpha"]["blend_weight"] = 0.0
    variants.append(("no_private_alpha", v))

    v = copy.deepcopy(base)
    v["alpha_pipeline"]["blend_weight"] = 0.0
    v["research_council"]["blend_weight"] = 0.0
    v["research_council"]["enabled"] = False
    v["microstructure"]["blend_weight"] = 0.0
    v["private_alpha"]["blend_weight"] = 0.0
    variants.append(("core_strategies_only", v))

    v = copy.deepcopy(base)
    v["strategies"]["weights"]["event_driven"] = 0.0
    _renorm_strategy_weights(v)
    variants.append(("no_event_driven", v))

    # Per-signal alpha validation (isolated alpha signal with controlled blend weight).
    for sig in alpha_signals:
        v = copy.deepcopy(base)
        v.setdefault("alpha_pipeline", {})
        v["alpha_pipeline"]["blend_weight"] = 0.12
        v["alpha_pipeline"]["enabled_signals"] = [sig]
        variants.append((f"alpha_only_{sig}", v))

    # Exclusion tests for the new factors.
    for sig in ("quality_profitability", "pead_signal"):
        if sig in alpha_signals:
            v = copy.deepcopy(base)
            v.setdefault("alpha_pipeline", {})
            enabled = [x for x in alpha_signals if x != sig]
            v["alpha_pipeline"]["enabled_signals"] = enabled
            variants.append((f"no_{sig}", v))

    rows: list[dict[str, float | str]] = []
    bench: dict[str, dict[str, float]] | None = None
    for name, cfg in variants:
        _, _, net, _, m = run_orchestrator_backtest(
            cfg=cfg,
            start_date=args.from_date,
            end_date=args.to_date,
            step_days=args.step_days,
            max_cycles=args.max_cycles,
            use_prices_override=True,
        )
        if bench is None:
            bench = _benchmarks(net.index, list(cfg["portfolio"]["symbols"]))
        rows.append({"variant": name, **m})

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values("sharpe", ascending=False)
    df.to_csv(out / "ablation_results.csv", index=False)

    if bench:
        bdf = pd.DataFrame(
            [{"variant": "benchmark_spy", **bench["SPY"]}, {"variant": "benchmark_equal_weight", **bench["EW5"]}]
        )
        bdf.to_csv(out / "benchmark_results.csv", index=False)

    print("Ablation complete.")
    print(df.to_string(index=False))
    if bench:
        print(pd.DataFrame([{"variant": "benchmark_spy", **bench["SPY"]}, {"variant": "benchmark_equal_weight", **bench["EW5"]}]).to_string(index=False))


if __name__ == "__main__":
    main()
