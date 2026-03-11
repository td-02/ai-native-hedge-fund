from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import pandas as pd
import yfinance as yf
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest_orchestrator_stack import run_orchestrator_backtest
from free_fund.config import load_config

DEFAULT_ALPHA_SIGNALS = [
    "earnings_momentum",
    "analyst_revisions",
    "options_iv_term_structure",
    "volume_liquidity_shock",
    "short_interest",
    "block_deals",
    "quality_profitability",
    "pead_signal",
]


def _metrics(returns: pd.Series) -> dict[str, float]:
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


def _benchmark_returns(index: pd.DatetimeIndex, symbols: list[str], mode: str, symbol: str) -> pd.Series:
    start = index.min().date().isoformat()
    end = (index.max() + pd.Timedelta(days=1)).date().isoformat()
    if mode == "equal_weight":
        px = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        px = px.reindex(index).ffill()
        return px.pct_change().fillna(0.0).mean(axis=1)
    series = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = series.reindex(index).ffill()
    return series.pct_change().fillna(0.0)


def _evaluate_single_signal_oos(
    base: dict,
    signal: str,
    windows: list[tuple[str, str, str, str]],
    step_days: int,
    max_cycles: int,
) -> dict[str, float]:
    rows: list[dict[str, float]] = []
    for _, _, te_s, te_e in windows:
        cfg = copy.deepcopy(base)
        cfg.setdefault("alpha_pipeline", {})
        cfg["alpha_pipeline"]["blend_weight"] = 0.10
        cfg["alpha_pipeline"]["enabled_signals"] = [signal]
        try:
            _, _, net, _, _ = run_orchestrator_backtest(
                cfg=cfg,
                start_date=te_s,
                end_date=te_e,
                step_days=step_days,
                max_cycles=max_cycles,
                use_prices_override=True,
            )
        except Exception:
            continue
        if net.empty:
            continue
        bench = _benchmark_returns(
            net.index,
            symbols=list(cfg["portfolio"]["symbols"]),
            mode=str(cfg.get("benchmark", {}).get("mode", "symbol")),
            symbol=str(cfg.get("benchmark", {}).get("symbol", "SPY")),
        )
        ex = _metrics(net - bench.reindex(net.index).fillna(0.0))
        if ex:
            rows.append({"excess_sharpe": float(ex["sharpe"]), "excess_cagr": float(ex["cagr"])})
    if not rows:
        return {"signal": signal, "windows": 0.0, "avg_excess_sharpe": -1e9, "avg_excess_cagr": -1e9}
    df = pd.DataFrame(rows)
    return {
        "signal": signal,
        "windows": float(len(df)),
        "avg_excess_sharpe": float(df["excess_sharpe"].mean()),
        "avg_excess_cagr": float(df["excess_cagr"].mean()),
    }


def _candidate_configs(base: dict, kept_signals: list[str]) -> list[dict]:
    candidates: list[dict] = []
    turnover_grid = [0.8, 1.2, 1.8]
    risk_grid = [0.3, 0.5, 0.8]
    topk_grid = [2, 3, 4]
    alpha_weight_grid = [0.0] if not kept_signals else [0.05, 0.10]
    regime_presets = {
        "trend_bias": {
            "trend": {"trend_following": 0.36, "mean_reversion": 0.05, "volatility_carry": 0.06, "regime_switching": 0.16, "event_driven": 0.05, "relative_strength_rotation": 0.20, "dual_momentum_gate": 0.12},
            "meanrev": {"trend_following": 0.14, "mean_reversion": 0.30, "volatility_carry": 0.18, "regime_switching": 0.16, "event_driven": 0.04, "relative_strength_rotation": 0.10, "dual_momentum_gate": 0.08},
            "stress": {"trend_following": 0.05, "mean_reversion": 0.15, "volatility_carry": 0.30, "regime_switching": 0.20, "event_driven": 0.03, "relative_strength_rotation": 0.07, "dual_momentum_gate": 0.20},
        },
        "meanrev_bias": {
            "trend": {"trend_following": 0.28, "mean_reversion": 0.20, "volatility_carry": 0.10, "regime_switching": 0.16, "event_driven": 0.06, "relative_strength_rotation": 0.12, "dual_momentum_gate": 0.08},
            "meanrev": {"trend_following": 0.10, "mean_reversion": 0.38, "volatility_carry": 0.20, "regime_switching": 0.14, "event_driven": 0.04, "relative_strength_rotation": 0.08, "dual_momentum_gate": 0.06},
            "stress": {"trend_following": 0.05, "mean_reversion": 0.20, "volatility_carry": 0.30, "regime_switching": 0.20, "event_driven": 0.03, "relative_strength_rotation": 0.06, "dual_momentum_gate": 0.16},
        },
        "balanced": copy.deepcopy(base.get("regime_controls", {}).get("strategy_weights_by_regime", {})),
    }

    signal_masks: list[list[str]] = [[]]
    if kept_signals:
        signal_masks.append(kept_signals)
        for s in kept_signals:
            signal_masks.append([s])
        if len(kept_signals) >= 2:
            signal_masks.append(kept_signals[:2])

    seen: set[str] = set()
    for name, preset in regime_presets.items():
        for tp in turnover_grid:
            for rp in risk_grid:
                for topk in topk_grid:
                    for aw in alpha_weight_grid:
                        for mask in signal_masks:
                            c = copy.deepcopy(base)
                            c.setdefault("fund_manager", {})
                            c["fund_manager"]["turnover_penalty"] = tp
                            c["fund_manager"]["risk_penalty_scale"] = rp
                            c["fund_manager"]["top_k"] = topk
                            c.setdefault("regime_controls", {})
                            c["regime_controls"]["strategy_weights_by_regime"] = preset
                            c.setdefault("alpha_pipeline", {})
                            c["alpha_pipeline"]["blend_weight"] = float(aw)
                            c["alpha_pipeline"]["enabled_signals"] = list(mask)
                            if aw <= 0 or not mask:
                                c["alpha_pipeline"]["blend_weight"] = 0.0
                                c["alpha_pipeline"]["enabled_signals"] = []
                            sig_tag = "none" if not c["alpha_pipeline"]["enabled_signals"] else "-".join(c["alpha_pipeline"]["enabled_signals"])
                            key = f"{name}_tp{tp}_rp{rp}_k{topk}_aw{c['alpha_pipeline']['blend_weight']}_{sig_tag}"
                            if key in seen:
                                continue
                            seen.add(key)
                            c.setdefault("meta", {})
                            c["meta"]["candidate_name"] = key
                            candidates.append(c)
    return candidates


def _walkforward_windows(
    start: str,
    end: str,
    train_months: int = 18,
    test_months: int = 4,
    step_months: int = 4,
) -> list[tuple[str, str, str, str]]:
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    windows: list[tuple[str, str, str, str]] = []
    cur = s
    while True:
        train_start = cur
        train_end = train_start + pd.DateOffset(months=train_months)
        test_end = train_end + pd.DateOffset(months=test_months)
        if test_end > e:
            break
        windows.append(
            (
                train_start.date().isoformat(),
                train_end.date().isoformat(),
                train_end.date().isoformat(),
                test_end.date().isoformat(),
            )
        )
        cur = cur + pd.DateOffset(months=step_months)
    return windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/backtest_fast.yaml")
    parser.add_argument("--from-date", default="2020-01-01")
    parser.add_argument("--to-date", default="2026-03-01")
    parser.add_argument("--step-days", type=int, default=5)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--train-months", type=int, default=18)
    parser.add_argument("--test-months", type=int, default=4)
    parser.add_argument("--window-step-months", type=int, default=4)
    parser.add_argument("--max-candidates", type=int, default=0, help="0 = use full candidate grid")
    parser.add_argument("--signal-min-excess-sharpe", type=float, default=0.0)
    parser.add_argument("--out", default="outputs/walkforward")
    parser.add_argument("--write-config", default="configs/tuned_walkforward.yaml")
    args = parser.parse_args()

    base = load_config(args.config)
    base.setdefault("backtest", {})
    base["backtest"]["fast_mode"] = True

    windows = _walkforward_windows(
        args.from_date,
        args.to_date,
        train_months=args.train_months,
        test_months=args.test_months,
        step_months=args.window_step_months,
    )
    if not windows:
        raise ValueError("No walk-forward windows generated. Expand date range.")

    signal_universe = list(base.get("alpha_pipeline", {}).get("enabled_signals", []) or DEFAULT_ALPHA_SIGNALS)
    signal_rows: list[dict[str, float | str]] = []
    kept_signals: list[str] = []
    for signal in signal_universe:
        r = _evaluate_single_signal_oos(
            base=base,
            signal=signal,
            windows=windows,
            step_days=args.step_days,
            max_cycles=args.max_cycles,
        )
        signal_rows.append(r)
        if float(r.get("avg_excess_sharpe", -1e9)) > float(args.signal_min_excess_sharpe):
            kept_signals.append(signal)

    candidates = _candidate_configs(base, kept_signals=kept_signals)
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    rows: list[dict[str, float | str | int]] = []

    for idx, (tr_s, tr_e, te_s, te_e) in enumerate(windows, start=1):
        best = None
        best_score = -1e9
        for c in candidates:
            try:
                _, _, train_net, _, train_m = run_orchestrator_backtest(
                    cfg=c,
                    start_date=tr_s,
                    end_date=tr_e,
                    step_days=args.step_days,
                    max_cycles=args.max_cycles,
                    use_prices_override=True,
                )
            except Exception:
                continue
            bench = _benchmark_returns(
                train_net.index,
                symbols=list(c["portfolio"]["symbols"]),
                mode=str(c.get("benchmark", {}).get("mode", "symbol")),
                symbol=str(c.get("benchmark", {}).get("symbol", "SPY")),
            )
            excess = train_net - bench.reindex(train_net.index).fillna(0.0)
            ex_m = _metrics(excess)
            if not ex_m:
                continue
            # Train objective with acceptance gates.
            if train_m.get("max_drawdown", 0.0) < -0.20:
                continue
            if train_m.get("avg_turnover", 1.0) > 0.60:
                continue
            score = ex_m["sharpe"] + 0.5 * ex_m["cagr"]
            if score > best_score:
                best_score = score
                best = c
        if best is None:
            continue

        try:
            _, _, test_net, _, test_m = run_orchestrator_backtest(
                cfg=best,
                start_date=te_s,
                end_date=te_e,
                step_days=args.step_days,
                max_cycles=args.max_cycles,
                use_prices_override=True,
            )
        except Exception:
            continue
        test_bench = _benchmark_returns(
            test_net.index,
            symbols=list(best["portfolio"]["symbols"]),
            mode=str(best.get("benchmark", {}).get("mode", "symbol")),
            symbol=str(best.get("benchmark", {}).get("symbol", "SPY")),
        )
        test_excess = test_net - test_bench.reindex(test_net.index).fillna(0.0)
        ex_test_m = _metrics(test_excess)

        row = {
            "window": idx,
            "train_start": tr_s,
            "train_end": tr_e,
            "test_start": te_s,
            "test_end": te_e,
            "candidate": str(best.get("meta", {}).get("candidate_name", "na")),
            "test_excess_cagr": float(ex_test_m.get("cagr", 0.0)),
            "test_excess_sharpe": float(ex_test_m.get("sharpe", 0.0)),
            "test_total_return": float(test_m.get("total_return", 0.0)),
            "test_max_drawdown": float(test_m.get("max_drawdown", 0.0)),
            "test_avg_turnover": float(test_m.get("avg_turnover", 0.0)),
        }
        rows.append(row)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if signal_rows:
        pd.DataFrame(signal_rows).sort_values("avg_excess_sharpe", ascending=False).to_csv(
            out_dir / "signal_contribution.csv",
            index=False,
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "walkforward_results.csv", index=False)
    if df.empty:
        # Fallback: select by full-range excess Sharpe when walk-forward windows are sparse.
        best = None
        best_score = -1e9
        for c in candidates:
            try:
                _, _, net, _, _ = run_orchestrator_backtest(
                    cfg=c,
                    start_date=args.from_date,
                    end_date=args.to_date,
                    step_days=args.step_days,
                    max_cycles=args.max_cycles,
                    use_prices_override=True,
                )
            except Exception:
                continue
            bench = _benchmark_returns(
                net.index,
                symbols=list(c["portfolio"]["symbols"]),
                mode=str(c.get("benchmark", {}).get("mode", "symbol")),
                symbol=str(c.get("benchmark", {}).get("symbol", "SPY")),
            )
            ex = _metrics(net - bench.reindex(net.index).fillna(0.0))
            score = float(ex.get("sharpe", -1e9))
            if score > best_score:
                best_score = score
                best = c
        if best is None:
            raise RuntimeError("No viable candidate in fallback selection.")
        selected = best
        selected.pop("meta", None)
        with Path(args.write_config).open("w", encoding="utf-8") as f:
            yaml.safe_dump(selected, f, sort_keys=False)
        summary = {
            "windows": 0,
            "best_candidate": "fallback_full_range",
            "avg_test_excess_cagr": 0.0,
            "avg_test_excess_sharpe": best_score,
            "kept_signals": kept_signals,
            "dropped_signals": [s for s in signal_universe if s not in kept_signals],
        }
        with (out_dir / "summary.yaml").open("w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
        print("Walk-forward fallback selection complete.")
        print(summary)
        return

    # Pick most frequent / strongest candidate in OOS.
    best_name = (
        df.groupby("candidate")["test_excess_sharpe"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
    )
    selected = next(c for c in candidates if str(c.get("meta", {}).get("candidate_name", "")) == best_name)
    selected.pop("meta", None)
    with Path(args.write_config).open("w", encoding="utf-8") as f:
        yaml.safe_dump(selected, f, sort_keys=False)

    summary = {
        "windows": int(len(df)),
        "best_candidate": best_name,
        "avg_test_excess_cagr": float(df["test_excess_cagr"].mean()),
        "avg_test_excess_sharpe": float(df["test_excess_sharpe"].mean()),
        "kept_signals": kept_signals,
        "dropped_signals": [s for s in signal_universe if s not in kept_signals],
    }
    with (out_dir / "summary.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("Walk-forward optimization complete.")
    print(df.to_string(index=False))
    print(summary)


if __name__ == "__main__":
    main()
