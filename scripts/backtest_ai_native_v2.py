from __future__ import annotations

import argparse
import copy
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from free_fund.ai_forecast_calibration import AIForecastCalibrator, generate_ai_forecasts
from free_fund.benchmark_relative_optimizer import optimize_benchmark_relative_weights
from free_fund.config import load_config
from free_fund.data import download_close_prices
from free_fund.meta_router import MetaAgentRouter
from free_fund.orchestrator import CentralizedHedgeFundSystem


def _metrics(returns: pd.Series) -> dict[str, float]:
    r = returns.dropna()
    if r.empty:
        return {}
    eq = (1.0 + r).cumprod()
    vol = float(r.std(ddof=0) * math.sqrt(252))
    sharpe = float((r.mean() * 252) / (vol + 1e-12))
    drawdown = eq / eq.cummax() - 1.0
    cagr = float(eq.iloc[-1] ** (252 / len(r)) - 1.0)
    return {
        "total_return": float(eq.iloc[-1] - 1.0),
        "cagr": cagr,
        "annual_vol": vol,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
    }


def _resolve_regime(window: pd.DataFrame) -> str:
    ret = window.pct_change().dropna()
    if len(ret) < 30:
        return "sideways"
    mkt = ret.mean(axis=1)
    ann_vol = float(mkt.std(ddof=0) * np.sqrt(252))
    trend = float((window.iloc[-1] / window.iloc[0]).mean() - 1.0)
    if ann_vol > 0.28:
        return "crisis"
    if trend > 0.04 and ann_vol < 0.22:
        return "bull"
    if trend < -0.03:
        return "bear"
    return "sideways"


def _benchmark_series(index: pd.DatetimeIndex, symbols: list[str], mode: str, symbol: str) -> pd.Series:
    start = str(index.min().date())
    end = str((index.max() + pd.Timedelta(days=1)).date())
    if mode == "equal_weight":
        data = yf.download(symbols, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        r = data.pct_change().fillna(0.0)
        return r.reindex(index).fillna(0.0).mean(axis=1)

    data = yf.download(symbol, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0]
    r = pd.Series(data).pct_change().fillna(0.0)
    return r.reindex(index).fillna(0.0)


def _save_run(out_dir: Path, name: str, weights: pd.DataFrame, gross: pd.Series, net: pd.Series, eq: pd.Series, metrics: dict[str, float]) -> None:
    d = out_dir / name
    d.mkdir(parents=True, exist_ok=True)
    weights.to_csv(d / "weights.csv")
    gross.to_csv(d / "gross_returns.csv", header=True)
    net.to_csv(d / "net_returns.csv", header=True)
    eq.to_csv(d / "equity_curve.csv", header=["equity"])
    pd.DataFrame([metrics]).to_csv(d / "metrics.csv", index=False)


def run_compare(cfg: dict, start_date: str, end_date: str | None, step_days: int, max_cycles: int) -> tuple[pd.DataFrame, dict]:
    symbols = list(cfg["portfolio"]["symbols"])
    lookback = int(cfg["portfolio"].get("lookback_days", 126))
    prices = download_close_prices(symbols=symbols, start_date=start_date, end_date=end_date)
    rets = prices.pct_change().fillna(0.0)

    rows = list(range(lookback, len(prices) - 1, max(1, int(step_days))))
    if max_cycles > 0:
        rows = rows[:max_cycles]
    if not rows:
        raise ValueError("No decision rows available")

    cfg_base = copy.deepcopy(cfg)
    cfg_base.setdefault("backtest", {})
    cfg_base["backtest"]["fast_mode"] = True
    cfg_base.setdefault("data_quality", {})
    cfg_base["data_quality"]["max_staleness_minutes"] = 60 * 24 * 365 * 20
    cfg_base["portfolio"]["start_date"] = start_date
    cfg_base["portfolio"]["end_date"] = None

    cfg_v2 = copy.deepcopy(cfg_base)
    v2cfg = cfg_v2.get("ai_native_v2", {})
    use_ai_native_v2 = bool(v2cfg.get("enabled", False))

    baseline = CentralizedHedgeFundSystem(cfg_base)
    v2_system = CentralizedHedgeFundSystem(cfg_v2)
    calibrator = AIForecastCalibrator(
        decay=float(v2cfg.get("calibration_decay", 0.95)),
        min_history=int(v2cfg.get("calibration_min_history", 20)),
    )
    router = MetaAgentRouter(config=v2cfg)

    bench_cfg = cfg.get("benchmark", {})
    bench_mode = str(bench_cfg.get("mode", "symbol"))
    bench_symbol = str(bench_cfg.get("symbol", symbols[0]))

    b_weights = []
    v2_weights = []
    b_ret = []
    v2_ret = []
    idx = []

    prev_b = pd.Series(0.0, index=symbols)
    prev_v = pd.Series(0.0, index=symbols)
    prev_forecasts: dict[str, dict[str, float | str]] = {}
    rel_perf_hist: list[float] = []

    for i in rows:
        dt = prices.index[i]
        base_decision = baseline.run_cycle(execute=False, prices_override=prices.iloc[: i + 1])
        v2_decision = v2_system.run_cycle(execute=False, prices_override=prices.iloc[: i + 1])

        w_b = pd.Series(base_decision.target_weights).reindex(symbols).fillna(0.0)
        w_v = pd.Series(v2_decision.target_weights).reindex(symbols).fillna(0.0)

        window = prices.iloc[: i + 1].tail(lookback)
        regime = _resolve_regime(window)
        routing = router.route(regime=regime, strategy_names=list(cfg["strategies"]["weights"].keys()), quality_scores=None)

        forecasts: dict[str, dict[str, float | str]] = {}
        w_v_opt = w_v.copy()
        diag: dict[str, float] = {"active_alpha": 0.0, "tracking_error": 0.0, "turnover": 0.0, "objective": 0.0}
        if use_ai_native_v2:
            forecasts = generate_ai_forecasts(
                prices_window=window,
                tickers=symbols,
                calibrator=calibrator,
                use_llm=bool(v2cfg.get("use_llm_forecasts", False)),
                timeout=int(v2cfg.get("llm_timeout_seconds", 8)),
            )

            bench_hist = _benchmark_series(index=window.index, symbols=symbols, mode=bench_mode, symbol=bench_symbol)
            w_v_opt, diag = optimize_benchmark_relative_weights(
                base_weights=w_v,
                alpha_views=forecasts,
                returns_window=window.pct_change().dropna().reindex(columns=symbols).fillna(0.0),
                benchmark_returns=bench_hist.pct_change().fillna(0.0) if "Close" in str(type(bench_hist)) else bench_hist,
                max_weight=float(cfg["portfolio"].get("max_weight", 0.30)),
                gross_limit=float(cfg["portfolio"].get("gross_limit", 1.0)) * float(routing.gross_scale),
                net_limit=float(cfg["risk_hard_limits"].get("net_limit", 0.30)),
                max_turnover=float(v2cfg.get("max_turnover_per_cycle", 0.10)),
                alpha_tilt_strength=float(v2cfg.get("alpha_tilt_strength", 0.15)),
                bab_tilt_strength=float(v2cfg.get("bab_tilt_strength", 0.10)),
                uncertainty_penalty=float(v2cfg.get("uncertainty_penalty", 0.50)),
                tracking_error_penalty=float(v2cfg.get("tracking_error_penalty", 1.00)),
            )
            # No-harm guard: if recent live ablation underperforms, fall back to baseline policy.
            if len(rel_perf_hist) >= 10 and float(np.mean(rel_perf_hist[-10:])) < 0.0:
                w_v_opt = w_v.copy()
            elif float(diag.get("objective", 0.0)) <= 0.0:
                w_v_opt = w_v.copy()

        rb = float((w_b - prev_b).abs().sum())
        rv = float((w_v_opt - prev_v).abs().sum())
        next_r = rets.iloc[i + 1]
        b_ret.append(float((w_b * next_r).sum()))
        v2_ret.append(float((w_v_opt * next_r).sum()))
        rel_perf_hist.append(float((w_v_opt * next_r).sum() - (w_b * next_r).sum()))
        b_weights.append(w_b)
        v2_weights.append(w_v_opt)
        idx.append(prices.index[i + 1])
        prev_b = w_b
        prev_v = w_v_opt

        # Update calibration with realized returns from previous forecast.
        if prev_forecasts and use_ai_native_v2:
            for t in symbols:
                try:
                    calibrator.update(
                        ticker=t,
                        pred=float(prev_forecasts.get(t, {}).get("expected_return", 0.0)),
                        realized=float(next_r.get(t, 0.0)),
                        confidence=float(prev_forecasts.get(t, {}).get("confidence", 0.4)),
                    )
                except Exception:
                    pass
        prev_forecasts = forecasts

    b_wdf = pd.DataFrame(b_weights, index=idx)
    v_wdf = pd.DataFrame(v2_weights, index=idx)
    b_gross = pd.Series(b_ret, index=idx, name="returns")
    v_gross = pd.Series(v2_ret, index=idx, name="returns")

    fee = float(cfg["costs"]["transaction_cost_bps"]) / 10000.0
    slip = float(cfg["costs"]["slippage_bps"]) / 10000.0
    b_turn = (b_wdf - b_wdf.shift(1).fillna(0.0)).abs().sum(axis=1)
    v_turn = (v_wdf - v_wdf.shift(1).fillna(0.0)).abs().sum(axis=1)
    b_net = b_gross - b_turn * (fee + slip)
    v_net = v_gross - v_turn * (fee + slip)

    bench = _benchmark_series(index=b_net.index, symbols=symbols, mode=bench_mode, symbol=bench_symbol)
    ew = _benchmark_series(index=b_net.index, symbols=symbols, mode="equal_weight", symbol=bench_symbol)

    summary = {
        "baseline": _metrics(b_net),
        "ai_native_v2": _metrics(v_net),
        "benchmark_spy": _metrics(bench),
        "benchmark_equal_weight": _metrics(ew),
        "baseline": {**_metrics(b_net), "avg_turnover": float(b_turn.mean()), "cycles": float(len(b_net))},
        "ai_native_v2": {**_metrics(v_net), "avg_turnover": float(v_turn.mean()), "cycles": float(len(v_net))},
    }

    table = pd.DataFrame(
        [
            {"variant": "baseline", **summary["baseline"]},
            {"variant": "ai_native_v2", **summary["ai_native_v2"]},
            {"variant": "benchmark_spy", **summary["benchmark_spy"]},
            {"variant": "benchmark_equal_weight", **summary["benchmark_equal_weight"]},
        ]
    )

    details = {
        "baseline": (b_wdf, b_gross, b_net, (1.0 + b_net).cumprod()),
        "ai_native_v2": (v_wdf, v_gross, v_net, (1.0 + v_net).cumprod()),
    }
    return table, details


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--from-date", default="2020-01-01")
    p.add_argument("--to-date", default=None)
    p.add_argument("--step-days", type=int, default=5)
    p.add_argument("--max-cycles", type=int, default=60)
    p.add_argument("--out", default="outputs/ai_native_v2_compare")
    args = p.parse_args()

    cfg = load_config(args.config)
    table, details = run_compare(
        cfg=cfg,
        start_date=args.from_date,
        end_date=args.to_date,
        step_days=int(args.step_days),
        max_cycles=int(args.max_cycles),
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison_metrics.csv", index=False)

    b_w, b_g, b_n, b_e = details["baseline"]
    v_w, v_g, v_n, v_e = details["ai_native_v2"]
    _save_run(out, "baseline", b_w, b_g, b_n, b_e, table[table["variant"] == "baseline"].iloc[0].drop("variant").to_dict())
    _save_run(out, "ai_native_v2", v_w, v_g, v_n, v_e, table[table["variant"] == "ai_native_v2"].iloc[0].drop("variant").to_dict())

    print("AI-native v2 comparison backtest complete.")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()

