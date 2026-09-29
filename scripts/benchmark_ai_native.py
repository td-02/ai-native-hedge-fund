"""Benchmark the AI-native (LLM-backed) decision path against the deterministic fallback.

The shipped configs never exercise the LLM layers end to end: ``backtest.fast_mode``
bypasses the research council, ``ai_native_v2.use_llm_forecasts`` is off everywhere, and
the CLI's default service pipeline skips the council. This script runs the project's own
``backtest_ai_native_v2.run_compare`` benchmark with the per-ticker forecasts genuinely
served by a local Ollama model, next to the deterministic fallback, and then runs one live
decision cycle through the full orchestrator with the LangGraph research council ON and OFF.

Every LLM call is probed. The LLM variant is only reported when the calls were really
answered by the model (no provider fallback, no empty JSON); otherwise the script exits
non-zero, so the numbers can never silently come from the disabled path.

Usage:
    python scripts/benchmark_ai_native.py --config configs/ai_native_llm.yaml \
        --from-date 2020-01-01 --to-date 2026-03-01 --step-days 5 --max-cycles 60 \
        --out outputs/ai_native_llm_benchmark
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import backtest_ai_native_v2 as ai_v2  # noqa: E402
import free_fund.ai_forecast_calibration as afc  # noqa: E402
import llm_router  # noqa: E402
from free_fund.config import load_config  # noqa: E402
from free_fund.orchestrator import CentralizedHedgeFundSystem  # noqa: E402
from free_fund.research_council import LLMResearchCouncil  # noqa: E402

LLM_VARIANT = "ai_native_v2_llm"
DET_VARIANT = "ai_native_v2_deterministic"
COUNCIL_ROLES = ("researcher", "news_analyst", "peer_reviewer")

# dataviz reference palette (light mode); slot order blue, orange, aqua, yellow is the
# validated adjacent-pair order, so the lines stay distinguishable under colour-vision deficiency.
SERIES = {
    LLM_VARIANT: ("#2a78d6", "AI-native v2 (LLM forecasts)", "v2 LLM"),
    DET_VARIANT: ("#eb6834", "AI-native v2 (deterministic fallback)", "v2 fallback"),
    "baseline": ("#1baf7a", "Baseline orchestrator", "Baseline"),
    "benchmark_spy": ("#eda100", "SPY", "SPY"),
}
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(s) - 1)
    return float(s[lo] + (s[hi] - s[lo]) * (pos - lo))


def summarize_calls(calls: list[dict]) -> dict:
    lat = [float(c["latency_ms"]) for c in calls]
    return {
        "calls": len(calls),
        "served_by_llm": int(sum(1 for c in calls if c["served_by_llm"])),
        "providers": dict(Counter(str(c.get("provider") or c.get("model") or "unknown") for c in calls)),
        "latency_ms": {
            "mean": float(statistics.fmean(lat)) if lat else 0.0,
            "p50": float(statistics.median(lat)) if lat else 0.0,
            "p95": _quantile(lat, 0.95),
            "max": float(max(lat)) if lat else 0.0,
        },
    }


class LLMProbe:
    """Records what actually served every ``llm_router.llm_chat`` call."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def install(self) -> None:
        original = llm_router.llm_chat
        probe = self

        def wrapped(prompt: str, system: str = "", json_mode: bool = False, timeout: int = 15) -> str:
            t0 = time.perf_counter()
            text = original(prompt, system=system, json_mode=json_mode, timeout=timeout)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            provider = str(llm_router._LAST_PROVIDER_USED)
            served = provider != "none" and text.strip() not in ("", "{}")
            probe.calls.append(
                {
                    "provider": provider,
                    "latency_ms": latency_ms,
                    "served_by_llm": served,
                    "response_chars": len(text),
                }
            )
            return text

        llm_router.llm_chat = wrapped
        # ai_forecast_calibration bound the function name at import time; patch that binding too.
        afc.llm_chat = wrapped


class CouncilProbe:
    """Records every research-council role call and whether the LLM answered it."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def install(self) -> None:
        original = LLMResearchCouncil._ask
        probe = self

        def wrapped(council, role, symbol, tool_context, notes=""):
            t0 = time.perf_counter()
            out = original(council, role, symbol, tool_context, notes=notes)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            summary = str(out.get("summary", ""))
            served = bool(council.enable) and summary != "disabled" and not summary.startswith("fallback:")
            probe.calls.append(
                {
                    "role": str(role),
                    "symbol": str(symbol),
                    "model": council._resolved_model,
                    "latency_ms": latency_ms,
                    "conviction": float(out.get("conviction", 0.0)),
                    "summary": summary,
                    "served_by_llm": served,
                }
            )
            return out

        LLMResearchCouncil._ask = wrapped


def ollama_preflight(generate_url: str, model: str) -> dict:
    base = generate_url.replace("/api/generate", "").rstrip("/")
    try:
        tags = requests.get(f"{base}/api/tags", timeout=5)
        tags.raise_for_status()
    except requests.RequestException as exc:
        raise SystemExit(f"Ollama is not reachable at {base}: {exc}. Start it with `ollama serve`.") from exc
    models = [str(m.get("name", "")) for m in (tags.json().get("models", []) or [])]
    if model not in models:
        raise SystemExit(f"Model {model!r} is not installed in Ollama (found {models}). Run `ollama pull {model}`.")
    version = str(requests.get(f"{base}/api/version", timeout=5).json().get("version", "unknown"))
    # llm_router.py resolves its Ollama target from the environment; make it match the config.
    os.environ["OLLAMA_BASE_URL"] = base
    os.environ["OLLAMA_MODEL"] = model
    return {"base_url": base, "model": model, "ollama_version": version, "installed_models": models}


def warm_up(timeout: int) -> dict:
    t0 = time.perf_counter()
    text = llm_router.llm_chat('Reply with the JSON object {"ok": true}.', system="", json_mode=True, timeout=timeout)
    elapsed = time.perf_counter() - t0
    if llm_router._LAST_PROVIDER_USED != "ollama" or text.strip() in ("", "{}"):
        raise SystemExit(
            f"LLM warm-up failed: provider={llm_router._LAST_PROVIDER_USED!r} response={text[:120]!r}"
        )
    return {"warm_up_sec": elapsed, "response": text}


def hardware_info() -> dict:
    def _sysctl(key: str) -> str:
        try:
            return subprocess.run(["sysctl", "-n", key], capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return ""

    mem = _sysctl("hw.memsize")
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _sysctl("machdep.cpu.brand_string") or platform.processor(),
        "memory_gb": round(int(mem) / 1024**3, 1) if mem.isdigit() else None,
        "python": platform.python_version(),
    }


def run_v2_variant(cfg: dict, use_llm: bool, args: argparse.Namespace, probe: LLMProbe) -> tuple[pd.DataFrame, dict, dict]:
    cfg_v = copy.deepcopy(cfg)
    v2 = cfg_v.setdefault("ai_native_v2", {})
    v2["enabled"] = True
    v2["use_llm_forecasts"] = bool(use_llm)
    cfg_v.setdefault("tracing", {})["enabled"] = False  # backtest cycles do not need TraceLM files

    sources: Counter = Counter()
    samples: list[dict] = []
    original = ai_v2.generate_ai_forecasts

    def observed(*a, **k):
        out = original(*a, **k)
        for ticker, fc in out.items():
            sources[str(fc.get("source", "unknown"))] += 1
            if len(samples) < 10:
                samples.append({"ticker": ticker, **{kk: vv for kk, vv in fc.items()}})
        return out

    ai_v2.generate_ai_forecasts = observed
    n0 = len(probe.calls)
    t0 = time.perf_counter()
    try:
        table, details = ai_v2.run_compare(
            cfg=cfg_v,
            start_date=args.from_date,
            end_date=args.to_date,
            step_days=int(args.step_days),
            max_cycles=int(args.max_cycles),
        )
    finally:
        ai_v2.generate_ai_forecasts = original
    elapsed = time.perf_counter() - t0
    usage = {
        "use_llm_forecasts": bool(use_llm),
        "elapsed_sec": elapsed,
        "forecast_sources": dict(sources),
        "forecast_samples": samples,
        **summarize_calls(probe.calls[n0:]),
    }
    return table, details, usage


def require_llm_served(usage: dict, n_expected: int, min_rate: float) -> None:
    calls, served = int(usage["calls"]), int(usage["served_by_llm"])
    llm_forecasts = int(usage["forecast_sources"].get("llm", 0))
    problems: list[str] = []
    if calls != n_expected:
        problems.append(f"expected {n_expected} LLM calls (cycles x symbols) but saw {calls}")
    if calls == 0 or served < calls * min_rate:
        problems.append(f"only {served}/{calls} calls were answered by the LLM")
    if llm_forecasts != served:
        problems.append(f"{llm_forecasts} forecasts are tagged source=llm but {served} calls were served")
    if set(usage["providers"]) - {"ollama"}:
        problems.append(f"a provider other than ollama answered: {usage['providers']}")
    if problems:
        raise SystemExit("AI-native path was NOT genuinely used: " + "; ".join(problems))


def run_live_cycle(cfg: dict, council_probe: CouncilProbe) -> dict:
    live = copy.deepcopy(cfg)
    live.setdefault("backtest", {})["fast_mode"] = False
    live["runtime"] = {"pipeline_mode": False}
    live.setdefault("tracing", {})["enabled"] = True
    live["execution"] = {
        "broker": "stub",
        "primary_broker": "stub",
        "backup_brokers": ["stub"],
        "market_mode": str(cfg.get("execution", {}).get("market_mode", "us")),
    }
    dq = live.setdefault("data_quality", {})
    dq["max_staleness_minutes"] = max(int(dq.get("max_staleness_minutes", 15)), 60 * 24 * 4)
    # The ResearchAgent overlay is exercised by `ainhf run`; keep it off here so the council is
    # the only LLM component that differs between the ON and OFF cycles.
    live.setdefault("agent", {})["enable_llm_research"] = False

    blend = float(cfg.get("research_council", {}).get("blend_weight", 0.10)) or 0.10
    on = copy.deepcopy(live)
    on.setdefault("research_council", {}).update({"enabled": True, "blend_weight": blend})
    off = copy.deepcopy(live)
    off.setdefault("research_council", {}).update({"enabled": False, "blend_weight": 0.0})

    trace_dir = ROOT / str(cfg.get("system", {}).get("output_dir", "outputs")) / "traces"
    before = {p.name for p in trace_dir.glob("trace_*.json")} if trace_dir.exists() else set()

    n0 = len(council_probe.calls)
    t0 = time.perf_counter()
    decision_on = CentralizedHedgeFundSystem(on).run_cycle(execute=False)
    on_sec = time.perf_counter() - t0
    council_calls = council_probe.calls[n0:]

    t0 = time.perf_counter()
    decision_off = CentralizedHedgeFundSystem(off).run_cycle(execute=False)
    off_sec = time.perf_counter() - t0
    after = {p.name for p in trace_dir.glob("trace_*.json")} if trace_dir.exists() else set()

    per_symbol: dict[str, dict] = {}
    for s in list(decision_on.symbols):
        roles = {c["role"]: c for c in council_calls if c["symbol"] == s}
        convs = [float(roles[r]["conviction"]) for r in COUNCIL_ROLES if r in roles]
        per_symbol[s] = {
            **{r: (float(roles[r]["conviction"]) if r in roles else None) for r in COUNCIL_ROLES},
            "council_score": float(sum(convs) / len(convs)) if convs else 0.0,
            "summaries": {r: roles[r]["summary"] for r in COUNCIL_ROLES if r in roles},
            "weight_council_on": float(decision_on.target_weights.get(s, 0.0)),
            "weight_council_off": float(decision_off.target_weights.get(s, 0.0)),
        }
    return {
        "council_calls": council_calls,
        "council_summary": summarize_calls(council_calls),
        "per_symbol": per_symbol,
        "decision_council_on": decision_on.to_dict(),
        "decision_council_off": decision_off.to_dict(),
        "elapsed_sec": {"council_on": on_sec, "council_off": off_sec},
        "new_trace_files": sorted(after - before),
    }


def build_equity_frame(details_llm: dict, details_det: dict | None, symbols: list[str], bench_mode: str, bench_symbol: str) -> pd.DataFrame:
    _, _, _, base_eq = details_llm["baseline"]
    _, _, _, llm_eq = details_llm["ai_native_v2"]
    idx = base_eq.index
    frame = {"baseline": base_eq, LLM_VARIANT: llm_eq.reindex(idx)}
    if details_det is not None:
        frame[DET_VARIANT] = details_det["ai_native_v2"][3].reindex(idx)
    spy = ai_v2._benchmark_series(index=idx, symbols=symbols, mode=bench_mode, symbol=bench_symbol)
    ew = ai_v2._benchmark_series(index=idx, symbols=symbols, mode="equal_weight", symbol=bench_symbol)
    frame["benchmark_spy"] = (1.0 + spy).cumprod()
    frame["benchmark_equal_weight"] = (1.0 + ew).cumprod()
    return pd.DataFrame(frame)


def plot_equity(eq: pd.DataFrame, out_png: Path, subtitle: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    plt.rcParams["font.family"] = "sans-serif"
    fig, ax = plt.subplots(figsize=(11, 5.4), dpi=160)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    keys = [k for k in SERIES if k in eq.columns]
    for key in keys:
        color, label, _ = SERIES[key]
        ax.plot(eq.index, eq[key], color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round", label=label)

    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.grid(axis="x", visible=False)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.2f}"))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.set_ylabel("Equity (start = 1.00)", color=INK2, fontsize=9)
    ax.set_title("AI-native v2 with real LLM forecasts vs the deterministic fallback", loc="left", color=INK, fontsize=12, fontweight="bold", pad=22)
    ax.text(0.0, 1.03, subtitle, transform=ax.transAxes, color=INK2, fontsize=9, va="bottom")

    # Direct end labels: nudge apart where lines converge and connect each with a hairline leader.
    span_days = max(30, (eq.index[-1] - eq.index[0]).days)
    x_end = eq.index[-1]
    x_lab = x_end + pd.Timedelta(days=int(span_days * 0.035))
    key_len = pd.Timedelta(days=int(span_days * 0.025))
    ends = sorted((float(eq[k].iloc[-1]), k) for k in keys)
    ymin, ymax = ax.get_ylim()
    min_gap = (ymax - ymin) * 0.065
    placed: list[float] = []
    for y, _ in ends:
        placed.append(y if not placed else max(y, placed[-1] + min_gap))
    for (y, k), yl in zip(ends, placed):
        color, _, short = SERIES[k]
        ax.plot([x_end, x_lab], [y, yl], color=AXIS, linewidth=1, zorder=1)
        ax.plot([x_lab, x_lab + key_len], [yl, yl], color=color, linewidth=2, solid_capstyle="round", zorder=2)
        ax.text(x_lab + key_len + pd.Timedelta(days=int(span_days * 0.01)), yl, f"{short} {y:.2f}", color=INK2, fontsize=8.5, va="center")
    ax.set_ylim(ymin, max(ymax, placed[-1] + min_gap * 0.8))
    ax.set_xlim(eq.index[0], x_end + pd.Timedelta(days=int(span_days * 0.30)))
    ax.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out_png, facecolor=SURFACE)
    plt.close(fig)


def _pct(x: float) -> str:
    return f"{100.0 * float(x):+.1f}%"


def render_snippet(meta: dict, table: pd.DataFrame, usage_llm: dict, usage_det: dict | None, live: dict | None) -> str:
    lines: list[str] = []
    hw = meta["hardware"]
    lines.append("### Setup")
    lines.append("")
    lines.append(f"- Model: `{meta['ollama']['model']}` served locally by Ollama {meta['ollama']['ollama_version']} on {hw['cpu']} ({hw['memory_gb']} GB).")
    lines.append(
        f"- Window: price data {meta['args']['from_date']} to {meta['args']['to_date']}; one decision every {meta['args']['step_days']} trading days, "
        f"{meta['cycles']} decision cycles from {meta.get('decision_start', '?')} to {meta.get('decision_end', '?')}; universe {', '.join(meta['symbols'])}."
    )
    lines.append(f"- Config: `{meta['args']['config']}` (LLM forecasts on, research council on, full orchestrator).")
    lines.append("")
    lines.append("### Backtest comparison")
    lines.append("")
    lines.append("| Variant | Sharpe | CAGR | Vol | Max DD | Total return | Avg turnover |")
    lines.append("|---|---|---|---|---|---|---|")
    names = {
        "baseline": "Baseline orchestrator (no v2 overlay)",
        LLM_VARIANT: "AI-native v2, LLM forecasts (Ollama)",
        DET_VARIANT: "AI-native v2, deterministic fallback",
        "benchmark_spy": "SPY",
        "benchmark_equal_weight": "Equal weight",
    }
    for _, row in table.iterrows():
        turnover = row.get("avg_turnover")
        turnover_txt = f"{float(turnover):.2f}" if pd.notna(turnover) else "n/a"
        lines.append(
            f"| {names.get(row['variant'], row['variant'])} | {float(row['sharpe']):.2f} | {_pct(row['cagr'])} | "
            f"{100.0 * float(row['annual_vol']):.1f}% | {100.0 * float(row['max_drawdown']):.1f}% | {_pct(row['total_return'])} | {turnover_txt} |"
        )
    lines.append("")
    lines.append("### LLM usage (proof the AI path ran)")
    lines.append("")
    lines.append("| Variant | LLM calls | Answered by LLM | Forecast source | Provider | Latency p50 / p95 | Wall clock |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, u in ((LLM_VARIANT, usage_llm), (DET_VARIANT, usage_det)):
        if u is None:
            continue
        lat = u["latency_ms"]
        srcs = ", ".join(f"{k}: {v}" for k, v in sorted(u["forecast_sources"].items()))
        prov = ", ".join(f"{k}: {v}" for k, v in sorted(u["providers"].items())) or "none"
        lines.append(
            f"| {names[label]} | {u['calls']} | {u['served_by_llm']} | {srcs} | {prov} | "
            f"{lat['p50'] / 1000.0:.2f} s / {lat['p95'] / 1000.0:.2f} s | {u['elapsed_sec'] / 60.0:.1f} min |"
        )
    if live is not None:
        lines.append("")
        lines.append("### Live decision cycle with the LLM research council")
        lines.append("")
        cs = live["council_summary"]
        lines.append(
            f"{cs['calls']} council calls (3 roles x {len(live['per_symbol'])} symbols), {cs['served_by_llm']} answered by the LLM, "
            f"latency p50 {cs['latency_ms']['p50'] / 1000.0:.2f} s, cycle wall clock {live['elapsed_sec']['council_on']:.0f} s with council vs {live['elapsed_sec']['council_off']:.0f} s without."
        )
        lines.append("")
        lines.append("| Symbol | Researcher | News analyst | Peer reviewer | Council score | Weight, council on | Weight, council off |")
        lines.append("|---|---|---|---|---|---|---|")
        order = [s for s in meta.get("symbols", []) if s in live["per_symbol"]] or list(live["per_symbol"])
        for s in order:
            r = live["per_symbol"][s]
            def _c(v):
                return "n/a" if v is None else f"{float(v):+.2f}"

            lines.append(
                f"| {s} | {_c(r['researcher'])} | {_c(r['news_analyst'])} | {_c(r['peer_reviewer'])} | {float(r['council_score']):+.3f} | "
                f"{float(r['weight_council_on']):.3f} | {float(r['weight_council_off']):.3f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/ai_native_llm.yaml")
    p.add_argument("--from-date", default="2020-01-01")
    p.add_argument("--to-date", default="2026-03-01")
    p.add_argument("--step-days", type=int, default=5)
    p.add_argument("--max-cycles", type=int, default=60, help="0 = all available cycles")
    p.add_argument("--out", default="outputs/ai_native_llm_benchmark")
    p.add_argument("--min-llm-rate", type=float, default=0.99, help="Minimum share of forecast calls the LLM must answer.")
    p.add_argument("--skip-deterministic", action="store_true", help="Do not run the deterministic-fallback variant.")
    p.add_argument("--skip-live-cycle", action="store_true", help="Do not run the live council ON/OFF cycle.")
    p.add_argument("--live-cycle-only", action="store_true", help="Reuse the backtest results already in --out and only run the live cycle.")
    p.add_argument("--no-plot", action="store_true")
    p.add_argument("--media-png", default="outputs/media/ai_native_llm_benchmark_equity.png", help="Where to write the README chart.")
    args = p.parse_args()

    cfg = load_config(args.config)
    acfg = cfg.get("agent", {})
    model = str(acfg.get("ollama_model", "llama3.1:8b"))
    generate_url = str(acfg.get("ollama_url", "http://localhost:11434/api/generate"))
    symbols = list(cfg["portfolio"]["symbols"])
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    usage_path = out / "llm_usage.json"

    started = datetime.now(timezone.utc)
    ollama = ollama_preflight(generate_url, model)
    llm_probe = LLMProbe()
    llm_probe.install()
    council_probe = CouncilProbe()
    council_probe.install()
    warm = warm_up(int(cfg.get("ai_native_v2", {}).get("llm_timeout_seconds", 120)))
    llm_probe.calls.clear()
    print(f"[benchmark] Ollama {ollama['ollama_version']} model={model} warm-up {warm['warm_up_sec']:.1f}s -> {warm['response']}")

    def _write_evidence(meta: dict, table: pd.DataFrame, usage_llm: dict, usage_det: dict | None, live: dict | None) -> str:
        meta["finished_utc"] = datetime.now(timezone.utc).isoformat()
        usage_path.write_text(json.dumps(meta, indent=2, sort_keys=True, default=str), encoding="utf-8")
        snippet = render_snippet(meta, table, usage_llm, usage_det, live)
        (out / "README_snippet.md").write_text(snippet, encoding="utf-8")
        return snippet

    if args.live_cycle_only:
        meta = json.loads(usage_path.read_text(encoding="utf-8"))
        table = pd.read_csv(out / "comparison_metrics.csv")
        usage_llm = meta["usage"][LLM_VARIANT]
        usage_det = meta["usage"].get(DET_VARIANT)
        meta["ollama"] = ollama
        print(f"[benchmark] reusing backtest results from {out} ({int(meta['cycles'])} cycles)")
    else:
        print("[benchmark] running AI-native v2 with LLM forecasts ...")
        table_llm, details_llm, usage_llm = run_v2_variant(cfg, use_llm=True, args=args, probe=llm_probe)
        cycles = int(table_llm.loc[table_llm["variant"] == "ai_native_v2", "cycles"].iloc[0])
        require_llm_served(usage_llm, n_expected=cycles * len(symbols), min_rate=float(args.min_llm_rate))
        print(f"[benchmark]   {usage_llm['served_by_llm']}/{usage_llm['calls']} forecast calls answered by the LLM in {usage_llm['elapsed_sec'] / 60.0:.1f} min")

        usage_det: dict | None = None
        details_det: dict | None = None
        table_det: pd.DataFrame | None = None
        if not args.skip_deterministic:
            print("[benchmark] running AI-native v2 with the deterministic fallback ...")
            table_det, details_det, usage_det = run_v2_variant(cfg, use_llm=False, args=args, probe=llm_probe)
            if usage_det["calls"] != 0:
                raise SystemExit("deterministic variant made LLM calls; the toggle is broken")

        rows = []
        rows.append({**table_llm[table_llm["variant"] == "baseline"].iloc[0].to_dict(), "variant": "baseline"})
        rows.append({**table_llm[table_llm["variant"] == "ai_native_v2"].iloc[0].to_dict(), "variant": LLM_VARIANT})
        if table_det is not None:
            base_llm = table_llm[table_llm["variant"] == "baseline"].iloc[0]
            base_det = table_det[table_det["variant"] == "baseline"].iloc[0]
            gap = abs(float(base_llm["sharpe"]) - float(base_det["sharpe"]))
            if gap > 1e-3:
                print(f"[benchmark] WARNING: baseline Sharpe differs by {gap:.4f} between the two runs")
            elif gap > 0:
                print(f"[benchmark] note: baseline Sharpe differs by {gap:.2e} between runs (live yfinance feed drift, not the orchestrator)")
            rows.append({**table_det[table_det["variant"] == "ai_native_v2"].iloc[0].to_dict(), "variant": DET_VARIANT})
        for bench in ("benchmark_spy", "benchmark_equal_weight"):
            rows.append({**table_llm[table_llm["variant"] == bench].iloc[0].to_dict(), "variant": bench})
        table = pd.DataFrame(rows)
        cols = ["variant", "sharpe", "cagr", "annual_vol", "max_drawdown", "total_return", "avg_turnover", "cycles"]
        table = table[[c for c in cols if c in table.columns]]
        table.to_csv(out / "comparison_metrics.csv", index=False)

        b_w, b_g, b_n, b_e = details_llm["baseline"]
        ai_v2._save_run(out, "baseline", b_w, b_g, b_n, b_e, rows[0])
        v_w, v_g, v_n, v_e = details_llm["ai_native_v2"]
        ai_v2._save_run(out, LLM_VARIANT, v_w, v_g, v_n, v_e, rows[1])
        if details_det is not None:
            d_w, d_g, d_n, d_e = details_det["ai_native_v2"]
            ai_v2._save_run(out, DET_VARIANT, d_w, d_g, d_n, d_e, rows[2])

        bench_cfg = cfg.get("benchmark", {})
        eq = build_equity_frame(details_llm, details_det, symbols, str(bench_cfg.get("mode", "symbol")), str(bench_cfg.get("symbol", symbols[0])))
        eq.to_csv(out / "equity_curves.csv", index=True)

        meta = {
            "started_utc": started.isoformat(),
            "args": vars(args),
            "symbols": symbols,
            "cycles": cycles,
            "decision_start": str(eq.index[0].date()),
            "decision_end": str(eq.index[-1].date()),
            "ollama": ollama,
            "warm_up": warm,
            "hardware": hardware_info(),
            "llm_router_provider_stats": llm_router.get_provider_stats(),
            "usage": {LLM_VARIANT: usage_llm, DET_VARIANT: usage_det},
        }
        _write_evidence(meta, table, usage_llm, usage_det, None)

        if not args.no_plot:
            try:
                subtitle = f"{model} via Ollama, {cycles} decision cycles from {meta['decision_start']} to {meta['decision_end']}, universe {', '.join(symbols)}"
                png = ROOT / str(args.media_png)
                png.parent.mkdir(parents=True, exist_ok=True)
                plot_equity(eq, png, subtitle)
                plot_equity(eq, out / "equity_curves.png", subtitle)
                print(f"[benchmark] chart written to {png.relative_to(ROOT)}")
            except Exception as exc:  # plotting is optional
                print(f"[benchmark] chart skipped: {exc}")

    live: dict | None = None
    if not args.skip_live_cycle:
        print("[benchmark] running one live decision cycle with the research council ON, then OFF ...")
        try:
            live = run_live_cycle(cfg, council_probe)
            (out / "council_live_cycle.json").write_text(json.dumps(live, indent=2, sort_keys=True, default=str), encoding="utf-8")
            cs = live["council_summary"]
            print(f"[benchmark]   {cs['served_by_llm']}/{cs['calls']} council calls answered by the LLM")
        except Exception as exc:  # the backtest evidence is already on disk; report and carry on
            print(f"[benchmark] live cycle failed: {exc!r}")
            traceback.print_exc()
            meta["live_cycle_error"] = repr(exc)
    elif (out / "council_live_cycle.json").exists():
        live = json.loads((out / "council_live_cycle.json").read_text(encoding="utf-8"))

    snippet = _write_evidence(meta, table, usage_llm, usage_det, live)
    print("AI-native LLM benchmark complete.")
    print(table.to_string(index=False))
    print(snippet)


if __name__ == "__main__":
    main()
